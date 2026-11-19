from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from data import TimePrice, Trip


_ENV: gp.Env | None = None


#기본 그로비 환경
def gurobi_env() -> gp.Env:
    global _ENV
    if _ENV is None:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        try:
            env.start()
        except Exception:
            env.dispose()
            raise
        _ENV = env
    return _ENV


@dataclass
class MasterModel:
    model: gp.Model                                               # Gurobi 모델
    risk: dict[int, float]                                        # 잔해 i → 위험지수 r_i
    columns: list[Trip] = field(default_factory=list)             # 들어온 트립들 (k번째 트립 = k번째 변수)
    x_vars: list[gp.Var] = field(default_factory=list)            # 트립마다 변수 x_c (columns와 같은 순서)
    debris_rows: dict[int, gp.Constr] = field(default_factory=dict)  # 잔해 i → 잔해 행
    time_rows: dict[float, gp.Constr] = field(default_factory=dict)  # 출발 시각 t → 시간 행
    risk_row: gp.Constr | None = None                             # 2단계에서만: 위험지수 합 ≥ R* 행


@dataclass
class MIPResult:
    selected: list[Trip]   # 고른 트립들
    risk_sum: float        # R* = 고른 트립들의 위험지수 합
    dv_sum: float          # 고른 트립들의 총 ΔV [m/s]
    prop_sum: float        # 고른 트립들의 총 추진제 [kg]
    status: str


def new_master(risk: dict[int, float]) -> MasterModel:
    #처음 빈 모델 생성
    model = gp.Model("master", env=gurobi_env())
    model.ModelSense = GRB.MAXIMIZE
    m = MasterModel(model=model, risk=risk)
    for i in risk:
        m.debris_rows[i] = model.addLConstr(gp.LinExpr(), GRB.LESS_EQUAL, 1.0, name=f"debris[{i}]")
    return m


def add_column(m: MasterModel, trip: Trip) -> bool:
    #트립 하나 열로 추가
    #이미 있으면 false 출력
    if any(c.arc_ids == trip.arc_ids for c in m.columns):
        return False

    # 1. 이 트립의 출발 시각 s̄에 시간 행이 없으면 만든다.
    t = trip.s_bar
    if t not in m.time_rows:
        busy = [x for c, x in zip(m.columns, m.x_vars) if c.s_bar <= t < c.e_bar]
        m.time_rows[t] = m.model.addLConstr(gp.quicksum(busy), GRB.LESS_EQUAL, 1.0, name=f"time[{t:.0f}]")

    # 2. 이 트립이 들어갈 행: 들르는 잔해의 행 + 진행 중인 시각의 시간 행
    rows = [m.debris_rows[i] for i in trip.debris]
    rows += [row for tk, row in m.time_rows.items() if trip.s_bar <= tk < trip.e_bar]

    # 3. 변수 x_c를 만든다. 1단계: 목적 계수 = 위험지수 합, 위 행들에 계수 1
    #    2단계: 목적 계수 = −ΔV, 위험지수 행에도 계수 = 위험지수 합
    profit = sum(m.risk[i] for i in trip.debris)
    coeffs = [1.0] * len(rows)
    obj = profit
    if m.risk_row is not None: #2단계에선 risk row에도 추가
        rows.append(m.risk_row)
        coeffs.append(profit)
        obj = -trip.dv_total
    x = m.model.addVar(lb=0.0, obj=obj, column=gp.Column(coeffs, rows))
    m.columns.append(trip)
    m.x_vars.append(x)
    return True


def fix_risk(m: MasterModel, r_star: float) -> None:
    """
    2단계로 바꾼다: 위험지수 합 ≥ r_star 행을 넣고, 목적을 총 ΔV 최소로 바꾼다.
    """
    if m.risk_row is not None:
        raise ValueError("이미 2단계로 전환한 마스터입니다.")
    m.model.update()
    profits = [sum(m.risk[i] for i in c.debris) for c in m.columns]
    m.risk_row = m.model.addLConstr(gp.quicksum(p * x for p, x in zip(profits, m.x_vars)),
                                    GRB.GREATER_EQUAL, r_star, name="risk")
    for c, x in zip(m.columns, m.x_vars):
        x.Obj = -c.dv_total


def solve_lp(m: MasterModel) -> tuple[float, dict[int, float], dict[float, float]]:
    """
    마스터 LP 풀기
    """
    m.model.optimize()
    if m.model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"마스터 LP가 최적으로 풀리지 않았습니다: {m.model.Status}")
    alpha = {i: row.Pi for i, row in m.debris_rows.items()}
    mu = {t: row.Pi for t, row in m.time_rows.items()}
    return m.model.ObjVal, alpha, mu


def time_price(mu: dict[float, float]) -> TimePrice:
    """시간 가격 함수 U(t) = (t보다 이른 시간 행들의 가격 μ 합)을 만든다.

    입력: 출발 시각 t → μ
    출력: U(t) 함수. 트립의 시간 사용료 = U(ē) − U(s̄)
    """
    times = sorted(mu)
    cumsum = [0.0]                     # cumsum[k] = 앞에서 k개 시간 행의 μ 합
    for t in times:
        # <= 행의 dual은 비음수. 수치 오차로 생긴 작은 음수만 0으로 보정한다.
        if mu[t] < -1e-7:
            raise ValueError("시간 가격은 음수일 수 없습니다.")
        cumsum.append(cumsum[-1] + max(0.0, mu[t]))

    def U(t: float) -> float:
        k = bisect.bisect_left(times, t)   # t보다 이른 시간 행의 개수
        return cumsum[k]

    return U


def solve_mip(m: MasterModel, time_limit_s: float | None = None) -> MIPResult:

    m.model.update()
    mip = m.model.copy()
    xs = mip.getVars()
    for x in xs:
        x.VType = GRB.BINARY

    risk_expr = gp.quicksum(sum(m.risk[i] for i in c.debris) * x for c, x in zip(m.columns, xs))
    dv_expr = gp.quicksum(c.dv_total * x for c, x in zip(m.columns, xs))
    if m.risk_row is None:
        mip.setObjectiveN(risk_expr, index=0, priority=1, weight=1.0, abstol=0.0, reltol=0.0, name="risk")
        mip.setObjectiveN(dv_expr, index=1, priority=0, weight=-1.0, name="dv")
    else:
        # R* 미증명인 경우에도 2단계 LP/pricing과 동일한 목적을 유지한다.
        mip.setObjective(-dv_expr, GRB.MAXIMIZE)
    mip.Params.MIPGap = 0.0
    mip.Params.MIPGapAbs = 1e-9
    if time_limit_s is not None:
        mip.Params.TimeLimit = time_limit_s
    mip.optimize()

    if mip.SolCount == 0:
        status_code = mip.Status
        mip.dispose()
        raise RuntimeError(f"마스터 MIP의 실행 가능한 해를 찾지 못했습니다: {status_code}")

    selected = []
    if mip.SolCount > 0:
        selected = [c for c, x in zip(m.columns, xs) if x.X > 0.5]
    selected.sort(key=lambda c: c.s)
    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit"}.get(mip.Status, f"gurobi_status_{mip.Status}")
    mip.dispose()

    return MIPResult(selected=selected,
                     risk_sum=sum(m.risk[i] for c in selected for i in c.debris),
                     dv_sum=sum(c.dv_total for c in selected),
                     prop_sum=sum(c.prop for c in selected),
                     status=status)
