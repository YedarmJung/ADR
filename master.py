"""마스터 문제: 모은 트립(열) 중 어떤 것들을 쓸지 고른다. (plan 3.3)

    최대화   Σ_c p_c · x_c                         p_c = 트립 c가 들르는 잔해들의 위험지수 합
    제약     Σ_{c가 잔해 i를 들름} x_c ≤ 1          잔해 행: 잔해 i마다 하나 (가격 α_i)
             Σ_{s̄_c ≤ t < ē_c} x_c ≤ 1              시간 행: 출발 시각 t마다 하나 (가격 μ_t)
             x_c ≥ 0       (최종 MIP에서는 x_c ∈ {0, 1})
    x_c ≤ 1은 따로 두지 않는다. 잔해 행 때문에 저절로 지켜지고, 따로 두면 그 상한이 가격 일부를
    가져가서 pricing의 rp 계산(α, μ만 씀)과 LP가 어긋난다.

2단계 (fix_risk 이후, 선행연구의 2순위 목적):
    최소화   Σ_c ΔV_c · x_c        (Gurobi 모델은 최대화라서 목적 계수를 −ΔV_c로 둔다)
    제약     위의 잔해 행, 시간 행 + 위험지수 행 Σ_c p_c · x_c ≥ R*   (가격 −λ, λ ≥ 0)

CG 동안 Gurobi 모델을 새로 만들지 않고, 열(변수)과 시간 행을 계속 더해 간다.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from data import TimePrice, Trip


_ENV: gp.Env | None = None


def gurobi_env() -> gp.Env:
    """마스터 LP/MIP가 공유하는 출력 없는 Gurobi 환경."""
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
    """CG 내내 유지되는 마스터 모델."""
    model: gp.Model                                               # Gurobi 모델
    risk: dict[int, float]                                        # 잔해 i → 위험지수 r_i
    columns: list[Trip] = field(default_factory=list)             # 들어온 트립들 (k번째 트립 = k번째 변수)
    x_vars: list[gp.Var] = field(default_factory=list)            # 트립마다 변수 x_c (columns와 같은 순서)
    debris_rows: dict[int, gp.Constr] = field(default_factory=dict)  # 잔해 i → 잔해 행
    time_rows: dict[float, gp.Constr] = field(default_factory=dict)  # 출발 시각 t → 시간 행
    risk_row: gp.Constr | None = None                             # 2단계에서만: 위험지수 합 ≥ R* 행 (1단계는 None)


@dataclass
class MIPResult:
    """최종 MIP를 푼 결과."""
    selected: list[Trip]   # 고른 트립들 (출발 시각 순)
    risk_sum: float        # R* = 고른 트립들의 위험지수 합 (1순위 목적값)
    dv_sum: float          # 고른 트립들의 총 ΔV [m/s] (2순위 목적값, 선행연구와 같은 기준)
    prop_sum: float        # 고른 트립들의 총 추진제 [kg] (참고용)
    status: str            # "optimal", "time_limit" 등


def new_master(risk: dict[int, float]) -> MasterModel:
    """빈 마스터 모델을 만든다. 잔해 행은 지금 다 만들고(처음엔 비어 있음), 시간 행은 열이 들어올 때 만든다.

    입력: 잔해 i → 위험지수 r_i
    출력: MasterModel
    """
    model = gp.Model("master", env=gurobi_env())
    model.ModelSense = GRB.MAXIMIZE
    m = MasterModel(model=model, risk=risk)
    for i in risk:
        m.debris_rows[i] = model.addLConstr(gp.LinExpr(), GRB.LESS_EQUAL, 1.0, name=f"debris[{i}]")
    return m


def add_column(m: MasterModel, trip: Trip) -> bool:
    """트립 하나를 열로 추가한다.

    입력: 마스터, 트립
    출력: 추가했으면 True, 똑같은 트립(같은 아크들)이 이미 있으면 False
    """
    if any(c.arc_ids == trip.arc_ids for c in m.columns):
        return False

    # 1. 이 트립의 출발 시각 s̄에 시간 행이 없으면 만든다.
    #    새 행에는 그 시각에 진행 중인(s̄_c ≤ t < ē_c) 기존 트립들이 들어간다.
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
    if m.risk_row is not None:
        rows.append(m.risk_row)
        coeffs.append(profit)
        obj = -trip.dv_total
    # 상한(ub)은 두지 않는다 (파일 맨 위 설명 참고)
    x = m.model.addVar(lb=0.0, obj=obj, column=gp.Column(coeffs, rows))
    m.columns.append(trip)
    m.x_vars.append(x)
    return True


def fix_risk(m: MasterModel, r_star: float) -> None:
    """2단계로 바꾼다: 위험지수 합 ≥ r_star 행을 넣고, 목적을 총 ΔV 최소로 바꾼다 (지금 있는 열 그대로).

    입력: 1단계를 마친 마스터, 1단계에서 찾은 최적 위험지수 합 R*
    이후 solve_lp의 LP 값 = −(총 ΔV). 위험지수 행의 가격은 m.risk_row.Pi = −λ (λ ≥ 0)
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
    """마스터 LP를 푼다 (x는 0 이상 실수. 잔해 행 때문에 1을 넘지 않는다).

    입력: 마스터
    출력: (LP 값, α, μ)
          α: 잔해 i → 잔해 행의 가격 (dual)
          μ: 출발 시각 t → 시간 행의 가격 (dual)
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
    """최종 MIP: x를 0/1로 바꿔서 한 번 푼다. LP 모델은 그대로 두고 복사본을 푼다.

    목적: 1단계는 위험지수 최대 후 ΔV 최소. 2단계는 위험지수 하한 아래 ΔV 최소.
    입력: 마스터, 시간 제한 [s]
    출력: MIPResult
    """
    m.model.update()                   # 아직 반영 안 된 열·행을 반영한 뒤 복사한다
    mip = m.model.copy()
    xs = mip.getVars()                 # 복사본의 변수 (columns와 같은 순서)
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
