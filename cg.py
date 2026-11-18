"""labeling2 기반 열 생성: 마스터 LP → pricing → 새 트립 추가 → 최종 MIP.

1단계는 위험지수 최대화, 선택적인 2단계는 위험지수 하한을 유지하면서 총 ΔV 최소화.
초기 단독 트립도 price2로 만든다. 경로 ILP나 실행 중 함수 교체를 사용하지 않는다.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import data as d
from data import ArcTable
from labeling2 import price2
from master import MasterModel, MIPResult, add_column, fix_risk, new_master, solve_lp, solve_mip, time_price


@dataclass
class IterLog:
    it: int
    lp_value: float
    exact: bool
    n_new_columns: int
    n_columns: int
    n_time_rows: int
    t_lp_s: float
    t_pricing_s: float


@dataclass
class CGResult:
    ub: float                    # 마지막 LP 값. ub_certified일 때만 전체 문제의 상한
    ub_certified: bool
    mip: MIPResult
    gap: float
    proven_optimal: bool
    logs: list[IterLog] = field(default_factory=list)
    master: MasterModel | None = None


@dataclass
class DvResult:
    lb: float                    # 마지막 LP의 총 ΔV. lb_certified일 때만 전체 문제의 하한
    lb_certified: bool
    mip: MIPResult
    gap: float
    logs: list[IterLog] = field(default_factory=list)


def run_cg(risk: dict[int, float], table: ArcTable, use_skip: bool = True,
           max_iters: int = d.MAX_ITERS, mip_time_limit_s: float = d.MIP_TIME_LIMIT_S,
           verbose: bool = True) -> CGResult:
    """1단계: 위험지수 합 최대. 빠른 pricing이 빈손이면 정확 모드로 확인한다."""
    if max_iters < 1:
        raise ValueError("max_iters는 1 이상이어야 합니다.")
    if any(not math.isfinite(r) or r < 0 for r in risk.values()):
        raise ValueError("위험지수는 유한한 0 이상의 값이어야 합니다.")
    m = new_master(risk)
    # 순가치 0, ΔV 가중치 1이면 각 잔해의 최소 ΔV 왕복 트립을 얻는다.
    zero = {i: 0.0 for i in risk}
    initial = price2(zero, zero, lambda t: 0.0, table, use_skip=False,
                     n_best=None, min_rp=-math.inf, k_max=1, dv_weight=1.0)
    for trip in initial.trips:
        add_column(m, trip)
    if verbose:
        print(f"[초기 열] {len(m.columns)}개 (단독 왕복 불가 {len(risk) - len(m.columns)}개)", flush=True)

    logs = []
    exact, certified = not use_skip, False
    total_risk = sum(risk.values())
    for it in range(1, max_iters + 1):
        t0 = time.perf_counter()
        lp_value, alpha, mu = solve_lp(m)
        U = time_price(mu)
        t1 = time.perf_counter()
        # 모든 위험지수를 확보한 정수해는 자명한 상한에 도달했다.
        if lp_value >= total_risk - d.TOL:
            if solve_mip(m, mip_time_limit_s).risk_sum >= total_risk - d.TOL:
                certified = True
                if verbose:
                    print(f"[CG {it}] 모든 잔해의 위험지수 합 {total_risk:g} 달성", flush=True)
                break

        res = price2(risk, alpha, U, table, use_skip=not exact)
        n_new = sum(add_column(m, trip) for trip in res.trips)
        t2 = time.perf_counter()
        logs.append(IterLog(it, lp_value, res.complete, n_new, len(m.columns), len(m.time_rows),
                            t1 - t0, t2 - t1))
        if verbose:
            print(f"[CG {it:3d}] LP={lp_value:10.4f}  {'정확' if res.complete else '빠름'}  "
                  f"새 열 {n_new:3d}  열 {len(m.columns):5d}  시간 행 {len(m.time_rows):4d}  "
                  f"({t1 - t0:.2f}s + {t2 - t1:.2f}s)", flush=True)
        if n_new:
            exact = not use_skip
        elif res.complete:
            # 양의 rp인데 이미 있던 열만 나온 경우는 수치 문제일 수 있으므로 인증하지 않는다.
            certified = not res.trips
            break
        else:
            exact = True

    # 반복 제한 직전에 추가한 열까지 반영한다. 이전 LP 값으로 gap을 계산하지 않는다.
    lp_value, _, _ = solve_lp(m)
    mip = solve_mip(m, mip_time_limit_s)
    gap = lp_value - mip.risk_sum
    # 위험지수가 실제로 RISK_STEP의 배수인지 확인한 경우에만 이산 상한을 사용한다.
    step = d.RISK_STEP
    on_grid = step > 0 and all(abs(r / step - round(r / step)) < 1e-8 for r in risk.values())
    closed = (math.floor(lp_value / step + d.TOL) * step <= mip.risk_sum + d.TOL
              if on_grid else gap <= d.TOL)
    proven = certified and mip.status == "optimal" and closed
    if verbose:
        print(f"[MIP] 위험지수={mip.risk_sum:g}  LP={lp_value:.4f}  "
              f"상한 인증={certified}  gap={gap:.4f}  ({mip.status})", flush=True)
    return CGResult(lp_value, certified, mip, gap, proven, logs, m)


def minimize_dv(result: CGResult, table: ArcTable, max_iters: int = d.MAX_ITERS,
                mip_time_limit_s: float = d.MIP_TIME_LIMIT_S, verbose: bool = True,
                use_skip: bool = True) -> DvResult:
    """2단계: 위험지수 >= 1단계 MIP 값인 해 중 총 ΔV 최소화.

1단계가 미증명이면 위험지수 전역 최적을 전제로 하지 않는다.
pricing은 v_i = λ r_i - α_i와 ΔV 가중치 1을 사용한다.
"""
    if max_iters < 1:
        raise ValueError("max_iters는 1 이상이어야 합니다.")
    if result.master is None:
        raise ValueError("2단계에는 1단계 마스터가 필요합니다.")
    m = result.master
    fix_risk(m, result.mip.risk_sum)
    logs = []
    exact, certified = not use_skip, False
    for it in range(1, max_iters + 1):
        t0 = time.perf_counter()
        lp_value, alpha, mu = solve_lp(m)
        lam = -m.risk_row.Pi
        U = time_price(mu)
        t1 = time.perf_counter()
        res = price2({i: lam * r for i, r in m.risk.items()}, alpha, U, table,
                     dv_weight=1.0, use_skip=not exact)
        n_new = sum(add_column(m, trip) for trip in res.trips)
        t2 = time.perf_counter()
        logs.append(IterLog(it, lp_value, res.complete, n_new, len(m.columns), len(m.time_rows),
                            t1 - t0, t2 - t1))
        if verbose:
            print(f"[ΔV {it:3d}] LP 총 ΔV={-lp_value:10.2f}  λ={lam:8.3f}  "
                  f"{'정확' if res.complete else '빠름'}  새 열 {n_new:3d}  "
                  f"열 {len(m.columns):5d}  ({t1 - t0:.2f}s + {t2 - t1:.2f}s)", flush=True)
        if n_new:
            exact = not use_skip
        elif res.complete:
            certified = not res.trips
            break
        else:
            exact = True
    lp_value, _, _ = solve_lp(m)
    mip = solve_mip(m, mip_time_limit_s)
    lb = -lp_value
    gap = mip.dv_sum - lb
    if verbose:
        print(f"[ΔV MIP] 총 ΔV={mip.dv_sum:.2f}  LP={lb:.2f}  하한 인증={certified}  "
              f"gap={gap:.2f} m/s  위험지수={mip.risk_sum:g}", flush=True)
    return DvResult(lb, certified, mip, gap, logs)


def save_result(result: CGResult | DvResult, table: ArcTable, path: Path) -> None:
    """두 단계 모두 같은 트립 형식으로 JSON 저장. 기존 결과 키를 유지한다."""
    trips = [{"sequence": [table.node_ids[i] for i in t.sequence], "arc_ids": list(t.arc_ids),
              "depart_s": t.s, "return_s": t.e, "prop_kg": t.prop, "dv_m_s": t.dv_total}
             for t in result.mip.selected]
    output = {"risk_sum": result.mip.risk_sum, "dv_sum": result.mip.dv_sum,
              "prop_sum": result.mip.prop_sum, "mip_status": result.mip.status,
              "trips": trips, "logs": [asdict(log) for log in result.logs],
              "settings": {name: getattr(d, name) for name in (
                  "M_DRY", "TANK_MAX", "KIT_MASS", "ISP", "TAU_S",
                  "S0", "HORIZON_S", "TIME_UNIT_S", "K_MAX")}}
    if isinstance(result, CGResult):
        output.update(ub=result.ub, ub_certified=result.ub_certified, gap=result.gap,
                      proven_optimal=result.proven_optimal)
    else:
        output.update(dv_lb=result.lb, dv_lb_certified=result.lb_certified, dv_gap=result.gap)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
