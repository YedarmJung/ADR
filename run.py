"""저장된 아크로 ADR 최적화 실행. python run.py --help 참고."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import data as d


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수가 필요합니다.")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("유한한 양수가 필요합니다.")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="arcs.csv/nodes.csv/instance.json 폴더 또는 기존 NPZ")
    parser.add_argument("output", type=Path, help="1단계 결과 JSON 경로. 2단계는 같은 이름에 _dv 추가")
    parser.add_argument("--max-iters", type=positive_int, default=d.MAX_ITERS, help="1단계 CG 최대 반복 수")
    parser.add_argument("--dv-iters", type=positive_int, default=d.MAX_ITERS, help="2단계 CG 최대 반복 수")
    parser.add_argument("--mip-time-limit", type=positive_float, default=d.MIP_TIME_LIMIT_S, help="MIP 시간 제한 [초]")
    parser.add_argument("--phase1-only", action="store_true", help="위험지수 최대화 단계만 실행")
    parser.add_argument("--exact", action="store_true", help="두 단계 모두 처음부터 정확한 pricing 사용")
    args = parser.parse_args(argv)

    # --help와 데이터 도구에는 Gurobi를 요구하지 않는다.
    from cg import minimize_dv, run_cg, save_result

    table, risk = d.load_data(args.input)
    print(f"pricing: labeling2 | 최대 방문: {d.K_MAX}", flush=True)
    print("아크 표:", d.table_stats(table), flush=True)
    print("위험지수 합:", sum(risk.values()), flush=True)

    def show(trips):
        for trip in trips:
            sequence = " → ".join(table.node_ids[i] for i in trip.sequence)
            print(f"  {trip.s / 86400:.2f}일 → {trip.e / 86400:.2f}일  {sequence}  "
                  f"ΔV {trip.dv_total:.1f} m/s  추진제 {trip.prop:.1f} kg", flush=True)

    t0 = time.perf_counter()
    result = run_cg(risk, table, use_skip=not args.exact, max_iters=args.max_iters,
                    mip_time_limit_s=args.mip_time_limit)
    show(result.mip.selected)
    save_result(result, table, args.output)
    print(f"1단계 저장: {args.output} ({time.perf_counter() - t0:.1f}s)", flush=True)
    if not args.phase1_only:
        if not result.proven_optimal:
            print(f"1단계 최적성 미증명: 위험지수 >= {result.mip.risk_sum:g} 조건으로 2단계를 진행합니다.", flush=True)
        t1 = time.perf_counter()
        dv = minimize_dv(result, table, max_iters=args.dv_iters, use_skip=not args.exact,
                         mip_time_limit_s=args.mip_time_limit)
        show(dv.mip.selected)
        dv_path = args.output.with_name(args.output.stem + "_dv.json")
        save_result(dv, table, dv_path)
        print(f"2단계 저장: {dv_path} ({time.perf_counter() - t1:.1f}s)", flush=True)
    result.master.model.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
