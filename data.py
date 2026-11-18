"""입력 데이터, 공통 자료형과 설정. 단위는 m, s, kg, m/s.

CSV 인스턴스는 arcs.csv, nodes.csv, instance.json으로 구성된다.
아크를 새로 계산하지 않으며 NumPy 외의 수치/최적화 패키지는 필요 없다.
설정은 load_data()에서 입력 제원으로 갱신하고 다른 모듈은 data.설정명으로 읽는다.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

M_DRY = 1000.0
TANK_MAX = 1200.0
KIT_MASS = 20.0
ISP = 320.0
G0 = 9.80665
C_EXH = ISP * G0
TAU_S = 1800.0
S0 = 0.0
HORIZON_S = 30 * 86400.0
TIME_UNIT_S = 600.0
K_MAX = 10
N_BEST = 50
TOL = 1e-6
MAX_ITERS = 500
MIP_TIME_LIMIT_S = 3600.0
RISK_STEP = 0.5
RISK_SEED = 0

TimePrice = Callable[[float], float]


@dataclass
class ArcTable:
    """동일 인덱스의 다섯 배열 원소가 아크 하나. 0번 노드는 디포."""
    i: np.ndarray
    j: np.ndarray
    t_dep: np.ndarray
    t_arr: np.ndarray
    dv: np.ndarray
    node_ids: list[str]


@dataclass(frozen=True)
class Trip:
    """완성된 디포 왕복 트립 = 마스터 열 하나."""
    debris: frozenset[int]
    sequence: tuple[int, ...]
    arc_ids: tuple[int, ...]
    s: float
    e: float
    s_bar: float
    e_bar: float
    prop: float
    dv_total: float


@dataclass
class PricingResult:
    trips: list[Trip] = field(default_factory=list)
    rps: list[float] = field(default_factory=list)
    # 정확 모드가 빈손이면 개선 열이 없다는 뜻. 모든 집합의 열거를 뜻하지 않는다.
    complete: bool = False


def floor_unit(t: float, unit: float | None = None) -> float:
    unit = TIME_UNIT_S if unit is None else unit
    return math.floor(t / unit) * unit


def ceil_unit(t: float, unit: float | None = None) -> float:
    unit = TIME_UNIT_S if unit is None else unit
    return math.ceil(t / unit) * unit


def time_fee(s_bar: float, e_bar: float, U: TimePrice) -> float:
    return U(e_bar) - U(s_bar)


def route_propellant(dvs: list[float], n_kits: int) -> float:
    """복귀 질량에서 로켓방정식을 역으로 적용해 필요한 추진제를 계산."""
    if n_kits != len(dvs) - 1:
        raise ValueError("키트 수는 다리 수 - 1이어야 합니다.")
    m = M_DRY
    for k in range(len(dvs) - 1, -1, -1):
        m *= math.exp(dvs[k] / C_EXH)
        if k > 0:
            m += KIT_MASS
    return m - M_DRY - KIT_MASS * n_kits


def arcs_among(table: ArcTable, nodes: set[int]) -> np.ndarray:
    node_list = list(nodes)
    return np.flatnonzero(np.isin(table.i, node_list) & np.isin(table.j, node_list))


def validate_table(table: ArcTable) -> None:
    """시간순 라벨링이 전제로 하는 아크 조건을 입력 시 검사."""
    arrays = (table.i, table.j, table.t_dep, table.t_arr, table.dv)
    if any(a.ndim != 1 or len(a) != len(table.i) for a in arrays):
        raise ValueError("아크 배열은 길이가 같은 1차원 배열이어야 합니다.")
    if not table.node_ids or table.node_ids[0] != "depot":
        raise ValueError("0번 노드는 depot이어야 합니다.")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("아크에는 유한한 수만 사용할 수 있습니다.")
    for nodes in (table.i, table.j):
        if np.any(nodes != nodes.astype(int)) or np.any(nodes < 0) or np.any(nodes >= len(table.node_ids)):
            raise ValueError("아크가 알 수 없는 노드를 참조합니다.")
    if np.any(table.i == table.j):
        raise ValueError("동일 노드로 이동하는 아크는 지원하지 않습니다.")
    if np.any(table.t_dep < 0) or np.any(table.t_arr <= table.t_dep) or np.any(table.dv < 0):
        raise ValueError("아크는 출발 >= 0, 도착 > 출발, ΔV >= 0이어야 합니다.")


def load_instance_folder(folder: Path) -> tuple[ArcTable, dict[int, float], dict]:
    """CSV의 ΔV는 km/s에서 m/s로 변환. 모든 잔해의 키트/작업시간은 같아야 한다."""
    folder = Path(folder)
    with (folder / "nodes.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["node"]))
    if len(rows) < 2 or [int(row["node"]) for row in rows] != list(range(len(rows))):
        raise ValueError("nodes.csv는 디포 0과 연속된 잔해 번호 1..n을 포함해야 합니다.")
    node_ids = ["depot"] + [row["norad_id"] or row["name"] for row in rows[1:]]
    risk = {int(row["node"]): float(row["risk_index"]) for row in rows[1:]}
    if any(not math.isfinite(r) or r < 0 for r in risk.values()):
        raise ValueError("위험지수는 유한한 0 이상의 값이어야 합니다.")
    with (folder / "arcs.csv").open(encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream))
        names = ("from_node", "to_node", "departure_s", "arrival_s", "delta_v_km_s")
        columns = tuple(header.index(name) for name in names)
        # ndmin=2로 아크가 하나뿐인 CSV도 같은 방식으로 읽는다.
        a = np.loadtxt(stream, delimiter=",", usecols=columns, ndmin=2)
    if a.size == 0:
        a = np.empty((0, 5))
    table = ArcTable(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4] * 1000.0, node_ids)
    validate_table(table)
    table.i, table.j = table.i.astype(int), table.j.astype(int)

    info = json.loads((folder / "instance.json").read_text(encoding="utf-8-sig"))
    servicer = info["servicer"]
    kits = {float(row["kit_mass_kg"]) for row in rows[1:]}
    services = {float(row["service_time_s"]) for row in rows[1:]}
    if len(kits) != 1 or len(services) != 1:
        raise ValueError("모든 잔해의 키트 질량과 작업시간이 같아야 합니다.")
    kit, service = kits.pop(), services.pop()
    if not math.isfinite(kit) or kit <= 0:
        raise ValueError("키트 질량은 유한한 양수여야 합니다.")
    settings = {
        "dry_kg": float(servicer["dry_mass_kg"]),
        "tank_kg": float(servicer["max_fuel_mass_kg"]),
        "isp_s": float(servicer["isp_s"]),
        "kit_kg": kit,
        "service_s": service,
        "horizon_s": float(info.get("mission_horizon_s", 30 * 86400.0)),
        "k_max": math.floor(float(servicer.get("max_kit_mass_kg", 10 * kit)) / kit + 1e-9),
    }
    if any(not math.isfinite(value) for value in settings.values()):
        raise ValueError("서비서 제원은 유한한 값이어야 합니다.")
    if any(settings[key] <= 0 for key in ("dry_kg", "isp_s", "horizon_s", "k_max")):
        raise ValueError("건조질량, 비추력, 계획 기간과 최대 키트 수는 양수여야 합니다.")
    if settings["tank_kg"] < 0 or service < 0:
        raise ValueError("탱크 용량과 작업시간은 음수일 수 없습니다.")
    return table, risk, settings


def load_arc_table(path: Path) -> ArcTable:
    """기존 NPZ 입력도 지원한다."""
    with np.load(path, allow_pickle=False) as f:
        table = ArcTable(f["i"], f["j"], f["t_dep"], f["t_arr"], f["dv"],
                         [str(s) for s in f["node_ids"]])
    validate_table(table)
    table.i, table.j = table.i.astype(int), table.j.astype(int)
    return table


def load_data(path: Path) -> tuple[ArcTable, dict[int, float]]:
    """인스턴스 폴더면 제원을 적용. NPZ면 기본 제원과 고정 시드의 위험지수 사용."""
    global M_DRY, TANK_MAX, KIT_MASS, ISP, C_EXH, TAU_S, HORIZON_S, K_MAX
    path = Path(path)
    if path.is_dir():
        table, risk, settings = load_instance_folder(path)
        M_DRY, TANK_MAX, KIT_MASS = settings["dry_kg"], settings["tank_kg"], settings["kit_kg"]
        ISP, TAU_S = settings["isp_s"], settings["service_s"]
        C_EXH = ISP * G0
        HORIZON_S, K_MAX = settings["horizon_s"], settings["k_max"]
        return table, risk
    table = load_arc_table(path)
    rng = np.random.default_rng(RISK_SEED)
    return table, {i: float(rng.integers(1, 6)) for i in range(1, len(table.node_ids))}


def table_stats(table: ArcTable) -> dict:
    n = len(table.node_ids)
    counts = [int(((table.i == i) & (table.j == j)).sum())
              for i in range(n) for j in range(n) if i != j]
    return {"n_arcs": len(table.dv), "pairs": len(counts), "pairs_without_arcs": counts.count(0),
            "arcs_per_pair_min_median_max": (min(counts), float(np.median(counts)), max(counts)) if counts else None,
            "dv_min_median_m_s": (float(table.dv.min()), float(np.median(table.dv))) if len(table.dv) else None}
