"""시간순 라벨링으로 마스터에 추가할 트립을 찾는다.

탐색 방식
  1. 칸을 (방문 집합 S, 위치 j)로 나누지 않는다. 위치 j마다 "대기 목록" 하나만 두고, S는 라벨 안에 넣는다.
     "모든 집합의 최선 트립"을 구하지 않고 rp가 큰 트립만 찾는다 (CG에는 이것으로 충분하다).
  2. 지배: 같은 위치에서 라벨 A가 B를 지배 ⇔ 아래가 모두 성립
         S_A ⊆ S_B      A가 들른 잔해는 B도 다 들렀다 → B 뒤에 붙일 남은 경로를 A 뒤에도 붙일 수 있다
         g_A ≥ g_B      지금까지 점수 g = Σ_{i∈S} v_i + U(s̄) − dv_weight·ΔV (클수록 좋다)
         ΔV_A ≤ ΔV_B, kit_A ≤ kit_B    탱크 여유
     시각 t는 비교하지 않는다: 시간 순으로 훑으면서 "이미 떠날 수 있게 된" 라벨끼리만 비교하기 때문이다.
  3. 빠른 pricing (use_skip=True. 정답 보장 없음):
         v_i > 0인 잔해만 들르고, 지배할 때 S_A ⊆ S_B 대신 |S_A| ≤ |S_B|만 본다 (라벨이 크게 줄어든다).
      cg.py가 빈손일 때 정확한 pricing (use_skip=False)으로 다시 확인한다.

kit는 Σ p·(exp(D_k/c) − 1)이다. 키트 자체의 질량을 뺀 운반 추진제
항을 비교해야 방문 개수가 다른 집합 사이에서도 지배가 성립한다.
정확 모드는 최대 rp와 개선 열의 존재 여부를 보존한다. 모든 집합의 상위 N개를 열거하지는 않는다.

진행: 모든 아크를 출발 시각 순으로 한 번 훑는다 (시간은 앞으로만 가므로).
  - 라벨이 잔해 k에 도착하면, 작업이 끝나는 시각 (도착 + τ)에 k의 대기 목록에 들어간다 (지배당하면 버린다).
  - 아크 i → k가 출발할 때, i의 대기 목록에 있는 라벨마다 이 아크를 붙인다. k가 디포면 트립 완성 → rp 계산.
  - rp 상한이 기준값 b (모은 트립이 n_best개면 n_best번째로 큰 rp, 아니면 min_rp) 이하인 라벨은 버린다.
"""

from __future__ import annotations

import bisect
import heapq
import itertools
import math
from collections import defaultdict
from typing import NamedTuple

import numpy as np

import data as d
from data import ArcTable, N_BEST, TOL, PricingResult, TimePrice, Trip
from data import arcs_among, ceil_unit, floor_unit, route_propellant


class Label(NamedTuple):
    """부분 경로 하나의 요약."""
    t: float             # 지금 잔해를 떠날 수 있는 가장 이른 시각 = 도착 시각 + τ [s]
    g: float             # 점수 = Σ_{i∈S} v_i + U(s̄) − dv_weight·ΔV. 디포로 닫으면 rp = g − U(ē) − dv_weight·(마지막 다리 ΔV)
    dv: float            # 지금까지의 ΔV 합 [m/s]
    kit: float           # 키트 운반 추진제 항 Σ p·expm1(D_k / c) [kg]
    S: frozenset[int]    # 들른 잔해 집합
    ub: float            # 이 라벨에서 나올 수 있는 rp의 상한 (가지치기에 쓴다)
    arc: int             # 마지막으로 탄 아크 번호
    prev: Label | None   # 바로 앞 라벨 (트립을 복원할 때 거꾸로 따라간다)


def dominates(a: Label, b: Label, fast: bool) -> bool:
    """같은 위치에서 둘 다 떠날 수 있는 라벨 a가 b를 지배하나. fast면 집합 대신 개수만 본다 (빠른 pricing)."""
    if a.g < b.g or a.dv > b.dv or a.kit > b.kit:
        return False
    return len(a.S) <= len(b.S) if fast else a.S <= b.S

def cal_min_out_deltav(table: ArcTable):
    """R(i, t) = 시각 t 이후에 노드 i를 떠나 디포로 돌아가는 데 드는 최소 ΔV [m/s]. 못 돌아가면 inf.

    재방문 금지, 최대 방문 수, 탱크는 무시하고 계산한다 → 실제 남은 ΔV의 올바른 하한.
    dual과 무관하므로 CG 전에 한 번만 계산하면 된다.

    방법: 아크를 출발 시각이 늦은 것부터 훑는다.
      아크 i → j (출발 t_dep, 도착 t_arr)를 타면 드는 최소 ΔV
          = dv + (j가 디포면 0, 아니면 R(j, t_arr + τ))
      t_arr + τ는 t_dep보다 늦으므로, R(j, t_arr + τ)는 이미 앞에서 다 계산되어 있다.
    """
    t_dep, t_arr, dv = table.t_dep.tolist(), table.t_arr.tolist(), table.dv.tolist()
    frm, to = table.i.tolist(), table.j.tolist()

    # 노드 i마다 두 목록을 같은 순서로 쌓는다 (출발 시각이 늦은 아크부터)
    #   neg_t[i]: −(출발 시각)      → 오름차순이 되도록 부호를 바꿔 저장 (bisect용)
    #   cost[i]:  그 시각 이후 출발해서 디포로 가는 최소 ΔV (지금까지 본 것 중 최소)
    neg_t = defaultdict(list)
    cost = defaultdict(list)

    def R(i: int, t: float) -> float:
        k = bisect.bisect_right(neg_t[i], -t)     # 출발 시각 ≥ t인 아크 개수
        return cost[i][k - 1] if k else math.inf

    for a in sorted(range(len(dv)), key=lambda x: t_dep[x], reverse=True):
        i, j = frm[a], to[a]
        if i == 0:
            continue                                   # 디포에서 나가는 아크는 필요 없다
        if j == 0:
            if t_arr[a] + d.S0 > d.HORIZON_S:
                continue                               # 기간 안에 못 돌아오는 복귀 아크
            c = dv[a]
        else:
            c = dv[a] + R(j, t_arr[a] + d.TAU_S)        # j에서 작업을 마친 뒤 디포까지
        if cost[i]:
            c = min(c, cost[i][-1])                    # 더 늦게 떠나는 길이 더 싸면 그 값
        neg_t[i].append(-t_dep[a])
        cost[i].append(c)

    return R

def usable_arc_ids(table, ids, v, U, k_max, dv_weight, min_rp):
    """
    아크 a를 반드시 한 번 타는 디포 왕복 트립에서 rp > min_rp가 아닌 아크는 다 미리 지우기
    """
    arc = np.asarray(ids, dtype=np.intp)
    if arc.size == 0 or min_rp == -math.inf:
        return arc.tolist()

    src, dst = table.i[arc], table.j[arc]
    dep = table.t_dep[arc].astype(float, copy=False)
    arr = table.t_arr[arc].astype(float, copy=False)
    ready = arr + d.TAU_S

    reward = np.zeros(len(table.node_ids))
    for node, value in v.items():
        reward[node] = value
    cost = dv_weight * table.dv[arc].astype(float, copy=False)
    gain = reward[dst] - cost

    # 잔해별로 시간상 연결할 수 있는 도착·출발 아크의 경계를 구한다.
    links = []
    for node in v:
        incoming = np.flatnonzero(dst == node)
        outgoing = np.flatnonzero(src == node)  # 이미 출발 시각순
        if not incoming.size or not outgoing.size:
            continue

        incoming = incoming[np.argsort(ready[incoming], kind="stable")]
        first_out = np.searchsorted(dep[outgoing], ready[incoming], side="left")
        n_in = np.searchsorted(ready[incoming], dep[outgoing], side="right")
        links.append((incoming, outgoing, first_out, n_in))

    def best_after(score):
        """각 도착 아크에서 작업을 마친 뒤 탈 수 있는 출발 아크의 최고 점수."""
        result = np.full(arc.size, -np.inf)
        for incoming, outgoing, first_out, _ in links:
            tail = np.maximum.accumulate(score[outgoing][::-1])[::-1]
            result[incoming] = np.append(tail, -np.inf)[first_out]
        return result

    def best_before(score):
        """각 출발 아크가 떠나기 전에 작업을 마친 도착 아크의 최고 점수."""
        result = np.full(arc.size, -np.inf)
        for incoming, outgoing, _, n_in in links:
            head = np.maximum.accumulate(score[incoming])
            result[outgoing] = np.r_[-np.inf, head][n_in]
        return result

    first = src == 0
    end = arr + d.S0
    back = (dst == 0) & (end <= d.HORIZON_S)
    if not first.any() or not back.any():
        return []

    # 기존 price2와 같은 방식으로 출발·복귀 시각을 반올림한다.
    start_fee = np.array([U(floor_unit(t)) for t in dep[first]])
    end_fee = np.array([U(ceil_unit(t)) for t in end[back]])

    prefix = np.full(arc.size, -np.inf)
    close = np.full(arc.size, -np.inf)
    prefix[first] = gain[first] + start_fee
    close[back] = -cost[back] - end_fee

    # suffix[h][a]는 a를 탄 뒤 잔해를 최대 h개 더 방문하고 복귀할 때의 최고 점수.
    suffix = [best_after(close)]
    for h in range(1, k_max):
        suffix.append(best_after(np.maximum(close, gain + suffix[h - 1])))

    # 출발 아크의 상한을 먼저 채우고, 방문 수별로 중간·복귀 아크의 상한을 구한다.
    upper = prefix + suffix[k_max - 1]
    for visited in range(1, k_max + 1):
        before = best_before(prefix)
        upper = np.maximum(upper, before + close)

        if visited < k_max:
            prefix = before + gain
            upper = np.maximum(upper, prefix + suffix[k_max - visited - 1])

    # 점수 합산 순서에 따른 부동소수점 오차로 개선 열을 놓치지 않도록 한다.
    scale = max(1.0, np.max(np.abs(reward)), cost.max(),
                np.max(np.abs(start_fee)), np.max(np.abs(end_fee)))
    slack = max(1e-8, 16 * np.finfo(float).eps * (k_max + 2)**2 * scale)
    return arc[upper >= min_rp - slack].tolist()

def price2(risk: dict[int, float], alpha: dict[int, float], U: TimePrice, table: ArcTable,
           use_skip: bool = True, n_best: int | None = N_BEST, min_rp: float = TOL, k_max: int | None = None,
           dv_weight: float = 0.0, R=None) -> PricingResult:
    """pricing 한 번. 자원 설정은 data.py에서 읽는다.

    rp = Σ_{i∈S} (r_i − α_i) − 사용료 − dv_weight·ΣΔV > min_rp인 후보를 최대 n_best개 반환한다.
    살아남은 후보 중 같은 방문 집합은 하나만 반환한다. 전역 상위 N개 보장은 아니다.
    use_skip=True면 빠른 pricing (정답 보장 없음, complete=False),
    False면 정확한 pricing (complete=True. 빈손이면 rp > min_rp인 트립이 정말 없다).

    risk: 잔해별 위험 지수
    alpha: 잔해별 마스터 쌍대 가격
    U:  누적 시간 간격 함수(듀얼값)
    table: 아크 데이터
    use_skipt: 어떤 모드일지
    n_best: 반환할 트립 수 상한
    min_rp: 반환 대상으로 인정할 rp의 하한
    k_max: 최대 방문 개수
    dv_weight: 델타 v 목적함수 가중치
    R: cal_min_out_deltav(table)의 결과 R(i, t). 없으면 여기서 계산한다 (약 0.3초)

    """
    k_max = d.K_MAX if k_max is None else k_max
    if k_max < 1 or (n_best is not None and n_best < 1):
        raise ValueError("k_max와 n_best는 1 이상이어야 합니다.")
    if not math.isfinite(dv_weight) or dv_weight < 0:
        raise ValueError("dv_weight는 유한한 0 이상의 값이어야 합니다.")
    fast = use_skip
    w = dv_weight
    v = {i: risk[i] - alpha[i] for i in risk}
    nodes = {i for i in risk if v[i] > 0} if fast else set(risk)   # 빠른 pricing은 v > 0인 잔해만 들른다
    t_dep, t_arr, dv = table.t_dep.tolist(), table.t_arr.tolist(), table.dv.tolist()   # 반복문에서 빠르게 쓰려고
    frm, to = table.i.tolist(), table.j.tolist()

    # 1. 쓸 아크: nodes ∪ {디포} 사이의 아크를 출발 시각 순으로. 기간 안에 디포로 돌아오지 못하는 아크는 뺀다
    ids = arcs_among(table, nodes | {0})
    ids = [a for a in ids[np.argsort(table.t_dep[ids], kind="stable")].tolist()
           if to[a] != 0 or t_arr[a] + d.S0 <= d.HORIZON_S]

    # 디포까지 돌아가는 최소 ΔV R(i, t) (앞으로 쓸 ΔV의 하한). dual과 무관해서 보통 cg.py가 한 번 계산해 넘겨준다
    if R is None:
        R = cal_min_out_deltav(table)

    positive = sorted(((v[i], i) for i in nodes if v[i] > 0), reverse=True)   # 양수 v (큰 순)

    extra_of: dict[frozenset[int], float] = {}   # S → S 밖에서 더 모을 수 있는 v의 상한 (같은 S는 한 번만 계산)
    best: dict[frozenset[int], tuple[float, int, Label]] = {}   # S → (rp, 마지막 아크, 그 앞 라벨) 완성된 트립 후보
    b = min_rp                                                  # 기준값 (트립을 모으면서 올라간다)
    waiting: dict[int, list[Label]] = defaultdict(list)         # 잔해 → 대기 목록 (떠날 수 있게 된, 서로 지배하지 않는 라벨) 노드별로 작업을 마치고 떠날 수 있는 라벨 목록
    arriving: list = []                                         # 힙: (떠날 수 있게 되는 시각, 번호, 잔해, 라벨) 
    order = itertools.count()                                   # 힙에서 시각이 같을 때 쓰는 번호


    def arrive(k: int, t: float, g: float, total_dv: float, kit: float, S: frozenset[int], a: int, prev):
        """
        아크가 하나 들어오면 그 아크의 시작점의 라벨들에 그 아크를 붙어봄.
        탱크 용량이 가망이 없거나 reduced cost의 상한이 b 이하면 버림.
        아니면 떠날 수 있게 되는 시각 t(도착 시간에 작업 시간 더한 값)에 k의 대기목록에 들어가도록 힙(arriving)에 넣는다.
        """
        rest = R(k, t)               # t 이후 k를 떠나 디포까지 돌아가는 최소 ΔV (못 돌아가면 inf)
        lb = total_dv + rest         # 디포까지 돌아가려면 적어도 이만큼 ΔV를 쓴다
        if d.M_DRY * math.expm1(lb / d.C_EXH) + kit > d.TANK_MAX:
            return                   # rest가 inf면 여기서 버려진다
        if S not in extra_of:
            extra_of[S] = sum(itertools.islice((x for x, i in positive if i not in S), k_max - len(S)))
        # rp 상한: 더 모을 v는 extra 이하, 사용료 U(ē) − U(s̄) ≥ U(t) − U(s̄) (ē > t), 남은 ΔV ≥ rest
        # 지금 가장 가치가 큰 잔해 순서로, 더 들릴 수 있는 만큼의 수를 더함. = 가치 상한
        ub = g + extra_of[S] - U(t) - w * rest
        if ub > b:
            heapq.heappush(arriving, (t, next(order), k, Label(t, g, total_dv, kit, S, ub, a, prev))) #힙에 넣기
    ids = usable_arc_ids(table, ids, v, U, k_max, dv_weight, min_rp)
    # 2. 아크를 출발 시각 순으로 훑는다
    for a in ids:
        # 이 아크가 떠나기 전에 떠날 수 있게 된 라벨을 대기 목록waiting에 넣는다
        # 힙의 가장 이른 준비 시각이 현재 아크 출발시각 이하면 그 라벨 사용 가능
        while arriving and arriving[0][0] <= t_dep[a]:
            _, _, j, lab = heapq.heappop(arriving) #현재 노드 및 라벨
            if lab.ub <= b or any(dominates(f, lab, fast) for f in waiting[j]):
                continue     # 그사이 기준값이 올라 가망이 없어졌거나, 더 나은 라벨이 이미 있다
            waiting[j] = [f for f in waiting[j] if not dominates(lab, f, fast)] + [lab] # 아니면 지금 라벨 추가 및 기존 라벨 제거

        i, k = frm[a], to[a]
        if i == 0:
            # 디포에서 출발하는 아크
            # 기존 라벨을 None으로 arrive 함수에 넣기. 
            kit = d.KIT_MASS * math.expm1(dv[a] / d.C_EXH)
            arrive(k, t_arr[a] + d.TAU_S, v[k] + U(floor_unit(t_dep[a])) - w * dv[a], dv[a], kit,
                   frozenset([k]), a, None)
        elif k == 0:
            # 디포로 복귀하는 아크
            # i 노드의 모든 가능한 라벨을 가져다가 (시간은 에초에 이 아크의 출발 시간보다 그 라벨 출발 가능 시간이 더 이른 녀석들만 waiting 에 들어가 있다)
            end = U(ceil_unit(t_arr[a] + d.S0))
            for lab in waiting[i]:
                total = lab.dv + dv[a]
                if d.M_DRY * math.expm1(total / d.C_EXH) + lab.kit > d.TANK_MAX:
                    continue
                rp = lab.g - end - w * dv[a] #reduced cost 최종 계산
                if rp > b and (lab.S not in best or rp > best[lab.S][0]):
                    best[lab.S] = (rp, a, lab)
                    if n_best is not None and len(best) >= n_best:
                        b = max(b, heapq.nlargest(n_best, (x[0] for x in best.values()))[-1])
        else:
            # 잔해 i → 잔해 k: 대기 목록의 라벨마다 이 아크를 붙인다
            # 잔해 i에서 기다리고 있는 라벨이 다 목표 waiting 변수 사용
            for lab in waiting[i]:
                if k in lab.S or len(lab.S) >= k_max:
                    continue
                total = lab.dv + dv[a]
                kit = lab.kit + d.KIT_MASS * math.expm1(total / d.C_EXH)
                arrive(k, t_arr[a] + d.TAU_S, lab.g + v[k] - w * dv[a], total, kit, lab.S | {k}, a, lab)

    # 3. rp가 큰 n_best개를 트립으로 복원한다 (마지막 라벨에서 prev를 거꾸로 따라가며 아크를 모은다)
    chosen = sorted(best.items(), key=lambda x: x[1][0], reverse=True)[:n_best]
    trips, rps = [], []
    for S, (rp, a, lab) in chosen: # 방문 집합, 점수, 복귀 아크, 마지막 라벨
        arc_ids = [a] #마지막 복귀 아크
        while lab is not None:
            arc_ids.append(lab.arc)
            lab = lab.prev
        arc_ids.reverse()
        dvs = [dv[x] for x in arc_ids]
        s, e = t_dep[arc_ids[0]], t_arr[arc_ids[-1]] + d.S0
        trips.append(Trip(debris=S, sequence=tuple(to[x] for x in arc_ids[:-1]), arc_ids=tuple(arc_ids),
                          s=s, e=e, s_bar=floor_unit(s), e_bar=ceil_unit(e),
                          prop=route_propellant(dvs, len(S)), dv_total=sum(dvs)))
        rps.append(rp)
    return PricingResult(trips=trips, rps=rps, complete=not fast)
