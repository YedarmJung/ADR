저장된 아크로 실행하는 ADR 최적화

현재 실행 코드는 루트의 다음 다섯 파일이다.
  data.py       CSV/NPZ 입력, 설정, ArcTable/Trip/PricingResult, 추진제 계산
  labeling2.py  시간순 라벨링 pricing (빠른 모드 + 정확 모드)
  master.py     트립 선택 LP/MIP, 쌍대가격
  cg.py         초기 열 생성, 위험지수 최대화, 총 ΔV 최소화, JSON 저장
  run.py        명령행 실행

기존 코드 21개는 old/에 원본 그대로 보관했다.
  old/*.py       루트에 있던 파이썬 파일 12개
  old/tmp/*.py   이전 보조 스크립트 2개
  old/tests/*.py 이전 테스트 7개
  old/archive_manifest.json  각 원본의 상대경로와 SHA-256
새 코드에서는 old/의 모듈을 import하지 않는다. 기존 데이터와 결과는 원래 위치에 있다.

필요 환경
  Python 3.10 이상, NumPy, gurobipy 및 사용 가능한 Gurobi 라이선스.
  이 프로젝트의 .venv에는 기존 라이선스와 맞는 gurobipy 13.0.3을 설치했다.
  SciPy, PyKEP, Matplotlib은 현재 다섯 파일의 실행에 필요하지 않다.
  다른 환경에서는 python -m pip install -r requirements.txt로 패키지를 설치한다.

입력 폴더
  arcs.csv       from_node,to_node,departure_s,arrival_s,delta_v_km_s
                 위 열을 포함해야 한다. 그 외 열이 있어도 된다. ΔV는 m/s로 변환한다.
  nodes.csv      node,name,norad_id,risk_index,kit_mass_kg,service_time_s
                 디포는 0, 잔해는 연속된 1..n. 모든 잔해의 키트 질량/작업시간은 같아야 한다.
  instance.json  servicer 제원, mission_horizon_s. max_kit_mass_kg로 최대 방문 수를 계산한다.
  기존 .npz 입력도 지원한다. 위험지수는 고정 시드로 1~5를 생성하고 data.py의 제원을 쓴다.

프로젝트 폴더에서 실행 (PowerShell)
  .\.venv\Scripts\python.exe -X utf8 run.py adr_scaling_n10-50/data/n10

기본 동작은 1단계 위험지수 최대화 후 2단계 총 ΔV 최소화다.
결과는 results/<시나리오>_<날짜_시각>/ 폴더에 저장된다 (예: results/n10_20261118_1305/).
  result.json     1단계 결과
  result_dv.json  2단계 결과
  run.log         화면 출력 전체

주요 선택사항
  --phase1-only      1단계만 실행
  --exact            두 단계 모두 처음부터 정확한 pricing 사용
  --max-iters 500    1단계 CG 최대 반복 수
  --dv-iters 500     2단계 CG 최대 반복 수
  --mip-time-limit 3600   각 MIP의 시간 제한 (초)
  --help            전체 사용법

알고리즘과 인증 범위
  라벨링은 아크 출발 시각 순으로 부분 경로를 확장한다. 현재 노드에서 작업이 끝난
  경로만 다음 아크를 탈 수 있다. 재방문, 키트 개수, 시간, 자원 제한을 검사한다.
  같은 노드의 라벨을 비교해 점수/자원/방문 집합에서 불리한 라벨을 제거한다.
  남은 순가치를 최대한 더해도 기준값을 넘을 수 없는 라벨도 제거한다.
  디포로 돌아오면 reduced profit을 계산하고 유망한 트립을 마스터에 추가한다.
  빠른 모드는 양수 순가치 노드와 방문 개수 기반 지배를 사용하므로 놓치는 해가 있을 수 있다.
  빠른 모드가 빈손이면 정확 모드가 확인한다. 정확 모드가 빈손이면 개선 열이 없다.
  집합 간 지배는 최대 reduced profit과 개선 열의 존재 여부를 보존하며, 모든 집합별
  상위 N개 트립을 열거하는 것은 아니다. 최적성은 입력 아크 DB와 반올림된 점유 구간 안에서다.
  추진제는 실제 탱크 한도로만 검사한다. 예전 ΔV 합 한도(dv_sum) 모드는 탱크를 넘는
  트립도 허용해서 뺐다. 결과 JSON의 dv_sum은 고른 트립들의 총 ΔV 합(목적값)이다.
  라벨에는 kit = Σ p·(exp(D_k/c)-1)를 저장한다. 방문 개수가 다른 라벨도
  실제 추진제 기여분을 비교하도록 기존 raw kit 지배 조건을 수정했다.
  1단계 미증명 시에도 2단계는 위험지수 >= 현재 R* 아래 총 ΔV 최소라는 목적을 유지한다.
  초기 단독 트립까지 labeling2로 생성한다. 예전 경로 ILP 및 ILP solution pool gap 채우기는
  현재 실행 경로에 포함하지 않았으며 old/에 보관했다.

결과를 읽을 때
  ub_certified가 true일 때만 ub가 전체 LP의 인증된 상한이다.
  dv_lb_certified가 true일 때만 dv_lb가 전체 LP의 인증된 하한이다.
  반복 제한으로 끝나면 인증이 안 될 수 있다. 마지막에 추가한 열까지 반영한 LP를 저장한다.
  1단계의 정수 최적성은 proven_optimal로 확인한다. 2단계는 dv_gap도 확인한다.

검증
  .\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -v
  작은 문제의 모든 가능한 경로와 정확 pricing을 비교한다. 두 목적,
  음수 순가치 경유, 키트 지배 반례, 두 단계 CG의 상하한, CLI/JSON을 검증한다.
  Gurobi 테스트는 라이선스에 등록된 사용자 계정에서 실행해야 한다.
