# Raman-Spectroscopy 안정화 패치

대상 저장소: `slimphy/Raman-Spectroscopy`의 2026-07-13 기준 `main` 브랜치 구조

이 패키지는 간헐적인 전체 PC 느려짐·멈춤, GUI 응답 없음, Python 프로세스 종료 가능성을 줄이기 위한 안정화 파일입니다. 원본 프로젝트 전체를 임의로 재작성하지 않고, 위험도가 높은 모듈은 완전 교체하고 `main.py`는 자동 적용 스크립트로 안전하게 변환합니다.

## 포함 파일

- `camera_controller.py`: DCAM 호출 직렬화, 짧고 제한된 프레임 대기, 스캔 작업의 카메라 독점권, 안전한 캡처/버퍼 종료
- `stage_controller.py`: 시리얼 연결·쓰기·응답·종료의 단일 잠금, 제한 시간, 전송 확인 후 좌표 캐시 갱신
- `raman_ml.py`: ML 추론 잠금, `torch.inference_mode()`, 안전한 가중치 로딩, CUDA 실패 시 CPU 폴백
- `stability_utils.py`: 회전 로그, 전역 예외 기록, 대용량 스펙트럼 큐브의 `float32`/`memmap`, 안전한 수식 계산, 종료 순서 제어
- `apply_stability_patch.py`: 원본 파일 백업 후 위 모듈을 복사하고 `main.py`를 자동 패치
- `self_test.py`: 실제 하드웨어 없이 핵심 동작을 검사
- `APPLY_PATCH.bat`, `RUN_SELF_TEST.bat`: Windows 편의 실행 파일
- `DIAGNOSIS_KO.md`: 진단 근거, 수정 내용, 기대 효과, 잔여 위험
- `ROLLBACK_KO.md`: 원복 방법

## 권장 적용 순서

1. 현재 정상 동작하는 프로젝트 폴더를 별도 복사하거나 Git commit을 만듭니다.
2. Raman 프로그램과 HCImage Live 등 카메라를 사용하는 프로그램을 모두 종료합니다.
3. 명령 프롬프트에서 먼저 하드웨어 없는 검사를 실행합니다.

```bat
cd Raman-Spectroscopy-stability-patch
python self_test.py
```

4. 패치가 현재 소스 구조와 맞는지 쓰기 없이 확인합니다.

```bat
python apply_stability_patch.py "C:\path\to\Raman-Spectroscopy" --dry-run
```

5. 실제 적용합니다.

```bat
python apply_stability_patch.py "C:\path\to\Raman-Spectroscopy"
```

또는 `APPLY_PATCH.bat`에 프로젝트 폴더를 끌어다 놓을 수 있습니다.

6. 문법을 확인합니다.

```bat
cd /d "C:\path\to\Raman-Spectroscopy"
python -m py_compile main.py camera_controller.py stage_controller.py raman_ml.py stability_utils.py
```

## 하드웨어 검증 순서

한 번에 큰 맵을 실행하지 말고 아래 순서로 검증하십시오.

1. 카메라 연결/해제 5회 반복
2. 라이브뷰 10분 실행, ROI·노출 변경
3. 스테이지 단독 이동과 정지
4. 소프트웨어 트리거 3×3 소형 맵
5. 하드웨어 트리거 3×3 소형 맵
6. 스캔 도중 중지, 프로그램 종료, 재실행
7. 실제 크기의 10% 맵 후 전체 맵

오류가 생기면 프로젝트 폴더의 `logs\raman_app.log`와 콘솔 메시지를 확인하십시오.

## 설정값

Windows에서 실행 전에 환경 변수로 조절할 수 있습니다.

```bat
set RAMAN_CAMERA_BUFFER_COUNT=4
set RAMAN_TRIGGER_SETTLE_SEC=0.10
set RAMAN_MAX_CUBE_RAM_MB=512
set RAMAN_MAX_CUBE_DISK_GB=32
set RAMAN_ML_FORCE_CPU=0
python main.py
```

- `RAMAN_MAX_CUBE_RAM_MB`: 이 크기 이하만 RAM에 저장합니다. 초과하면 `raman_cache`의 디스크 매핑 파일을 사용합니다.
- `RAMAN_MAX_CUBE_DISK_GB`: 실수로 지나치게 큰 전체 스펙트럼 맵을 생성하지 못하게 막는 상한입니다.
- `RAMAN_ML_FORCE_CPU=1`: CUDA/드라이버 문제를 분리 진단할 때 사용합니다.

## 중요한 제한

- 실제 ORCA-Quest2/DCAMAPI, CoaXPress 보드, 피에조 스테이지가 연결된 상태의 장시간 검증은 이 환경에서 수행하지 못했습니다.
- 자동 패처는 2026-07-13에 확인한 코드 패턴을 기준으로 합니다. 이후 저장소가 크게 변경되었다면 `--dry-run` 결과에서 `SKIP(not found)` 또는 오류를 확인한 뒤 수동 병합이 필요합니다.
- 디스크 매핑은 RAM 폭증을 막지만, 전체 스펙트럼 저장량 자체가 크면 디스크 쓰기 속도와 용량이 병목이 됩니다.
