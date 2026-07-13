# 원복 방법

`apply_stability_patch.py`는 덮어쓰는 각 원본 파일 옆에 다음 형식의 백업을 만듭니다.

```text
main.py.bak_YYYYMMDD_HHMMSS
camera_controller.py.bak_YYYYMMDD_HHMMSS
stage_controller.py.bak_YYYYMMDD_HHMMSS
raman_ml.py.bak_YYYYMMDD_HHMMSS
```

프로그램을 종료한 뒤 같은 타임스탬프의 파일을 원래 이름으로 복사하십시오.

예시:

```bat
copy /Y main.py.bak_20260713_120000 main.py
copy /Y camera_controller.py.bak_20260713_120000 camera_controller.py
copy /Y stage_controller.py.bak_20260713_120000 stage_controller.py
copy /Y raman_ml.py.bak_20260713_120000 raman_ml.py
del stability_utils.py
```

Git 저장소에서 변경 전 commit이 있다면 다음 방식이 더 안전합니다.

```bat
git status
git diff
git restore main.py camera_controller.py stage_controller.py raman_ml.py
del stability_utils.py
```

측정 데이터와 모델 파일은 원복 대상에 포함되지 않습니다.
