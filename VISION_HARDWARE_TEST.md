# Vision CCD + CSN210 독립 테스트 프로그램

`main.py`와 연결되지 않은 장비 검증용 GUI입니다.

## 실행

```powershell
.\.venv\Scripts\python.exe vision_hardware_test.py
```

카메라가 연결되어 있지 않을 때는 장치 목록에서 `Simulation`을 선택하면 GUI,
십자선, 캘리브레이션, ROI 기능을 먼저 시험할 수 있습니다.

## 카메라

- `재검색`에서 IC4가 인식한 DFK 37AUX290을 찾습니다.
- `Auto brightness (노출 + 게인)`은 DFK 카메라의 `ExposureAuto`와 `GainAuto`를
  함께 `Continuous`로 설정합니다. 체크를 끄면 두 자동 기능을 `Off`로 바꾸어
  현재 노출과 게인 값을 유지합니다.
- `중앙 십자선`으로 센서 영상의 중앙을 표시합니다.

카메라가 목록에 없으면 The Imaging Source의 USB3 Vision GenTL 드라이버와 IC4
지원 여부를 먼저 확인하세요. 이 프로그램은 일반 웹캠 번호 대신 공식
`imagingcontrol4` API의 모델/시리얼 열거를 사용합니다.

## 캘리브레이션

1. `20X` 또는 `100X`를 선택합니다.
2. 표준물에서 알고 있는 실제 길이를 µm 단위로 입력합니다.
3. `선 길이 측정 모드`를 켜고 영상에서 해당 길이의 양 끝을 드래그합니다.
   측정선 양 끝에는 선과 수직인 엔드캡이 표시되어 `H` 모양으로 보입니다.
4. 같은 길이를 여러 번 재면 `µm/px` 평균이 자동으로 계산됩니다.
5. `vision_calibration 저장`을 누르면 두 배율의 샘플과 평균이
   `vision_calibration.json`에 함께 저장됩니다.

## Mapping ROI

`사각형 영역 선택`을 켜고 드래그하면 원본 카메라 픽셀 좌표 기준 중심점,
X/Y 길이, 현재 배율의 보정값을 적용한 µm 길이를 표시합니다. `ROI JSON 복사`로
향후 매핑 코드에 넣을 좌표를 복사할 수 있습니다.

## CSN210

1. `CSN210 연결`을 누릅니다.
2. `Home`을 누르고 Homed 상태가 `Yes`가 될 때까지 기다립니다.
3. `Position 1 (20X)` 또는 `Position 2 (100X)`를 누릅니다.

연결하기 전에 제조사 `CSN210_Control.exe`를 종료해야 합니다. 두 프로그램이 같은
USB 장치를 동시에 열 수 없기 때문에, 실행 중인 경우 테스트 GUI가 연결 대신 안내를
표시합니다. DLL 호출은 백그라운드에서 실행되므로 장치 응답이 늦어도 영상 GUI는
멈추지 않습니다. 테스트 앱을 닫으면 CSN210 세션도 종료됩니다.

충돌이 표시되면 위치 이동 버튼이 비활성화됩니다. 간섭 원인을 제거한 뒤 다시
Home 하세요. 이 장비 구성에서는 Position 1이 20X, Position 2가 100X입니다.

메인 프로그램의 `Vision & ROI` 탭에서 objective를 전환하면 CSN210 이동 완료를
확인한 뒤 piezo stage에 다음 상대 보정을 적용합니다.

- 100X -> 20X: X +68 µm, Y +31 µm, Z -72 µm
- 20X -> 100X: X -68 µm, Y -31 µm, Z +72 µm

Vision ROI의 `Send Map Area`는 현재 배율의 `vision_calibration.json` 값을 사용해
영상 중심 기준 X/Y 범위를 계산하고, 소수점 첫째 자리까지 반올림한 값을
`Mapping & Scan` 탭의 Start/End에 넣습니다. Vision 탭의 ROI 상세 표시는 기존처럼
소수점 셋째 자리까지 유지됩니다. 피에조 좌표계에 맞춰 영상 오른쪽은 `+X`,
영상 위쪽은 `+Y`, 영상 아래쪽은 `-Y`로 변환합니다. Mapping 스캔을 위해 최종
Y Start/End는 작은 값에서 큰 값 순서로 전달됩니다.

Vision ROI를 보낼 때 현재 stage X/Y를 영상 중심의 기준 좌표로 더합니다. 그 뒤
objective를 전환하면 기존에 보낸 Mapping X/Y Start/End에도 objective 보정 이동량을
더해 동일한 시료 영역을 유지합니다. 100X에서 설정한 Z 영점을 사용하므로 Mapping의
Z Start/End 값은 objective 전환 시 변경하지 않습니다.

## Mapping 결과 행 보정

메인 프로그램의 `Mapping & Scan` 탭에서 `교대 행 X 보정`을 켜고
`홀수 -1 / 짝수 +1`을 선택하면 1, 3, 5...번째 행은 X축으로 한 칸 왼쪽,
2, 4, 6...번째 행은 한 칸 오른쪽으로 동시에 이동합니다. 원본 map 배열은
변경하지 않으며 화면 표시, 보정 화면의 스펙트럼 클릭 좌표, 현재 표시 데이터 CSV
저장에 적용됩니다. 이동 후 비게 되는 양 끝 셀은 `NaN`으로 처리됩니다.

Mapping의 자동 Raw CSV와 `현재 표시된 데이터 저장`으로 생성하는 CSV/PNG는 모두
프로젝트의 `mapping_results` 폴더에 저장됩니다. CSN210 SDK가 작업 폴더를 잠시
변경하더라도 이 절대 경로는 영향을 받지 않습니다.

SDK DLL 기본 경로는 다음과 같습니다.

```text
C:\Program Files\Thorlabs\CSN210 4.0\bin\x64\ThorObjectiveChanger.dll
```

다른 위치를 사용하려면 환경 변수 `CSN210_DLL`에 DLL 전체 경로를 설정하세요.
