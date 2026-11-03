# Jetson 카메라 응용 실습 (EmbeddingGemma 2 8~10단계) — AIGJ_01_001

Jetson Orin Nano Super 8GB에 USB 웹캠을 붙여 실시간 물체 추출·검색(8단계), 사람 동작 인식(9단계), 카메라 2대 접근 감지(10단계)를 해 본 과제 저장소다.

> **범위**: 강의자료 "Jetson 실습 - EmbeddingGemma 2 멀티모달 임베딩" 중 카메라 응용(8~10단계).
> 앞부분(1~7단계)은 [embedding_gemma2-AIGJ_01_001](https://github.com/AIP-1/embedding_gemma2-AIGJ_01_001) 에 따로 제출했다. 8~10단계도 1~2단계 설치와 `lab/eg2.py` 임베딩 로더를 그대로 쓰므로 코드 전체가 함께 들어 있고, 두 저장소의 커밋 이력이 같다.

## 요약

- 웹캠 물체 검색(8단계): 라벨 목록 밖 물체는 가장 비슷한 이름으로 억지로 붙는다. YOLOE 박스 + 임베딩 재라벨(`--mode yoloe --relabel`)이 캔·컵·휴대폰을 모두 맞혔다. 약 12 fps, 크롭당 0.75 s, RAM 3.5 GB
- 사람 동작 인식(9단계): 프레임 단위 판정은 횟수가 실제의 6~16배로 부풀고, 시간 안정화로 1/3~1/4로 줄었다. 학습 모델(GRU)은 같은 녹화 검증 100%였지만 처음 보는 위치에서 25%, 3m 실시간 14%
- 두 카메라 접근 감지(10단계, 김민서와 공동): 렌즈 보정 RMS 0.063px(1호기 카메라)·1.81px(2호기 카메라), 바닥 보드 위치 보정을 두 번 해서 위치 차이 0.1cm 이하. 한 카메라 발목 방식(ankle1)은 0.5·1.0m 자리를 0.76·1.28m로 재서 약 +27cm의 일정한 오차(25.5 fps), 그중 +13~17cm는 발목 높이 편향으로 계산됐다. 카메라를 바닥 쪽으로 숙인 구도 때문에 2m 이상과 두 카메라 삼각측량(stereo·fused)은 재지 못했다(stereo 21.7 fps, 카메라 시간 차 15.5 ms). 10-6 접근 알람은 실측하지 못하고 합성 신호로 로직만 검증했다(5개 시나리오 15/15, 버그 1건 수정)
- 자세한 결과, 실패 → 원인 → 해결: [reports/step8.md](reports/step8.md) (8단계), [reports/step9.md](reports/step9.md) (9단계), [reports/step10/](reports/step10/README.md) (10단계)

## 역할

| 단계 | 담당 |
|---|---|
| 8단계 (웹캠 물체 검색: yolo·relabel·yoloe 비교, `--relabel` 확장) | 성찬 |
| 9단계 (사람 동작 인식: raw vs smooth 비교, 녹화·GRU 학습) | 성찬 |
| 10단계 (카메라 2대 근접 감지: 렌즈·위치 보정, 거리 측정) | 성찬, 김민서 (AIGJ_01_002) |
| 공통 기반 1~6단계 (환경, 임베딩 로더) | 민서 (iamminseo) |

10단계는 1호기·2호기 카메라를 한 대씩 써서 김민서와 함께 진행했다. 코드·결과·로그는 [reports/step10/](reports/step10/README.md) 에 있다.

## 환경

- Jetson Orin Nano Engineering Reference Developer Kit Super, RAM 8GB
- JetPack 7.2.1 (L4T R39.2.1), Ubuntu 24.04, Python 3.12, CUDA 13.2
- torch 2.14.1+cu132, torchvision 0.29.1+cu132, transformers 5.20.1, sentence-transformers 6.2.0
- USB 웹캠 (`/dev/video0`)

## 설치

```bash
python3 -m venv ~/eg2 && source ~/eg2/bin/activate
pip install -U pip wheel
python -m pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/cu132
pip freeze | grep -E '^(torch|torchvision)==' > ~/eg2/torch-pin.txt
pip install -c ~/eg2/torch-pin.txt -U \
  "sentence-transformers[image,audio,video]>=6.1.0" transformers accelerate \
  soundfile librosa av pillow huggingface_hub psutil
pip uninstall -y torchaudio     # PyPI torchaudio(CUDA 13.0)가 torch(13.2)와 충돌
hf download google/embeddinggemma-2
```

## 실행 (`lab/` 폴더에서)

```bash
cd lab && mkdir -p media
python check.py                        # torch·CUDA·버전 확인
python test_text.py                    # 3단계: 한국어 텍스트 검색, 차원 768/256/128 비교
python record.py                       # 웹캠 10초 녹화 → media/clip.mp4, 2초마다 frame_XX.jpg
python test_vision2.py                 # 4단계: 이미지·동영상 임베딩과 검색
python test_vision3.py                 # 4단계 개선: 장면 맞춤 질의 정답률, prompt_name 유무 비교
python make_audio.py                   # 5단계: 경보음·잡음 생성 (16 kHz 모노)
arecord -D plughw:0,0 -f S16_LE -r 16000 -c 1 -d 5 media/speech.wav   # 웹캠 마이크로 5초 녹음
python test_audio.py                   # 5단계: 음성 검색
python test_audio2.py                  # 5단계 개선: 녹음 크기 확인, 소리 종류 vs 말 내용 질의 비교
for m in text image video audio; do python bench.py $m 2>/dev/null | grep RESULT; done   # 6단계 측정
python data/make_data.py && python eval_ir.py && python train_ft.py   # 7단계: 데이터 생성, 기준 평가, 미세 조정
python app.py --mode yoloe --relabel   # 8단계: 웹캠 물체 검색, http://<Jetson IP>:8200
python pose_app.py                     # 9단계: 사람 동작 인식, http://<Jetson IP>:8200
# 10단계: lab/prox/README.md 참고 (1호기 앱 + 2호기 cam_server.py 포트 8210)
```

`test_vision3.py` 의 질의와 정답 번호는 이 저장소의 [reports/frames/](reports/frames/) 장면 기준이다. 새로 녹화하면 장면에 맞게 바꿔야 한다.

공용 보드라서 모델을 올리기 전에 `free -h` 로 available이 2.5GB 이상인지 확인한다. 비전 모델을 돌리는 동안 보드 전체 RAM이 5.8GB까지 올라간다.

## 알아둘 점

- `{"video": "clip.mp4"}` 처럼 파일 경로로 넣으면 torchcodec이 필요한데, 이 보드에서는 NVIDIA판 ffmpeg 8 라이브러리를 못 찾아 실패한다. PyAV로 1초에 1장씩 프레임을 뽑아 `{"video": numpy 배열}` 로 넣으면 동작한다.
- `record.py` 로 쓴 mp4는 fps 메타데이터가 정확하지 않아, 읽을 때 15프레임 간격(=1 fps)으로 고정한다.
- 음성도 같은 이유로 `{"audio": "x.wav"}` 대신 soundfile로 읽은 배열을 넣는다. 이 보드의 ffmpeg에는 소리 생성 기능(lavfi)이 없어 `make_audio.py`로 만든다.
- 갑자기 전원이 나갔다 켜지면 GPU 초기화가 실패할 수 있다 (`torch.cuda.is_available()` False, dmesg `ACR bootstrap failed`). 전원을 끄고 다시 켜면 풀린다.
- 영상·모델 파일은 git에 올리지 않는다 (`.gitignore`).

## AI 활용

Claude가 명령·스크립트를 단계별로 제안하고, 민서가 Jetson에서 직접 실행해 출력으로 검증했다. 결과 문서의 숫자는 모두 실제 실행 출력에서 옮겼다.
