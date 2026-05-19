# RL Sim-to-Finance

강화학습 기반 시뮬레이션 환경에서 학습한 에이전트를 금융 도메인에 적용하는 프로젝트입니다.
NVIDIA Omniverse의 Sim-to-Real 철학을 금융 시장에 응용합니다.

---

## 프로젝트 개요

시뮬레이션 환경(LunarLander-v2)에서 학습한 PPO 에이전트를 커스텀 주식 트레이딩 환경에 전이(transfer)하고,
실사용 가능한 웹 인터페이스를 통해 인터랙티브하게 결과를 확인할 수 있습니다.

```
[Simulation Env]          [Finance Domain]
LunarLander-v2   →→→→→   Stock Trading Env
PPO Agent        transfer  PPO Agent
물리 기반 제어            매수/매도/홀드 결정
```

---

## 주요 기능

- LunarLander-v2 환경에서 PPO 에이전트 학습 및 시각화
- yfinance 기반 실제 주가 데이터로 커스텀 트레이딩 환경 구성
- 동일 PPO 구조로 시뮬레이션 → 금융 도메인 전이 실험
- Streamlit 웹 앱: 종목 입력 → 에이전트 트레이딩 결과 실시간 시각화

---

## 프로젝트 구조

```
rl-sim-to-finance/
├── README.md
├── requirements.txt
├── src/
│   ├── envs/
│   │   ├── trading_env.py          # 커스텀 주식 트레이딩 gym 환경
│   │   └── healthcare_env.py       # (2차 개발 예정)
│   ├── agents/
│   │   ├── ppo_agent.py            # PPO 에이전트
│   │   └── sac_agent.py            # (2차 개발 예정)
│   ├── data/
│   │   └── fetcher.py              # yfinance 데이터 수집
│   └── utils/
│       └── visualizer.py           # 학습 결과 시각화
├── app/
│   └── streamlit_app.py            # 웹 데모
├── notebooks/
│   ├── 01_lunarlander_ppo.ipynb    # 1단계: 시뮬레이션 환경 학습
│   └── 02_trading_ppo.ipynb        # 2단계: 금융 도메인 적용
└── results/
    ├── plots/                       # 학습 커브, 시각화 결과
    └── models/                      # 학습된 모델 저장
```

---

## 실행 환경

- Python 3.10+
- Google Colab (T4 GPU) 또는 로컬 환경

### 의존성 설치

```bash
pip install -r requirements.txt
```

### 웹 앱 실행

```bash
streamlit run app/streamlit_app.py
```

---

## 실험 결과

> 학습 완료 후 업데이트 예정

| 환경 | 알고리즘 | 최종 평균 리워드 | 수렴 에피소드 |
|------|----------|----------------|-------------|
| LunarLander-v2 | PPO | - | - |
| Stock Trading | PPO | - | - |

---

## 로드맵

### 1차 (현재)
- [x] 프로젝트 구조 설계
- [ ] LunarLander PPO 학습
- [ ] 커스텀 트레이딩 환경 구현
- [ ] Streamlit 웹 데모

### 2차 (예정)
- [ ] 알고리즘 비교 (PPO vs SAC vs A2C)
- [ ] 멀티모달 입력 (뉴스 텍스트 + 가격 데이터)
- [ ] 헬스케어 도메인 확장
- [ ] Omniverse / MuJoCo 시뮬레이션 환경 고도화

---

## 참고 자료

- [Stable-Baselines3 문서](https://stable-baselines3.readthedocs.io/)
- [Gymnasium 문서](https://gymnasium.farama.org/)
- [NVIDIA Omniverse Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [yfinance](https://github.com/ranaroussi/yfinance)
