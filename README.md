# Korean Classical Chinese Punctuation Prediction Model
# 한국 고전한문 구두점 예측 모델

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-yachagye-181717?logo=github)](https://github.com/yachagye/korean-classical-chinese-punctuation)

[English](#english) | [한국어](#korean)

---

## <a id="korean"></a>한국어

### 개요

한국 고전 한문 텍스트에 자동으로 구두점을 예측하는 딥러닝 모델입니다. Chinese-RoBERTa 기반 다중 레이블 토큰 분류 모델로, 선행 연구를 통해 축적된 교감표점 텍스트를 활용하여 7종의 구두점을 예측합니다.

**주요 활용 분야**:
- 텍스트 전처리 및 정규화
- 색인·검색 시스템 구축
- 번역 전처리
- OCR 후처리
- 디지털 인문학 연구

### 주요 특징

- **높은 정확도**: F1 Score 0.9050 달성
- **대규모 학습**: 4억 2천만 자, 340만 개 샘플
- **7종 구두점**: , 。 · ? ! 《 》
- **도메인 특화**: 실록, 등록, 일기, 문집 등 다양한 장르 지원
- **즉시 사용**: GUI 실행파일 제공

### 성능

**전체 성능 (검증 데이터)**
- F1 Score: 0.9050
- Precision: 0.9057
- Recall: 0.9043

**외부 검증 (미학습 데이터)**

고리점만 지정된 미학습 데이터를 대상으로 표점 위치 일치 성능 평가:

| 데이터셋 | F1 Score | 설명 |
|---------|----------|------|
| 한국문집총간 | 0.8784 | 문집류 고전 한문 |
| 일성록 | 0.9065 | 일기류 고전 한문 |

**구두점별 성능**

| 구두점 | F1 Score | Precision | Recall |
|--------|----------|-----------|--------|
| ? | 0.9436 | 0.9419 | 0.9454 |
| , | 0.9127 | 0.9130 | 0.9124 |
| 。 | 0.8818 | 0.9054 | 0.8594 |
| · | 0.8759 | 0.9157 | 0.8394 |
| 《 | 0.7367 | 0.8155 | 0.6717 |
| 》 | 0.7311 | 0.8024 | 0.6713 |
| ! | 0.6369 | 0.8114 | 0.5241 |

**도메인별 성능**

| 도메인 | F1 Score | 설명 |
|--------|----------|------|
| 실록 | 0.9162 | 조선왕조실록 등 편년체 사료 |
| 등록 | 0.9114 | 비변사등록 등 관청 등록류 |
| 지리지 | 0.9116 | 지리지, 읍지 등 |
| 전기 | 0.8606 | 인물 전기, 행장 등 |
| 법령 | 0.8485 | 법전, 의례서 등 |
| 문집 | 0.8354 | 한국문집총간 등 시문집 |
| 일기 | 0.8229 | 승정원일기, 개인 일기 등 |

### 📦 데이터 및 모델 다운로드

**Google Drive 공개 저장소**: https://drive.google.com/drive/folders/1WGueOa8Oz7kqv4ha7_9pgFRKOzXWId2H?usp=drive_link

모든 학습 데이터, 모델, 코드, 실행파일을 무료로 다운로드할 수 있습니다.

#### 폴더 구조

```
한국 고전한문 구두점 예측 모델/
│
├── 전처리 텍스트/              # 원본 교감표점 텍스트 (ZIP)
│   ├── 기타.zip
│   ├── 등록.zip                # 비변사등록 등
│   ├── 문집.zip                # 한국문집총간 등
│   ├── 법령.zip                # 법전류
│   ├── 연대기.zip              # 조선왕조실록
│   ├── 일기.zip                # 승정원일기 등
│   ├── 전기.zip                # 인물 전기
│   └── 지리지.zip              # 지리지, 읍지
│
├── 학습 데이터/                # 전처리 완료 JSONL (ZIP)
│   ├── train.zip              # 학습 데이터 (~340만 샘플)
│   └── val.zip                # 검증 데이터
│
├── 모델(.ckpt)/               # 학습된 모델 체크포인트
│   └── best_model_9050.zip   # F1: 0.9050 모델 (약 3.6GB)
│       └── checkpoint.ckpt
│
├── 코드/                      # 전체 소스코드
│   ├── [전처리 스크립트]
│   │   ├── 1_1_전처리_한글,가나,숫자행 제거.py
│   │   ├── 1_2_전처리_구두점 변환 26종.py
│   │   ├── 1_3_전처리_한자,구두점 26종 보존, 기타...
│   │   ├── 1_4_전처리_구두점 변환 7종.py
│   │   ├── 1_5_전처리_구두점 없는 행 제거_7종.py
│   │   └── 1_6_전처리_구두점 중복 제거_7종.py
│   │
│   ├── [학습 데이터 생성]
│   │   ├── 2_학습데이터생성_구두점7_jsonl.py
│   │   └── 3_학습데이터_검증_구두점7_jsonl.py
│   │
│   ├── [모델 학습 및 평가]
│   │   ├── 4_0_구두점_학습_v1_구두점7_Lightning.py
│   │   └── 6_F1 평가.py
│   │
│   └── [추론 및 활용]
│       ├── 구두점7_추론모델.py        # 핵심 추론 모듈
│       ├── 구두점7_지정_txt.py       # TXT 파일 처리
│       ├── 구두점7_지정_csv.py       # CSV 파일 처리
│       └── 구두점7_검증_위치정확도.py
│
└── 한국 고전한문 구두점 예측 프로그램 v1.0/
    ├── README.txt                    # 사용 설명서
    └── 한문구두점추론.zip             # Windows 실행파일 (3.6GB)
        └── 한문구두점추론.exe
```

#### 다운로드 가이드

**1. 실행파일만 필요한 경우 (일반 사용자)**
```
📥 다운로드: 한국 고전한문 구두점 예측 프로그램 v1.0/한문구두점추론.zip
📦 크기: 약 3.6GB
💻 용도: Windows에서 바로 실행 (Python 불필요)
```

**2. Python 코드 실행 (개발자)**
```
📥 다운로드: 
   - 코드/ 폴더 전체
   - 모델(.ckpt)/best_model_9050.zip
📦 크기: 약 3.7GB
💻 사용법:
   python 구두점7_지정_txt.py --checkpoint checkpoint.ckpt --input your_file.txt
```

**3. 모델 학습/연구 (AI 연구자)**
```
📥 다운로드:
   - 학습 데이터/train.zip, val.zip
   - 코드/4_0_구두점_학습_v1_구두점7_Lightning.py
   - 모델(.ckpt)/best_model_9050.zip (미세조정 시)
📦 크기: 약 6GB
💻 용도: 모델 재학습, 미세조정, 실험
```

**4. 원본 텍스트 연구 (역사학자/인문학자)**
```
📥 다운로드: 전처리 텍스트/ 폴더 (필요한 ZIP만)
📦 크기: 각 ZIP 100MB-2GB
💻 용도: 데이터 분석, 코퍼스 구축, 다른 연구 활용
```

**5. 완전 재현 (Full Reproduction)**
```
📥 다운로드: 전체 폴더
📦 크기: 약 10-15GB
💻 용도: 원본 데이터부터 모델 배포까지 전 과정 재현
📝 과정:
   1. 전처리 텍스트/ 압축 해제
   2. 전처리 스크립트 6단계 실행
   3. 학습 데이터 생성 (JSONL)
   4. 모델 학습 (Lightning)
   5. 평가 및 검증
```

#### 학습 데이터 상세 정보

**train.zip 압축 해제 시**: `train.jsonl` (약 2.5GB)
- 샘플 수: 약 340만 개
- 총 문자 수: 약 4억 2천만 자
- 형식: JSONL (한 줄에 한 샘플)

**JSONL 구조 예시**:
```json
{
  "text": "太祖康獻大王姓李諱成桂字君晉",
  "labels": [
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0],
    ...
  ],
  "length": 15,
  "source": "조선왕조실록",
  "domain": "실록"
}
```

**labels 인덱스**: [,  。 · ? ! 《 》]
- 예: `[1,0,0,0,0,0,0]` = 쉼표(,)
- 예: `[0,1,0,0,0,0,0]` = 마침표(。)

#### 원본 텍스트 ZIP 파일 정보

| ZIP 파일 | 압축 해제 후 | 주요 문헌 | 장르 |
|----------|-------------|----------|------|
| 연대기.zip | ~2GB | 조선왕조실록 | 편년체 사료 |
| 등록.zip | ~1.5GB | 비변사등록 | 관청 등록 |
| 일기.zip | ~1.2GB | 승정원일기 | 일기류 |
| 문집.zip | ~1GB | 한국문집총간 | 시문집 |
| 법령.zip | ~500MB | 경국대전 등 | 법전류 |
| 지리지.zip | ~300MB | 각 도 읍지 | 지리서 |
| 전기.zip | ~200MB | 행장, 묘지명 | 전기류 |
| 기타.zip | ~100MB | 기타 문헌 | 혼합 |

**압축 형식**: UTF-8 인코딩 TXT 파일
**구두점**: 원본 교감표점 (26종 → 전처리 후 7종으로 변환)

### 빠른 시작

#### 방법 1: Windows 실행 파일 (권장 - 일반 사용자)

```
1. Google Drive에서 "한문구두점추론.exe" 다운로드 (3.6GB)
2. ZIP 압축 해제
3. 한문구두점추론.exe 실행
4. GUI에서 파일 선택 → 처리 시작
```

**다운로드 링크**: [Google Drive](https://drive.google.com/drive/folders/1WGueOa8Oz7kqv4ha7_9pgFRKOzXWId2H?usp=drive_link) (별도 제공)

#### 방법 2: Python 코드 실행 (개발자/연구자)

**Python 코드**

```python
from 구두점7_추론모델 import PunctuationPredictor

# 모델 로드
predictor = PunctuationPredictor(
    checkpoint_path="path/to/checkpoint.ckpt"
)

# 예측
text = "太祖康獻大王姓李諱成桂字君晉古諱旦號松軒"
result = predictor.predict(text)
print(result)
# 출력: 太祖康獻大王, 姓李, 諱成桂, 字君晉。古諱旦, 號松軒。
```

**GUI 실행파일**

```bash
# Windows용 실행파일 빌드
python 구두점_지정_실행파일_빌드.py

# 실행
./dist/한문구두점추론.exe
```

### 학습 데이터

**출처**
- 조선왕조실록 (전체) - 연대기류
- 비변사등록 (전체) - 등록류
- 승정원일기 (전체) - 일기류
- 한국사료총서 (일기류, 문집)
- 한국사데이터베이스 (법령, 등록)

**규모**
- 총 문자 수: 약 4억 2천만 자
- 학습 샘플: 약 340만 개
- 데이터 유형: 8개 장르 (연대기, 문집, 일기, 등록, 법령, 지리지, 전기 등)
- 구두점 종류: 7종 (, 。 · ? ! 《 》)

**전처리**
- 교감표점 텍스트 수집·정제
- 7종 표점으로 표준화
- 6단계 전처리 파이프라인

### 모델 아키텍처

- **Base Model**: Chinese-RoBERTa (`hfl/chinese-roberta-wwm-ext`)
- **Task**: Multi-label Classification
- **Labels**: 7 punctuation marks
- **Training**:
  - GPU: L40S 48GB
  - Batch Size: 160 (effective)
  - Learning Rate: 2e-5
  - Epochs: 3
  - Mixed Precision: bf16

### 디렉토리 구조

```
korean-classical-chinese-punctuation/
├── data/                    # 데이터 전처리 스크립트
├── models/                  # 학습 및 추론 코드
├── evaluation/              # 평가 스크립트
├── applications/            # GUI 및 응용 프로그램
├── results/                 # 평가 결과
└── docs/                    # 문서
```

### 인용

이 모델을 사용하시는 경우 다음과 같이 인용해주세요:

**APA 스타일:**
```
양정현 (2025). 딥러닝 기반 한국 고전한문 표점 추론 자동화 모델의 구축과 활용. 
역사학연구, 100, [페이지]. [DOI 발급 후 추가]
```

**BibTeX:**
```bibtex
@article{yang2025punctuation,
  title={딥러닝 기반 한국 고전한문 표점 추론 자동화 모델의 구축과 활용},
  author={양정현},
  journal={역사학연구},
  year={2025},
  volume={100},
  publisher={호남사학회}
}
```

**논문 정보:**
- 저널: 역사학연구 (The Korean Journal of History)
- 권호: 100호
- 발행: 2025년 11월
- 출판사: 호남사학회
- KCI 저널: https://www.kci.go.kr/kciportal/po/search/poCitaView.kci?sereId=001257

### 라이선스 및 사용 조건

**라이선스**: CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike)

#### ✅ 허용되는 사용

**학술 연구**:
- 논문 작성 및 인용
- 학술 발표 및 교육
- 연구 목적 수정 및 개선
- 비영리 연구 프로젝트

**비영리 활용**:
- 교육 기관의 교재 및 실습
- 공공 기관의 디지털 아카이브 구축
- 오픈소스 프로젝트 통합
- 문화재 디지털화 사업

#### ❌ 제한되는 사용

**상업적 이용**:
- 유료 서비스 또는 제품 판매
- 기업의 영리 목적 활용
- 상업적 라이선스 재배포
- 광고 수익 목적 사용

**상업적 이용 문의**: yachagye@naver.com
- 개별 협의를 통해 상업적 라이선스 부여 가능
- 연구재단 지원 프로젝트 성과 활용 규정 준수

#### 📋 조건

1. **저작자 표시** (Attribution): 
   - 원저작자 및 출처 명시
   - 논문 인용 필수

2. **비영리** (NonCommercial):
   - 상업적 목적 사용 금지
   - 사전 협의 필요

3. **동일 조건 변경 허락** (ShareAlike):
   - 파생 저작물도 같은 라이선스 적용
   - 오픈소스 정신 계승

**전체 라이선스 조문**: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

### 프로젝트 정보

- **연구 기관**: 순천대학교 지리산권문화연구원
- **연구 과제**: 전통 차 제조기술의 역사적 복원과 현대적 계승을 위한 DB 구축 (한국연구재단)

### 향후 개선 과제

논문에서 제안된 향후 연구 방향:

1. **이중 경로 구조 (Two-Track System)**
   - 쌍 구조 표점(《》) 성능 개선
   - 장거리 의존성 모델링 강화

2. **문헌 유형별 적응형 모듈**
   - 도메인별 특화 미세조정
   - 장르 적응형 아키텍처

3. **다중과제 통합**
   - 문장 구조 분석과의 결합
   - 개체명 인식(NER) 통합
   - Multi-task Learning 구조

### 제한 사항

1. **쌍 구조 표점**: 서명 인용부호(《》)는 F1 ~0.73으로 다른 구두점 대비 낮은 성능
2. **희소 데이터**: 느낌표(!)는 학습 데이터 부족으로 재현율 저하
3. **컨텍스트 윈도우**: 최대 512 토큰 (약 450-500자) 제한
4. **도메인 편향**: 공식 기록물 중심 학습으로 사적 문헌에서 성능 저하 가능

### 문의

- **개발자**: 양정현 (순천대학교 지리산권문화연구원 학술연구교수)
- **이메일**: yachagye@naver.com
- **GitHub**: https://github.com/yachagye/korean-classical-chinese-punctuation
- **Issues**: https://github.com/yachagye/korean-classical-chinese-punctuation/issues
- **상업적 이용 문의**: 이메일로 사전 협의

### 면책 조항

본 프로그램의 구두점 예측 결과는 완벽하지 않을 수 있습니다. 중요한 학술 자료 또는 출판물에 사용하실 경우, 반드시 전문가의 검토를 거쳐 사용하시기 바랍니다.

---

## <a id="english"></a>English

### Overview

A deep learning model for automatic punctuation prediction in Korean Classical Chinese texts. Based on Chinese-RoBERTa with multi-label classification for 7 types of punctuation marks.

### Key Features

- **High Accuracy**: F1 Score 0.9050
- **Large-scale Training**: 420M characters, 3.4M samples
- **7 Punctuation Types**: , 。 · ? ! 《 》
- **Domain-specific**: Supports various genres (chronicles, diaries, collections)
- **Ready-to-use**: GUI application provided

### Performance

**Overall Performance**
- F1 Score: 0.9050
- Precision: 0.9057
- Recall: 0.9043

**Per-punctuation Performance**

| Punctuation | F1 Score | Precision | Recall |
|-------------|----------|-----------|--------|
| ? | 0.9436 | 0.9419 | 0.9454 |
| , | 0.9127 | 0.9130 | 0.9124 |
| 。 | 0.8818 | 0.9054 | 0.8594 |
| · | 0.8759 | 0.9157 | 0.8394 |
| 《 | 0.7367 | 0.8155 | 0.6717 |
| 》 | 0.7311 | 0.8024 | 0.6713 |
| ! | 0.6369 | 0.8114 | 0.5241 |

### Quick Start

#### Installation

```bash
git clone https://github.com/yachagye/korean-classical-chinese-punctuation.git
cd korean-classical-chinese-punctuation
pip install -r requirements.txt
```

#### Download Model

Download from Google Drive: https://drive.google.com/drive/folders/1WGueOa8Oz7kqv4ha7_9pgFRKOzXWId2H?usp=drive_link

#### Usage Example

```python
from 구두점7_추론모델 import PunctuationPredictor

predictor = PunctuationPredictor(checkpoint_path="path/to/checkpoint.ckpt")
text = "太祖康獻大王姓李諱成桂字君晉古諱旦號松軒"
result = predictor.predict(text)
print(result)
```

### Citation

```bibtex
@article{yang2025punctuation,
  title={Development and Application of a Deep Learning–Based Model for Automated Punctuation Inference in Korean Classical Chinese},
  author={Yang, Jeonghyeon},
  journal={The Korean Journal of History (Yoksahak Yongu)},
  year={2025},
  volume={100},
  publisher={Honam Historical Society}
}
```

### License

CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike)

For commercial use, please contact: yachagye@naver.com

### Contact

- **Author**: Jeonghyeon Yang
- **Email**: yachagye@naver.com
- **GitHub**: https://github.com/yachagye/korean-classical-chinese-punctuation
