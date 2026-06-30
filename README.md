# Korean Classical Chinese Punctuation Prediction Model
# 한국 고전한문 구두점 예측 모델

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-yachagye-181717?logo=github)](https://github.com/yachagye/korean-classical-chinese-punctuation)
[![DOI](https://img.shields.io/badge/DOI-10.37924/JSSW.100.9-blue)](https://doi.org/10.37924/JSSW.100.9)
[![DOI](https://img.shields.io/badge/DOI-10.17948/kcs.2026.111.7-blue)](https://doi.org/10.17948/kcs.2026..111.7)

[English](#english) | [한국어](#korean)

---

## <a id="korean"></a>한국어

### 개요

한국 고전 한문 텍스트에 자동으로 구두점을 예측하는 딥러닝 모델입니다. 선행 연구를 통해 축적된 교감표점 텍스트를 활용하여 7종의 구두점을 예측합니다.

**주요 활용 분야**:
- 텍스트 전처리 및 정규화
- 색인·검색 시스템 구축
- 번역 전처리
- OCR 후처리
- 디지털 인문학 연구

### 주요 특징

- **높은 정확도**: F1 Score 0.9110 (v2.5)
- **대규모 학습**: 4억 2천만 자, 340만 개 샘플
- **7종 구두점**: , 。 · ? ! 《 》
- **도메인 특화**: 연대기, 등록, 일기, 문집 등 다양한 장르 지원
- **즉시 사용**: GUI 실행파일 제공

### 🔄 v2.0 → v2.5 업데이트

v2.5는 v2.0에서 공개된 SIKU-BERT 기반 한국 고전한문 구두점 예측 모델의 안정성과 재현성을 개선한 업데이트입니다.
모델 구조, 태스크 정의, 데이터 스킴은 v2.0과 동일하게 유지됩니다.

**Name**: 한국 고전한문 구두점 예측 모델 v2.5  
**Base Model**: SIKU-BERT/sikuroberta  
**Task**: Multi-label Token Classification  
**Labels**: 7개 (, 。 · ? ! 《 》)

**v2.5 수정사항**:
- PatternAwareLoss 제거
- BCEWithLogitsLoss 기반 표준 다중 라벨 손실 함수 사용
- 토큰 마스크 적용 후 가중 평균(masked mean) 방식으로 손실 계산
- 안정성 및 재현성 중심의 단순화된 학습 정책

> v2.0(PatternAwareLoss)과 v2.5(표준 BCE)는 동등한 성능을 보였으며, 닫힘 기호(》)에서 v2.0이 재현율의 소폭 우위를 나타낸 것 외에 유의미한 차이는 관찰되지 않았다. 충분한 학습 데이터와 도메인 특화 사전학습이 확보된 조건에서는 연속 표점의 구조적 패턴이 손실 함수의 명시적 규칙 없이도 학습 과정에서 확률적으로 수렴함을 확인하였다. 이에 추가 가정이 없는 단순한 학습 구조의 v2.5를 최종 모델로 채택하였다. (민족문화연구 111)

**검증 결과 - F1: 0.9110 (v2.5)**  
7종 구두점:
- ?: F1=0.9485, P=0.9435, R=0.9535
- ,: F1=0.9182, P=0.9181, R=0.9183
- 。: F1=0.8880, P=0.9103, R=0.8668
- ·: F1=0.8866, P=0.9133, R=0.8615
- 《: F1=0.7805, P=0.8178, R=0.7465
- 》: F1=0.7735, P=0.8108, R=0.7395
- !: F1=0.7015, P=0.8437, R=0.6003

### 모델 버전

| 버전 | 사전학습 모델 | F1 Score | 비고 |
|------|--------------|----------|------|
| **v2.5** | SikuRoBERTa (`SIKU-BERT/sikuroberta`) | **0.9110** | 최신 권장 (민족문화연구 111 게재) |
| v2.0 | SikuRoBERTa (`SIKU-BERT/sikuroberta`) | 0.9110 | 설계 검증(reference) 모델 (민족문화연구 111 게재) |
| v1.0 | Chinese-RoBERTa (`hfl/chinese-roberta-wwm-ext`) | 0.9050 | 역사학연구 100 게재 |

### 성능

**전체 성능**

| 버전 | F1 Score | Precision | Recall |
|------|----------|-----------|--------|
| **v2.5** | **0.9110** | - | - |
| v2.0 | 0.9110 | 0.9117 | 0.9103 |
| v1.0 | 0.9050 | 0.9057 | 0.9043 |

**구두점별 성능**

| 구두점 | v1.0 F1 | v2.0 F1 | v2.5 F1 | v1.0 Precision | v2.0 Precision | v2.5 Precision | v1.0 Recall | v2.0 Recall | v2.5 Recall |
|--------|---------|---------|---------|----------------|----------------|----------------|-------------|-------------|-------------|
| ? | 0.9436 | 0.9486 | 0.9485 | 0.9419 | 0.9437 | 0.9435 | 0.9454 | 0.9536 | 0.9535 |
| , | 0.9127 | 0.9182 | 0.9182 | 0.9130 | 0.9175 | 0.9181 | 0.9124 | 0.9190 | 0.9183 |
| 。 | 0.8818 | 0.8879 | 0.8880 | 0.9054 | 0.9109 | 0.9103 | 0.8594 | 0.8661 | 0.8668 |
| · | 0.8759 | 0.8864 | 0.8866 | 0.9157 | 0.9134 | 0.9133 | 0.8394 | 0.8609 | 0.8615 |
| 《 | 0.7367 | 0.7817 | 0.7805 | 0.8155 | 0.8204 | 0.8178 | 0.6717 | 0.7465 | 0.7465 |
| 》 | 0.7311 | 0.7764 | 0.7735 | 0.8024 | 0.8126 | 0.8108 | 0.6713 | 0.7432 | 0.7395 |
| ! | 0.6369 | 0.7007 | 0.7015 | 0.8114 | 0.8399 | 0.8437 | 0.5241 | 0.6011 | 0.6003 |

**도메인별 성능 (v1.0 → v2.5)**

| 도메인 | v1.0 F1 | v2.5 F1 | 개선폭 | 데이터 규모(총 문자 수) |
|--------|---------|---------|--------|------|
| 지리지 | 0.9116 | 0.9385 | +0.0269 | 501,942 |
| 연대기 | 0.9162 | 0.9199 | +0.0037 | 30,682,976 |
| 등록 | 0.9114 | 0.9257 | +0.0143 | 1,896,232 |
| 전기 | 0.8606 | 0.8953 | +0.0347 | 591,983 |
| 법령 | 0.8485 | 0.8799 | +0.0314 | 907,893 |
| 문집 | 0.8354 | 0.8693 | +0.0339 | 1,885,268 |
| 일기 | 0.8229 | 0.8515 | +0.0286 | 544,768 |

*v1.0에서 이미 0.91 이상의 안정적 성능을 보인 연대기는 소폭 개선에 그친 반면, 상대적으로 성능이 낮았던 일기·문집·법령·전기 유형에서 개선폭이 크게 나타난다(상세 분석은 민족문화연구 111 참조).*

**외부 검증 (미학습 데이터, v1)**

고리점만 지정된 미학습 데이터를 대상으로 표점 위치 일치 성능 평가:

| 데이터셋 | F1 Score | 데이터 규모(총 문자 수) | 출처 |
|---------|----------|------|------|
| 한국문집총간 | 0.8784 | 166,763,095 | 고전종합DB |
| 일성록 | 0.9065 | 6,743,710 | 규장각한국학연구원 |

### 📦 데이터 및 모델 다운로드

**Google Drive 공개 저장소**: https://drive.google.com/drive/folders/1WGueOa8Oz7kqv4ha7_9pgFRKOzXWId2H?usp=drive_link

#### 폴더 구조

```

한국 고전한문 구두점 예측 모델/
│
├── 전처리 텍스트/              # 전처리 완료 텍스트 (표점 ○, ZIP)
│   ├── 기타.zip
│   ├── 등록.zip
│   ├── 문집.zip
│   ├── 법령.zip
│   ├── 연대기.zip
│   ├── 일기.zip
│   ├── 전기.zip
│   └── 지리지.zip
│
├── 학습 데이터/                # 전처리 완료 JSONL (ZIP)
│   ├── train.zip              # 학습 데이터
│   └── val.zip                # 검증 데이터
│
├── 모델(.ckpt)/               # 학습된 모델 체크포인트
│   ├── v1.0_best_model_9050.zip    # v1: F1 0.9050 (논문 버전)
│   ├── v2.0_best_model_9110.zip    # v2.0: F1 0.9110 (v2 최초 공개)
│   └── v2.5_best_model_9110.zip    # v2.5: F1 0.9110 (최신 권장)
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
│   │   ├── 4_0_구두점_학습_v1_구두점7_ChineseRoBERTa_Lightning.py
│   │   ├── 4_0_구두점_학습_v2_1_구두점7_SikuRoBERTa_Lightning.py
│   │   ├── 4_0_구두점_학습_v2_2_구두점7_SikuRoBERTa_Lightning.py
│   │   └── 6_F1 평가.py
│   │
│   └── [추론 및 활용]
│       ├── 구두점7_추론모델.py        # 핵심 추론 모듈
│       ├── 구두점7_지정_txt.py       # TXT 파일 처리
│       ├── 구두점7_지정_csv.py       # CSV 파일 처리
│       ├── 구두점7_검증_위치정확도.py
│       ├── 구두점_지정_실행파일_빌드_v1_ChineseRoBERTa.py
│       └── 구두점_지정_실행파일_빌드_v2_SikuRoBERTa.py
│
└── 한국 고전한문 구두점 예측 프로그램 v1.0/
├── README_v1.0.txt                    # 사용 설명서
└── 한문구두점추론_v1.0.zip             # Windows 실행파일
└── 한문구두점추론.exe
├── README_v2.0.txt                    # 사용 설명서
└── 한문구두점추론_v2.0.zip             # Windows 실행파일
└── 한문구두점추론_v2.exe
├── README_v2.5.txt                    # 사용 설명서
└── 한문구두점추론_v2.5.zip             # Windows 실행파일
└── 한문구두점추론_v2_2.exe

```

#### 다운로드 가이드

**1. 실행파일만 필요한 경우 (일반 사용자)**
```

📥 다운로드: 한국 고전한문 구두점 예측 프로그램/한문구두점추론_v2.5.zip (v2.5 권장)
📦 크기: 약 3.6GB
💻 용도: Windows에서 바로 실행 (Python 불필요)

```

**2. Python 코드 실행 (개발자)**
```

📥 다운로드:

* 코드/ 폴더 전체
* 모델(.ckpt)/best_model_v2_5_9110.zip (v2.5 권장)
  💻 사용법:
  python 구두점7_지정_txt.py --checkpoint checkpoint.ckpt --input your_file.txt

```

**3. 모델 학습/연구 (AI 연구자)**
```

📥 다운로드:

* 학습 데이터/train.zip, val.zip
* 코드/4_0_구두점_학습_v1_구두점7_ChineseRoBERTa_Lightning.py (v1)
* 코드/4_0_구두점_학습_v2_구두점7_SikuRoBERTa_Lightning.py (v2)
* 모델(.ckpt)/ (미세조정 시)
  💻 용도: 모델 재학습, 미세조정, 실험

```

**4. 원본 텍스트 연구 (역사학자/인문학자)**
```

📥 다운로드: 전처리 텍스트/ 폴더 (필요한 ZIP만)
💻 용도: 데이터 분석, 코퍼스 구축, 다른 연구 활용

```

**5. 완전 재현 (Full Reproduction)**
```

📥 다운로드: 전체 폴더
💻 용도: 원본 데이터부터 모델 배포까지 전 과정 재현
📝 과정:

1. 전처리 텍스트/ 압축 해제
2. 전처리 스크립트 6단계 실행
3. 학습 데이터 생성 (JSONL)
4. 모델 학습 (Lightning)
5. 평가 및 검증

````

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
````

**labels 인덱스**: [,  。 · ? ! 《 》]

* 예: `[1,0,0,0,0,0,0]` = 쉼표(,)
* 예: `[0,1,0,0,0,0,0]` = 마침표(。)

#### 원본 텍스트 ZIP 파일 정보

| ZIP 파일  | 압축 해제 후 | 주요 문헌    |
| ------- | ------- | -------- |
| 연대기.zip | ~2GB    | 조선왕조실록 등 |
| 등록.zip  | ~1.5GB  | 각사등록     |
| 일기.zip  | ~1.2GB  | 묵재일기 등   |
| 문집.zip  | ~1GB    | 한국문집총간   |
| 법령.zip  | ~500MB  | 경국대전 등   |
| 지리지.zip | ~300MB  | 대동지지 등   |
| 전기.zip  | ~200MB  | 국조인물고 등  |
| 기타.zip  | ~100MB  |          |

* **압축 형식**: UTF-8 인코딩 TXT 파일
* **구두점**: 원본 교감표점 (26종 → 전처리 후 7종으로 변환)

### 빠른 시작

#### 방법 1: Windows 실행 파일 (권장 - 일반 사용자)

```
1. Google Drive에서 "한문구두점추론.exe" 다운로드
2. ZIP 압축 해제
3. 한문구두점추론.exe 실행
4. GUI에서 파일 선택 → 처리 시작
```

**다운로드 링크**: [Google Drive](https://drive.google.com/drive/folders/1WGueOa8Oz7kqv4ha7_9pgFRKOzXWId2H?usp=drive_link)

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
# Windows용 실행파일 빌드 (v1)
python 구두점_지정_실행파일_빌드_v1_ChineseRoBERTa.py

# Windows용 실행파일 빌드 (v2)
python 구두점_지정_실행파일_빌드_v2_SikuRoBERTa.py

# 실행
./dist/한문구두점추론.exe
```

### 학습 데이터

**출처**

* 국사편찬위원회 한국사데이터베이스([https://db.history.go.kr/](https://db.history.go.kr/))
* 한국고전종합DB([https://db.itkc.or.kr/](https://db.itkc.or.kr/))
* 한국학중앙연구원 디지털장서각([https://jsg.aks.ac.kr/](https://jsg.aks.ac.kr/))

**규모**

* 총 문자 수: 약 4억 2천만 자
* 학습 샘플: 약 340만 개
* 데이터 유형: 8개 장르 (연대기, 문집, 일기, 등록, 법령, 지리지, 전기 등)
* 구두점 종류: 7종 (, 。 · ? ! 《 》)

**전처리**

* 교감표점 텍스트 수집·정제
* 7종 표점으로 표준화
* 6단계 전처리 파이프라인

### 모델 아키텍처

**v2.5 (최신 권장)**

* **Base Model**: SikuRoBERTa (`SIKU-BERT/sikuroberta`)
* **Task**: Multi-label Classification
* **Labels**: 7 punctuation marks
* **Training**:

  * GPU: L40S 48GB
  * Batch Size: 160 (effective)
  * Learning Rate: 2e-5
  * Epochs: 3
  * Mixed Precision: bf16

**v2.0**

* **Base Model**: SikuRoBERTa (`SIKU-BERT/sikuroberta`)
* 기타 설정 동일

**v1 (논문 게재 버전)**

* **Base Model**: Chinese-RoBERTa (`hfl/chinese-roberta-wwm-ext`)
* 기타 설정 동일

### 디렉토리 구조

```
korean-classical-chinese-punctuation/
├── preprocessing/           # 전처리 스크립트 (1_1 ~ 1_6)
├── data_generation/         # 학습 데이터 생성 (2, 3)
├── training/                # 모델 학습 및 평가 (4_0, 6)
├── inference/               # 추론 및 활용 (구두점7_*)
└── build/                   # 실행파일 빌드 스크립트
```

### 인용

이 모델을 사용하시는 경우 다음과 같이 인용해주세요:

**APA 스타일:**

```
양정현 (2025). 딥러닝 기반 한국 고전한문 표점 추론 자동화 모델의 구축과 활용. 
역사학연구, 100, 267-297. https://doi.org/10.37924/JSSW.100.9

양정현 (2026). 해석의 관습: 한국 고전 한문 전용 표점 추론 모델의 개선. 
민족문화연구, 111, 7-29. https://doi.org/10.17948/kcs.2026..111.7
```

**BibTeX:**

```bibtex
@article{yang2025punctuation,
  title={딥러닝 기반 한국 고전한문 표점 추론 자동화 모델의 구축과 활용},
  author={양정현},
  journal={역사학연구},
  volume={100},
  pages={267--297},
  year={2025},
  publisher={호남사학회},
  doi={10.37924/JSSW.100.9}
}

@article{yang2026interpretation,
  title={해석의 관습: 한국 고전 한문 전용 표점 추론 모델의 개선},
  author={양정현},
  journal={민족문화연구},
  volume={111},
  pages={7--29},
  year={2026},
  publisher={고려대학교 민족문화연구원},
  doi={10.17948/kcs.2026..111.7}
}
```

**논문 정보:**

* v1.0 (선행 연구)

  * 저널: 역사학연구 (The Korean Journal of History)
  * 권호: 100호
  * 발행: 2025년 11월 30일
  * 출판사: 호남사학회
  * DOI: [10.37924/JSSW.100.9](https://doi.org/10.37924/JSSW.100.9)

* v2.0–v2.5 (개선 연구)

  * 저널: 민족문화연구 (Korean Cultural Studies)
  * 권호: 111호
  * 발행: 2026년
  * 출판사: 고려대학교 민족문화연구원
  * DOI: [10.17948/kcs.2026..111.7](https://doi.org/10.17948/kcs.2026..111.7)

### 라이선스 및 사용 조건

**라이선스**: CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike)

#### ✅ 허용되는 사용

**학술 연구**:

* 논문 작성 및 인용
* 학술 발표 및 교육
* 연구 목적 수정 및 개선
* 비영리 연구 프로젝트

**비영리 활용**:

* 교육 기관의 교재 및 실습
* 공공 기관의 디지털 아카이브 구축
* 오픈소스 프로젝트 통합
* 문화재 디지털화 사업

#### ❌ 제한되는 사용

**상업적 이용**:

* 유료 서비스 또는 제품 판매
* 기업의 영리 목적 활용
* 상업적 라이선스 재배포
* 광고 수익 목적 사용

**상업적 이용 문의**: [yachagye@naver.com](mailto:yachagye@naver.com)

* 개별 협의를 통해 상업적 라이선스 부여 가능
* 연구재단 지원 프로젝트 성과 활용 규정 준수

#### 📋 조건

1. **저작자 표시** (Attribution):

   * 원저작자 및 출처 명시
   * 논문 인용 필수

2. **비영리** (NonCommercial):

   * 상업적 목적 사용 금지
   * 사전 협의 필요

3. **동일 조건 변경 허락** (ShareAlike):

   * 파생 저작물도 같은 라이선스 적용
   * 오픈소스 정신 계승

**전체 라이선스 조문**: [https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

### 향후 개선 과제

논문에서 제안된 향후 연구 방향:

1. **이중 경로 구조 (Two-Track System)**

   * 쌍 구조 표점(《》) 성능 개선
   * 장거리 의존성 모델링 강화

2. **문헌 유형별 적응형 모듈**

   * 도메인별 특화 미세조정
   * 장르 적응형 아키텍처

3. **다중과제 통합**

   * 문장 구조 분석과의 결합
   * 개체명 인식(NER) 통합
   * Multi-task Learning 구조

### 제한 사항

1. **쌍 구조 표점**: 서명 인용부호(《》)는 F1 ~0.78로 다른 구두점 대비 낮은 성능
2. **희소 데이터**: 느낌표(!)는 학습 데이터 부족으로 재현율 저하
3. **모델 컨텍스트**: 512 토큰 단위로 처리 (슬라이딩 윈도우로 긴 텍스트 자동 처리)
4. **도메인 편향**: 공식 기록물 중심 학습으로 사적 문헌에서 성능 저하 가능

### 문의

* **개발자**: 양정현
* **이메일**: [yachagye@naver.com](mailto:yachagye@naver.com)
* **GitHub**: [https://github.com/yachagye/korean-classical-chinese-punctuation](https://github.com/yachagye/korean-classical-chinese-punctuation)
* **Issues**: [https://github.com/yachagye/korean-classical-chinese-punctuation/issues](https://github.com/yachagye/korean-classical-chinese-punctuation/issues)
* **상업적 이용 문의**: 이메일로 사전 협의

### 면책 조항

본 프로그램의 구두점 예측 결과는 완벽하지 않을 수 있습니다. 중요한 학술 자료 또는 출판물에 사용하실 경우, 반드시 전문가의 검토를 거쳐 사용하시기 바랍니다.

---

## <a id="english"></a>English

### Overview

A deep learning model for automatically predicting punctuation marks in Korean Classical Chinese texts. The model predicts 7 types of punctuation marks using collated punctuation texts accumulated through previous research.

**Key Applications**:

* Text preprocessing and normalization
* Index and search system construction
* Translation preprocessing
* OCR post-processing
* Digital humanities research

### Key Features

* **High Accuracy**: F1 Score 0.9110 (v2.5)
* **Large-scale Training**: 420M characters, 3.4M samples
* **7 Punctuation Types**: , 。 · ? ! 《 》
* **Domain-specific**: Supports various genres (chronicles, registers, diaries, collections)
* **Ready-to-use**: GUI executable provided

### 🔄 v2.0 → v2.5 Update

Version 2.5 is an incremental update of the v2.0 release, focusing on training stability and reproducibility.
The model architecture, task definition, and data scheme remain identical to v2.0.

**Name**: Korean Classical Chinese Punctuation Prediction Model v2.5
**Base Model**: SIKU-BERT/sikuroberta
**Task**: Multi-label Token Classification
**Labels**: 7 punctuation marks (, 。 · ? ! 《 》)

**v2.5 Updates**:

* Removal of PatternAwareLoss
* Adoption of standard BCEWithLogitsLoss for multi-label classification
* Masked-mean loss computation based on token attention masks
* Simplified training policy prioritizing stability and reproducibility

> v2.0 (PatternAwareLoss) and v2.5 (standard BCE) showed equivalent performance, with no significant difference observed apart from a slight recall advantage of v2.0 on the closing bracket (》). Given sufficient training data and domain-specific pretraining, the structural pattern of consecutive punctuation marks converges probabilistically during training even without an explicit loss-function rule. Accordingly, v2.5—the simpler training structure without additional assumptions—was adopted as the final model. (Korean Cultural Studies, vol. 111)

**Validation - F1: 0.9110 (v2.5)**
Per punctuation:

* ?: F1=0.9485, P=0.9435, R=0.9535
* ,: F1=0.9182, P=0.9181, R=0.9183
* 。: F1=0.8880, P=0.9103, R=0.8668
* ·: F1=0.8866, P=0.9133, R=0.8615
* 《: F1=0.7805, P=0.8178, R=0.7465
* 》: F1=0.7735, P=0.8108, R=0.7395
* !: F1=0.7015, P=0.8437, R=0.6003

### Model Versions

| Version  | Pre-trained Model                               | F1 Score   | Note                                          |
| -------- | ----------------------------------------------- | ---------- | --------------------------------------------- |
| **v2.5** | SikuRoBERTa (`SIKU-BERT/sikuroberta`)           | **0.9110** | Latest Recommended (Korean Cultural Studies 111) |
| v2.0     | SikuRoBERTa (`SIKU-BERT/sikuroberta`)           | 0.9110     | Reference Model (Korean Cultural Studies 111) |
| v1.0     | Chinese-RoBERTa (`hfl/chinese-roberta-wwm-ext`) | 0.9050     | Korean Journal of History 100                 |

### Performance

**Overall Performance**

| Version  | F1 Score   | Precision | Recall |
| -------- | ---------- | --------- | ------ |
| **v2.5** | **0.9110** | -         | -      |
| v2.0     | 0.9110     | 0.9117    | 0.9103 |
| v1.0     | 0.9050     | 0.9057    | 0.9043 |

**Per-punctuation Performance**

| Punctuation | v1.0 F1 | v2.0 F1 | v2.5 F1 | v1.0 Precision | v2.0 Precision | v2.5 Precision | v1.0 Recall | v2.0 Recall | v2.5 Recall |
| ----------- | ------- | ------- | ------- | -------------- | -------------- | -------------- | ----------- | ----------- | ----------- |
| ?           | 0.9436  | 0.9486  | 0.9485  | 0.9419         | 0.9437         | 0.9435         | 0.9454      | 0.9536      | 0.9535      |
| ,           | 0.9127  | 0.9182  | 0.9182  | 0.9130         | 0.9175         | 0.9181         | 0.9124      | 0.9190      | 0.9183      |
| 。           | 0.8818  | 0.8879  | 0.8880  | 0.9054         | 0.9109         | 0.9103         | 0.8594      | 0.8661      | 0.8668      |
| ·           | 0.8759  | 0.8864  | 0.8866  | 0.9157         | 0.9134         | 0.9133         | 0.8394      | 0.8609      | 0.8615      |
| 《           | 0.7367  | 0.7817  | 0.7805  | 0.8155         | 0.8204         | 0.8178         | 0.6717      | 0.7465      | 0.7465      |
| 》           | 0.7311  | 0.7764  | 0.7735  | 0.8024         | 0.8126         | 0.8108         | 0.6713      | 0.7432      | 0.7395      |
| !           | 0.6369  | 0.7007  | 0.7015  | 0.8114         | 0.8399         | 0.8437         | 0.5241      | 0.6011      | 0.6003      |

**Domain-specific Performance (v1.0 → v2.5)**

| Domain      | v1.0 F1 | v2.5 F1 | Δ F1    | Data Size (Total Characters) |
| ----------- | ------- | ------- | ------- | ---------------------------- |
| Gazetteers  | 0.9116  | 0.9385  | +0.0269 | 501,942                      |
| Chronicles  | 0.9162  | 0.9199  | +0.0037 | 30,682,976                   |
| Registers   | 0.9114  | 0.9257  | +0.0143 | 1,896,232                    |
| Biographies | 0.8606  | 0.8953  | +0.0347 | 591,983                      |
| Legal Codes | 0.8485  | 0.8799  | +0.0314 | 907,893                      |
| Collections | 0.8354  | 0.8693  | +0.0339 | 1,885,268                    |
| Diaries     | 0.8229  | 0.8515  | +0.0286 | 544,768                      |

*Chronicles, which already exceeded 0.91 in v1.0, show only a marginal gain, whereas Diaries, Collections, Legal Codes, and Biographies—relatively lower-performing types—show larger gains (see Korean Cultural Studies, vol. 111, for detailed analysis).*

**External Validation (Unseen Data, v1)**

Performance evaluation on punctuation position matching for unseen data with only sentence markers:

| Dataset                     | F1 Score | Data Size (Total Characters) | Source                                  |
| --------------------------- | -------- | ---------------------------- | --------------------------------------- |
| Korean Literary Collections | 0.8784   | 166,763,095                  | ITKC Database                           |
| Ilseongrok                  | 0.9065   | 6,743,710                    | Kyujanggak Institute for Korean Studies |

### 📦 Data and Model Downloads

**Google Drive Public Repository**: [https://drive.google.com/drive/folders/1WGueOa8Oz7kqv4ha7_9pgFRKOzXWId2H?usp=drive_link](https://drive.google.com/drive/folders/1WGueOa8Oz7kqv4ha7_9pgFRKOzXWId2H?usp=drive_link)

All training data, models, code, and executables are available for free download.

#### Folder Structure

```
Korean Classical Chinese Punctuation Model/
│
├── Preprocessed Texts/          # Preprocessed texts file (ZIP)
│   ├── Miscellaneous.zip
│   ├── Registers.zip            
│   ├── Collections.zip          
│   ├── Legal Codes.zip          
│   ├── Chronicles.zip           
│   ├── Diaries.zip             
│   ├── Biographies.zip         
│   └── Gazetteers.zip
│
├── Training Data/               # Preprocessed JSONL (ZIP)
│   ├── train.zip               # Training data
│   └── val.zip                 # Validation data
│
├── Models(.ckpt)/              # Trained model checkpoints
│   ├── best_model_9050.zip     # v1: F1 0.9050 (Paper version)
│   ├── best_model_9110.zip     # v2.0: F1 0.9110 (Initial v2 release)
│   └── best_model_v2_5_9110.zip     # v2.5: F1 0.9110 (Latest recommended)
│
├── Code/                       # Complete source code
│   ├── [Preprocessing Scripts]
│   │   ├── 1_1_preprocessing_remove_korean_etc.py
│   │   ├── 1_2_preprocessing_convert_26_punctuations.py
│   │   ├── 1_3_preprocessing_preserve_chinese_26_punct.py
│   │   ├── 1_4_preprocessing_convert_7_punctuations.py
│   │   ├── 1_5_preprocessing_remove_unpunctuated_lines.py
│   │   └── 1_6_preprocessing_remove_duplicate_punct.py
│   │
│   ├── [Training Data Generation]
│   │   ├── 2_generate_training_data_7punct_jsonl.py
│   │   └── 3_validate_training_data_7punct_jsonl.py
│   │
│   ├── [Model Training and Evaluation]
│   │   ├── 4_0_train_punctuation_v1_7punct_ChineseRoBERTa_Lightning.py
│   │   ├── 4_0_train_punctuation_v2_1_7punct_SikuRoBERTa_Lightning.py
│   │   ├── 4_0_train_punctuation_v2_2_7punct_SikuRoBERTa_Lightning.py
│   │   └── 6_F1_evaluation.py
│   │
│   └── [Inference and Applications]
│       ├── punctuation_7_inference_model.py    # Core inference module
│       ├── punctuation_7_process_txt.py       # TXT file processing
│       ├── punctuation_7_process_csv.py       # CSV file processing
│       ├── punctuation_7_validate_accuracy.py
│       ├── build_executable_v1_ChineseRoBERTa.py
│       └── build_executable_v2_SikuRoBERTa.py
│
└── Korean Classical Chinese Punctuation Program/
    ├── README_v1.0.txt                    # User manual
    └── ChinesePunctuationInference_v1.0.zip   # Windows executable
        └── ChinesePunctuationInference.exe
    ├── README_v2.0.txt                    # User manual
    └── ChinesePunctuationInference_v2.0.zip   # Windows executable
        └── ChinesePunctuationInference_v2.exe
    ├── README_v2.5.txt                    # User manual
    └── ChinesePunctuationInference_v2.5.zip   # Windows executable
        └── ChinesePunctuationInference_v2_2.exe
```

#### Download Guide

**1. Executable Only (General Users)**

```
📥 Download: Korean Classical Chinese Punctuation Program/ChinesePunctuationInference_v2.5.zip (v2.5 recommended)
📦 Size: ~3.6GB
💻 Purpose: Run directly on Windows (Python not required)
```

**2. Python Code Execution (Developers)**

```
📥 Download: 
   - Code/ folder (all files)
   - Models(.ckpt)/best_model_v2_5_9110.zip (v2.5 recommended)
💻 Usage:
   python punctuation_7_process_txt.py --checkpoint checkpoint.ckpt --input your_file.txt
```

**3. Model Training/Research (AI Researchers)**

```
📥 Download:
   - Training Data/train.zip, val.zip
   - Code/4_0_train_punctuation_v1_7punct_ChineseRoBERTa_Lightning.py (v1)
   - Code/4_0_train_punctuation_v2_7punct_SikuRoBERTa_Lightning.py (v2)
   - Models(.ckpt)/ (for fine-tuning)
💻 Purpose: Model retraining, fine-tuning, experiments
```

**4. Original Text Research (Historians/Humanists)**

```
📥 Download: Preprocessed Texts/ folder (selected ZIPs)
💻 Purpose: Data analysis, corpus construction, other research
```

**5. Full Reproduction**

```
📥 Download: All folders
💻 Purpose: Complete reproduction from raw data to deployment
📝 Process:
   1. Extract Preprocessed Texts/ ZIPs
   2. Run 6-stage preprocessing scripts
   3. Generate training data (JSONL)
   4. Train model (Lightning)
   5. Evaluation and validation
```

#### Training Data Details

**train.zip when extracted**: `train.jsonl` (~2.5GB)

* Sample count: ~3.4M
* Total characters: ~420M
* Format: JSONL (one sample per line)

**JSONL Structure Example**:

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
  "source": "Annals of Joseon Dynasty",
  "domain": "Chronicles"
}
```

**labels index**: [,  。 · ? ! 《 》]

* Example: `[1,0,0,0,0,0,0]` = comma (,)
* Example: `[0,1,0,0,0,0,0]` = period (。)

#### Original Text ZIP Information

| ZIP File          | Extracted Size | Main Documents                 |
| ----------------- | -------------- | ------------------------------ |
| Chronicles.zip    | ~2GB           | Annals of Joseon Dynasty, etc. |
| Registers.zip     | ~1.5GB         | Government registers           |
| Diaries.zip       | ~1.2GB         | Mukjae Diary, etc.             |
| Collections.zip   | ~1GB           | Korean Literary Collections    |
| Legal Codes.zip   | ~500MB         | Gyeongguk Daejeon, etc.        |
| Gazetteers.zip    | ~300MB         | Daedong Jiji, etc.             |
| Biographies.zip   | ~200MB         | Gukjo Inmulgo, etc.            |
| Miscellaneous.zip | ~100MB         |                                |

* **Compression format**: UTF-8 encoded TXT files
* **Punctuation**: Original collated punctuation (26 types → converted to 7 types after preprocessing)

### Quick Start

#### Method 1: Windows Executable (Recommended - General Users)

```
1. Download "ChinesePunctuationInference_v2.0.zip" from Google Drive (v2 recommended)
2. Extract ZIP
3. Run ChinesePunctuationInference_v2.exe
4. Select file in GUI → Start processing
```

**Download Link**: [Google Drive](https://drive.google.com/drive/folders/1WGueOa8Oz7kqv4ha7_9pgFRKOzXWId2H?usp=drive_link)

#### Method 2: Python Code Execution (Developers/Researchers)

**Python Code**

```python
from 구두점7_추론모델 import PunctuationPredictor

# Load model
predictor = PunctuationPredictor(
    checkpoint_path="path/to/checkpoint.ckpt"
)

# Prediction
text = "太祖康獻大王姓李諱成桂字君晉古諱旦號松軒"
result = predictor.predict(text)
print(result)
# Output: 太祖康獻大王, 姓李, 諱成桂, 字君晉。古諱旦, 號松軒。
```

**Note**: Python files use Korean names (e.g., `구두점7_추론모델.py`). The English names in examples are for reference only.

**GUI Executable**

```bash
# Build Windows executable (v1)
python 구두점_지정_실행파일_빌드_v1_ChineseRoBERTa.py

# Build Windows executable (v2)
python 구두점_지정_실행파일_빌드_v2_SikuRoBERTa.py

# Run
./dist/한문구두점추론.exe
```

### Training Data

**Sources**

* National Institute of Korean History Database ([https://db.history.go.kr/](https://db.history.go.kr/))
* Korean Classics Comprehensive DB ([https://db.itkc.or.kr/](https://db.itkc.or.kr/))
* Academy of Korean Studies Digital Library ([https://jsg.aks.ac.kr/](https://jsg.aks.ac.kr/))

**Scale**

* Total characters: ~420M
* Training samples: ~3.4M
* Data types: 8 genres (chronicles, collections, diaries, registers, legal codes, gazetteers, biographies, etc.)
* Punctuation types: 7 (, 。 · ? ! 《 》)

**Preprocessing**

* Collection and refinement of collated punctuation texts
* Standardization to 7 punctuation types
* 6-stage preprocessing pipeline

### Model Architecture

**v2.5 (Latest Recommended)**

* **Base Model**: SikuRoBERTa (`SIKU-BERT/sikuroberta`)
* **Task**: Multi-label Classification
* **Labels**: 7 punctuation marks
* **Training**:

  * GPU: L40S 48GB
  * Batch Size: 160 (effective)
  * Learning Rate: 2e-5
  * Epochs: 3
  * Mixed Precision: bf16
  
**v2**

* **Base Model**: SikuRoBERTa (`SIKU-BERT/sikuroberta`)
* Other settings identical

**v1 (Paper Version)**

* **Base Model**: Chinese-RoBERTa (`hfl/chinese-roberta-wwm-ext`)
* Other settings identical

### Directory Structure

```
korean-classical-chinese-punctuation/
├── preprocessing/           # Preprocessing scripts (1_1 ~ 1_6)
├── data_generation/         # Training data generation (2, 3)
├── training/                # Model training and evaluation (4_0, 6)
├── inference/               # Inference and applications (punctuation_7_*)
└── build/                   # Executable build scripts
```

### Citation

If you use this model, please cite:

**APA Style:**

```
Yang, J. (2025). Development and Application of a Deep Learning–Based Model 
for Automated Punctuation Inference in Korean Classical Chinese. 
The Korean Journal of History (Yoksahak Yongu), 100, 267-297. 
https://doi.org/10.37924/JSSW.100.9

Yang, J. (2026). Conventions of Interpretation: Improving a Punctuation 
Prediction Model for Korean Classical Chinese. Korean Cultural Studies, 
111, 7-29. https://doi.org/10.17948/kcs.2026..111.7
```

**BibTeX:**

```bibtex
@article{yang2025punctuation,
  title={Development and Application of a Deep Learning--Based Model for Automated Punctuation Inference in Korean Classical Chinese},
  author={Yang, Junghyun},
  journal={The Korean Journal of History (Yoksahak Yongu)},
  volume={100},
  pages={267--297},
  year={2025},
  publisher={Honam Historical Society},
  doi={10.37924/JSSW.100.9}
}

@article{yang2026interpretation,
  title={Conventions of Interpretation: Improving a Punctuation Prediction Model for Korean Classical Chinese},
  author={Yang, Junghyun},
  journal={Korean Cultural Studies},
  volume={111},
  pages={7--29},
  year={2026},
  publisher={Research Institute of Korean Studies, Korea University},
  doi={10.17948/kcs.2026..111.7}
}
```

**Paper Information:**

* v1.0 (Prior Study)

  * Journal: The Korean Journal of History (Yoksahak Yongu)
  * Volume: 100
  * Publication: November 30, 2025
  * Publisher: Honam Historical Society
  * DOI: [10.37924/JSSW.100.9](https://doi.org/10.37924/JSSW.100.9)

* v2.0–v2.5 (Improvement Study)

  * Journal: Korean Cultural Studies
  * Volume: 111
  * Publication: 2026
  * Publisher: Research Institute of Korean Studies, Korea University
  * DOI: [10.17948/kcs.2026..111.7](https://doi.org/10.17948/kcs.2026..111.7)

### License and Terms of Use

**License**: CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike)

#### ✅ Permitted Uses

**Academic Research**:

* Paper writing and citation
* Academic presentations and education
* Research-purpose modification and improvement
* Non-profit research projects

**Non-commercial Applications**:

* Educational materials and practice in educational institutions
* Digital archive construction by public institutions
* Open-source project integration
* Cultural heritage digitization projects

#### ❌ Restricted Uses

**Commercial Use**:

* Sale of paid services or products
* Corporate profit-oriented use
* Commercial license redistribution
* Use for advertising revenue

**Commercial Use Inquiries**: [yachagye@naver.com](mailto:yachagye@naver.com)

* Commercial licenses can be granted through individual negotiation
* Compliance with Korean Research Foundation project output utilization regulations

#### 📋 Conditions

1. **Attribution**:

   * Must specify original author and source
   * Paper citation required

2. **NonCommercial**:

   * Commercial use prohibited
   * Prior consultation required

3. **ShareAlike**:

   * Derivative works must use same license
   * Continue open-source spirit

**Full License Terms**: [https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

### Future Improvements

Future research directions proposed in the paper:

1. **Two-Track System**

   * Improve performance for paired punctuation (《》)
   * Enhance long-distance dependency modeling

2. **Document Type-Adaptive Modules**

   * Domain-specific fine-tuning
   * Genre-adaptive architecture

3. **Multi-task Integration**

   * Combine with sentence structure analysis
   * Integrate Named Entity Recognition (NER)
   * Multi-task Learning structure

### Limitations

1. **Paired Punctuation**: Title quotation marks (《》) show lower performance (~F1 0.78) compared to other punctuation
2. **Sparse Data**: Exclamation marks (!) have low recall due to insufficient training data
3. **Model Context**: Processes in 512-token units (automatic handling of long texts via sliding window)
4. **Domain Bias**: Training focused on official records may lead to performance degradation on private documents

### Contact

* **Developer**: Junghyun Yang
* **Email**: [yachagye@naver.com](mailto:yachagye@naver.com)
* **GitHub**: [https://github.com/yachagye/korean-classical-chinese-punctuation](https://github.com/yachagye/korean-classical-chinese-punctuation)
* **Issues**: [https://github.com/yachagye/korean-classical-chinese-punctuation/issues](https://github.com/yachagye/korean-classical-chinese-punctuation/issues)
* **Commercial Use Inquiries**: Prior consultation via email

### Disclaimer

The punctuation prediction results of this program may not be perfect. For important academic materials or publications, please use after expert review.
