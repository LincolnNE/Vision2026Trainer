# Cosmos CDN 자동 스크래핑 이미지 분류 모델 학습 파이프라인

이 프로젝트는 cosmos.so CDN에서 이미지를 자동으로 스크래핑하고 이미지 분류 모델을 학습하는 완전 자동화된 파이프라인을 제공합니다.

## 주요 기능

- **자동 웹 스크래핑**: cosmos.so CDN에서 이미지 URL 자동 수집
- **지능형 카테고리 분류**: URL 경로 기반 자동 라벨링
- **이미지 다운로드 및 전처리**: 224x224 리사이즈, RGB 정규화
- **CNN 모델 학습**: PyTorch 기반 이미지 분류 모델
- **완전 자동화**: 스크래핑부터 모델 학습까지 원클릭 실행

## 데이터 형식

### 자동 스크래핑 결과:
- **x_train**: `https://cdn.cosmos.so/book/layout/book1.webp` (스크래핑된 이미지 URL)
- **category**: `"book_layout"` (URL 경로에서 자동 추출된 카테고리)
- **y_train**: `"book_layout"` (자동 생성된 라벨)

### 지원하는 자동 카테고리:
- `book_layout`: 책 레이아웃
- `photography`: 사진
- `design`: 디자인
- `artwork`: 예술 작품
- `minimal_design`: 미니멀 디자인
- `abstract_art`: 추상 예술
- `texture`: 텍스처
- `pattern`: 패턴
- `layout`: 레이아웃
- `creative`: 크리에이티브
- `black_white`: 흑백
- `monochrome`: 모노크롬
- `colorful`: 컬러풀
- `vintage`: 빈티지
- `modern`: 모던
- `classic`: 클래식

## 프로젝트 구조

```
Vision2025Trainer/
├── dataset/
│   ├── x_train.csv      # 스크래핑된 이미지 URL과 카테고리
│   └── y_train.csv      # 자동 생성된 라벨
├── models/
│   └── model.pt         # 학습된 PyTorch 모델
├── results/
│   └── metrics.png      # 학습 결과 시각화 그래프
├── auto_scraping_pipeline.py    # 자동 스크래핑 메인 파이프라인
├── image_classification_pipeline.py  # 기존 수동 파이프라인
├── test_auto_scraping.py        # 자동 스크래핑 테스트 스크립트
├── test_pipeline.py             # 기존 파이프라인 테스트 스크립트
├── requirements.txt              # 필요한 패키지 목록
└── README.md                    # 이 파일
```

## 설치 및 실행

### 1. 환경 설정

```bash
# Python 3.10 이상 필요
python3 --version

# 가상환경 생성 (선택사항)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate     # Windows

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 자동 스크래핑 파이프라인 실행

```bash
# 완전 자동화된 파이프라인 실행
python3 auto_scraping_pipeline.py
```

### 3. 테스트 실행

```bash
# 자동 스크래핑 파이프라인 테스트
python3 test_auto_scraping.py

# 기존 수동 파이프라인 테스트
python3 test_pipeline.py
```

## 사용법

### 자동 스크래핑 파이프라인

파이프라인은 다음과 같은 단계로 자동 실행됩니다:

1. **웹 스크래핑**: cosmos.so CDN에서 이미지 URL 자동 수집
2. **카테고리 분류**: URL 경로 기반 자동 라벨링
3. **CSV 생성**: x_train.csv, y_train.csv 파일 생성
4. **데이터 전처리**: 이미지 다운로드, 리사이즈, 정규화
5. **모델 학습**: CNN 모델 학습 (10 에포크)
6. **결과 저장**: 모델 파일 및 시각화 그래프 저장

### 커스터마이징

#### 스크래핑 설정 변경

```python
# auto_scraping_pipeline.py의 main() 함수에서
pipeline = AutoScrapingPipeline()

# 스크래핑 페이지 수 조정
image_urls, labels = pipeline.scrape_and_categorize(max_pages=10)  # 기본값: 5
```

#### 카테고리 키워드 추가

```python
# auto_scraping_pipeline.py의 CosmosScraper 클래스에서
category_keywords = {
    'book': 'book_layout',
    'art': 'artwork',
    'photo': 'photography',
    # 새로운 키워드 추가
    'nature': 'nature_photography',
    'urban': 'urban_design',
    # ...
}
```

#### 스크래핑 대상 URL 변경

```python
# auto_scraping_pipeline.py의 CosmosScraper 클래스에서
def __init__(self, base_url: str = "https://your-custom-cdn.com", timeout: int = 10):
    self.base_url = base_url
    # ...
```

### 로컬 파일 사용

실제 이미지 파일이 로컬에 있는 경우, `CosmosImageDataset`의 `base_url` 매개변수를 사용할 수 있습니다:

```python
dataset = CosmosImageDataset(
    image_urls, 
    labels, 
    transform=transform,
    base_url="/path/to/your/images"  # 로컬 이미지 디렉토리
)
```

## 스크래핑 기능 상세

### 지원하는 이미지 형식
- `.webp`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`

### 스크래핑 전략
1. **HTML 파싱**: BeautifulSoup을 사용한 구조화된 데이터 추출
2. **링크 추적**: 페이지 간 링크를 따라가며 이미지 탐색
3. **중복 제거**: 동일한 이미지 URL 자동 제거
4. **오류 처리**: 네트워크 오류 및 파싱 오류에 대한 견고한 처리

### 카테고리 추출 알고리즘
1. **URL 경로 분석**: `/book/layout/`, `/photo/minimal/` 등에서 키워드 추출
2. **파일명 분석**: `book_spread_001.webp` 등에서 카테고리 추출
3. **키워드 매칭**: 미리 정의된 키워드 사전과 매칭
4. **기본 카테고리**: 매칭되지 않는 경우 'general' 카테고리 할당

## 모델 구조

사용되는 CNN 모델 구조:

```
Conv2D(3→32) → ReLU → MaxPool2D
Conv2D(32→64) → ReLU → MaxPool2D  
Conv2D(64→128) → ReLU → MaxPool2D
Flatten → Dense(128×28×28→512) → ReLU → Dropout(0.5)
Dense(512→num_classes) → Softmax
```

## 출력 파일

실행 완료 후 다음 파일들이 생성됩니다:

- `./dataset/x_train.csv`: 스크래핑된 이미지 URL과 자동 생성된 카테고리
- `./dataset/y_train.csv`: 자동 생성된 라벨 정보  
- `./models/model.pt`: 학습된 PyTorch 모델
- `./results/metrics.png`: 학습 손실 및 정확도 그래프

## 요구사항

- Python 3.10 이상
- PyTorch 2.0.0 이상
- BeautifulSoup4 4.11.0 이상
- 인터넷 연결 (스크래핑 및 이미지 다운로드용)
- 충분한 디스크 공간 (이미지 캐싱용)

## 문제 해결

### 일반적인 오류

1. **스크래핑 실패**: 
   - 네트워크 연결 확인
   - 대상 사이트의 robots.txt 확인
   - User-Agent 헤더 업데이트

2. **이미지 다운로드 실패**: 
   - URL 유효성 검사
   - 타임아웃 설정 조정

3. **메모리 부족**: 
   - 배치 크기 줄이기
   - 이미지 해상도 조정

4. **카테고리 분류 부정확**: 
   - 키워드 사전 업데이트
   - URL 패턴 분석

### 로그 확인

프로그램 실행 시 상세한 로그가 출력되므로, 문제 발생 시 로그를 확인하세요.

### 성능 최적화

- `max_pages` 파라미터로 스크래핑 범위 조정
- `timeout` 파라미터로 네트워크 타임아웃 조정
- `batch_size` 파라미터로 메모리 사용량 조정

## 라이선스

이 프로젝트는 Apache License 2.0 하에 라이선스가 부여됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## 주의사항

- 웹 스크래핑 시 대상 사이트의 이용약관을 준수하세요
- 과도한 요청으로 서버에 부하를 주지 않도록 주의하세요
- 저작권이 있는 이미지의 사용에 주의하세요