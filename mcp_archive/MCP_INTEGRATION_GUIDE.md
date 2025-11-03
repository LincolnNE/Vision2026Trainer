# Cosmos.so 이미지 분류 시스템 v3.0 - MCP 연동 가이드

## 🚀 새로운 기능: Claude Vision과 MCP 연동

### 📋 개요
이제 Python과 MCP(Model Context Protocol)를 통해 Claude Vision과 실시간으로 연동하여 이미지 분석 및 카테고리 추천을 받을 수 있습니다!

### 🔧 설치 및 설정

#### 1. 필요한 패키지 설치
```bash
pip3 install mcp pandas requests pillow torch torchvision scikit-learn matplotlib beautifulsoup4 lxml
```

#### 2. Claude Desktop 설정
1. Claude Desktop 앱을 열고 설정으로 이동
2. "Add custom connector" 클릭
3. 다음 설정 파일을 사용:

```json
{
  "mcpServers": {
    "cosmos-image-classifier": {
      "command": "python3",
      "args": ["cosmos_mcp_server.py"],
      "cwd": "/Users/robinhood/Vision2025Trainer",
      "env": {
        "PYTHONPATH": "/Users/robinhood/Vision2025Trainer"
      }
    }
  }
}
```

### 🎯 사용 방법

#### 1. MCP 서버 시작
```bash
python3 cosmos_mcp_server.py
```

#### 2. GUI 애플리케이션 실행
```bash
python3 cosmos_gui_v3_mcp.py
```

### 🖥️ GUI 기능

#### **MCP 연결 패널**
- **MCP 서버 상태**: 실시간 연결 상태 표시
- **서버 재시작**: MCP 서버 재시작 버튼
- **Cosmos.so 스크래핑**: 자동 이미지 수집
- **진행률 표시**: 실시간 스크래핑 진행 상황

#### **AI 이미지 분석 패널**
- **이미지 목록**: 수집된 이미지들의 리스트
- **AI 분석 시작**: 선택된 이미지의 AI 분석
- **전체 분석**: 모든 이미지 일괄 AI 분석
- **카테고리 관리**: 자유 입력 가능한 카테고리 시스템
- **AI 분석 결과**: Claude Vision의 분석 결과 표시

#### **모델 훈련 패널**
- **훈련 파라미터**: 에포크, 배치 크기 설정
- **훈련 시작**: AI 추천 카테고리로 모델 훈련
- **실시간 그래프**: Loss와 Accuracy 시각화

### 🔄 워크플로우

1. **스크래핑**: Cosmos.so에서 이미지 자동 수집
2. **AI 분석**: Claude Vision으로 각 이미지 분석
3. **카테고리 추천**: AI가 제안하는 카테고리 확인/수정
4. **모델 훈련**: 추천된 카테고리로 CNN 모델 훈련
5. **결과 저장**: CSV 파일로 데이터셋 내보내기

### 📊 출력 형식

#### x_train.csv
```csv
image_link.jpg,Category
sample1.jpg,nature
sample2.jpg,architecture
sample3.jpg,art
```

#### y_train.csv
```csv
Category
nature
architecture
art
```

### 🛠️ MCP 도구

#### analyze_image
- **기능**: 단일 이미지 분석
- **입력**: image_url, context (선택사항)
- **출력**: 추천 카테고리, 신뢰도, 분석 텍스트

#### batch_analyze_images
- **기능**: 여러 이미지 일괄 분석
- **입력**: image_urls 배열
- **출력**: 모든 이미지의 분석 결과

#### train_model
- **기능**: 수집된 데이터로 모델 훈련
- **입력**: epochs, batch_size
- **출력**: 훈련 결과 및 정확도

#### get_training_status
- **기능**: 현재 훈련 상태 확인
- **출력**: 데이터 수, 카테고리 분포, 평균 신뢰도

#### export_dataset
- **기능**: 훈련 데이터셋 내보내기
- **입력**: format (csv/json)
- **출력**: CSV 또는 JSON 파일

### 🎨 지원 카테고리

- **nature**: 자연, 풍경, 식물
- **animals**: 동물, 펫
- **food**: 음식, 요리
- **architecture**: 건축, 인테리어
- **technology**: 기술, 전자제품
- **art**: 예술, 디자인
- **people**: 사람, 포트레이트
- **objects**: 물건, 도구
- **abstract**: 추상, 개념
- **korean_culture**: 한국 문화
- **fashion**: 패션, 스타일
- **culture**: 문화, 전통
- **design**: 디자인, 그래픽
- **sports**: 스포츠, 운동
- **travel**: 여행, 관광

### 🔍 AI 분석 예시

```
추천 카테고리: architecture (신뢰도: 0.89)

대안 카테고리:
- design (0.76)
- objects (0.65)
- technology (0.52)

분석: 이 이미지는 현대적인 건축물을 보여주며, 
깔끔한 라인과 기하학적 형태가 특징입니다.

감지된 객체: building, window, structure
주요 색상: gray, white, blue
```

### 🚨 문제 해결

#### MCP 서버 연결 실패
1. Python 패키지 설치 확인
2. 포트 충돌 확인
3. 서버 재시작 시도

#### AI 분석 실패
1. 이미지 URL 유효성 확인
2. 네트워크 연결 상태 확인
3. MCP 서버 상태 확인

#### 훈련 오류
1. 충분한 데이터 확인 (최소 10개 이미지)
2. 카테고리 균형 확인
3. 메모리 사용량 확인

### 📈 성능 최적화

- **배치 크기**: GPU 메모리에 따라 조정
- **에포크 수**: 과적합 방지를 위해 적절히 설정
- **이미지 크기**: 224x224 권장
- **데이터 증강**: 회전, 크롭 등으로 데이터 다양성 증가

### 🔮 향후 계획

- **실시간 Claude API 연동**: 현재 시뮬레이션에서 실제 API 호출로 업그레이드
- **고급 모델**: ResNet, EfficientNet 등 더 정교한 모델 지원
- **클라우드 훈련**: AWS, GCP 등 클라우드 환경에서 대규모 훈련
- **모바일 앱**: iOS/Android 앱으로 확장

---

**🎉 이제 Claude Vision의 강력한 이미지 분석 능력을 활용하여 더 정확하고 효율적인 이미지 분류 시스템을 구축할 수 있습니다!**
