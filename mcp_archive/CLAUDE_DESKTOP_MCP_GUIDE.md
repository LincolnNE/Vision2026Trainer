# 🚀 Claude Desktop 커스텀 MCP 연결 가이드

## 🎯 **Claude Desktop에서 우리 MCP 서버 연결하기**

### 📋 **단계별 설정 방법**

#### 1. **"Add custom connector" 다이얼로그 열기**
- Claude Desktop에서 설정 → Connectors → "Add custom connector" 클릭

#### 2. **기본 정보 입력**
- **Name**: `cosmos-image-classifier`
- **Remote MCP server URL**: 비워둠 (로컬 서버 사용)

#### 3. **Advanced Settings 열기**
- "Advanced settings" 섹션을 클릭하여 확장

#### 4. **로컬 서버 설정**
다음 중 하나의 방법을 선택:

### 🔧 **방법 1: 직접 실행 (권장)**

#### A. MCP 서버 실행
```bash
cd /Users/robinhood/Vision2025Trainer
python3 cosmos_mcp_server.py
```

#### B. Claude Desktop 설정
Advanced Settings에서:
- **Command**: `python3`
- **Args**: `["cosmos_mcp_server.py"]`
- **Working Directory**: `/Users/robinhood/Vision2025Trainer`
- **Environment Variables**:
  ```json
  {
    "PYTHONPATH": "/Users/robinhood/Vision2025Trainer"
  }
  ```

### 🔧 **방법 2: 설정 파일 사용**

#### A. 설정 파일 생성
`~/Library/Application Support/Claude/claude_desktop_config.json` 파일 생성:

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

#### B. Claude Desktop 재시작
- Claude Desktop을 완전히 종료하고 다시 시작

### 🎯 **연결 확인**

연결이 성공하면 Claude에서 다음과 같이 사용할 수 있습니다:

```
@cosmos-image-classifier analyze_image "https://cdn.cosmos.so/646ade0c-beff-4003-bcae-c977de3ea7dd?format=webp&w=1080"
```

### 🛠️ **사용 가능한 도구들**

1. **analyze_image**
   - 이미지 분석 및 카테고리 추천
   - 사용법: `analyze_image "이미지URL"`

2. **batch_analyze_images**
   - 여러 이미지 일괄 분석
   - 사용법: `batch_analyze_images ["URL1", "URL2", "URL3"]`

3. **train_model**
   - 수집된 데이터로 모델 훈련
   - 사용법: `train_model {"epochs": 5, "batch_size": 8}`

4. **get_training_status**
   - 현재 훈련 상태 확인
   - 사용법: `get_training_status`

5. **export_dataset**
   - 데이터셋을 CSV로 내보내기
   - 사용법: `export_dataset {"format": "csv"}`

### 🔍 **실제 사용 예시**

#### 이미지 분석
```
Claude, 이 이미지를 분석해줘:
@cosmos-image-classifier analyze_image "https://cdn.cosmos.so/f85e4901-04d7-4a73-8e47-ac812eef354e?format=webp&w=1080"
```

#### 일괄 분석
```
여러 이미지를 한번에 분석해줘:
@cosmos-image-classifier batch_analyze_images [
  "https://cdn.cosmos.so/646ade0c-beff-4003-bcae-c977de3ea7dd?format=webp&w=1080",
  "https://cdn.cosmos.so/a22716e5-1442-432c-b320-05b3ad24deec?rect=33%2C0%2C528%2C529&format=webp&w=1080"
]
```

#### 모델 훈련
```
수집된 데이터로 모델을 훈련해줘:
@cosmos-image-classifier train_model {"epochs": 10, "batch_size": 16}
```

### 🚨 **문제 해결**

#### 연결 실패
1. MCP 서버가 실행 중인지 확인
2. Python 경로가 올바른지 확인
3. 필요한 패키지가 설치되어 있는지 확인

#### 도구 사용 불가
1. Claude Desktop 재시작
2. MCP 서버 재시작
3. 설정 파일 문법 확인

### 🎉 **완성!**

이제 Claude Desktop에서 직접 우리의 이미지 분류 시스템을 사용할 수 있습니다!

- ✅ **실시간 이미지 분석**
- ✅ **AI 카테고리 추천**
- ✅ **모델 훈련**
- ✅ **데이터셋 관리**

모든 것이 Claude 채팅 인터페이스에서 가능합니다! 🚀
