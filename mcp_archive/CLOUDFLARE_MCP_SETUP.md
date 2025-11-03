# Cloudflare MCP 서버 설정 가이드

## 문제 상황
`mcp.cdnscraper.dev` 도메인에서 SSL handshake 실패 (Error 525)가 발생하고 있습니다. 이는 Cloudflare와 원본 서버 간의 SSL 설정이 호환되지 않기 때문입니다.

## 해결 방법

### 1. 로컬 테스트 환경 구성

#### SSL 인증서 생성
```bash
# SSL 인증서 생성 스크립트 실행
python3 generate_ssl_cert.py

# 또는 수동으로 생성
openssl genrsa -out key.pem 2048
openssl req -new -x509 -key key.pem -out cert.pem -days 365 \
  -subj "/C=KR/ST=Seoul/L=Seoul/O=Cosmos/OU=IT/CN=mcp.cdnscraper.dev"
```

#### HTTPS MCP 서버 실행
```bash
# HTTPS MCP 서버 시작
python3 https_mcp_server.py
```

서버가 `https://localhost:3000`에서 실행됩니다.

### 2. Cloudflare SSL 설정

#### SSL/TLS 모드 변경
1. Cloudflare 대시보드에 로그인
2. `mcp.cdnscraper.dev` 도메인 선택
3. **SSL/TLS** 탭으로 이동
4. **SSL/TLS 암호화 모드**를 다음 중 하나로 변경:
   - **Full (strict)**: 원본 서버에 유효한 SSL 인증서 필요
   - **Full**: 원본 서버에 SSL 인증서 필요 (자체 서명도 허용)
   - **Flexible**: 원본 서버는 HTTP 허용

#### 권장 설정
- **개발 환경**: Full (자체 서명 인증서 허용)
- **운영 환경**: Full (strict) + Let's Encrypt 인증서

### 3. 원본 서버 설정

#### HTTPS 서버 구성
```python
# https_mcp_server.py에서 SSL 컨텍스트 설정
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain("cert.pem", "key.pem")
```

#### 포트 설정
- **HTTP**: 80번 포트
- **HTTPS**: 443번 포트 (Cloudflare가 프록시)

### 4. Claude Desktop MCP 설정

#### claude_desktop_config.json 업데이트
```json
{
  "mcpServers": {
    "cosmos-image-classifier": {
      "command": "python3",
      "args": ["https_mcp_server.py"],
      "env": {
        "MCP_SERVER_URL": "https://mcp.cdnscraper.dev"
      }
    }
  }
}
```

### 5. 테스트 방법

#### 로컬 테스트
```bash
# 서버 상태 확인
curl -k https://localhost:3000/health

# MCP 엔드포인트 테스트
curl -k -X POST https://localhost:3000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{}}'
```

#### Cloudflare 테스트
```bash
# Cloudflare를 통한 테스트
curl https://mcp.cdnscraper.dev/health
curl -X POST https://mcp.cdnscraper.dev/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{}}'
```

### 6. 문제 해결

#### Error 525 해결
1. **SSL/TLS 모드 확인**: Full 또는 Full (strict) 사용
2. **원본 서버 SSL 확인**: 유효한 인증서 필요
3. **포트 확인**: 443번 포트에서 HTTPS 서비스 실행
4. **방화벽 확인**: Cloudflare가 원본 서버에 접근 가능

#### 인증서 문제
```bash
# 인증서 유효성 확인
openssl x509 -in cert.pem -text -noout

# 인증서 체인 확인
openssl verify cert.pem
```

### 7. 운영 환경 설정

#### Let's Encrypt 인증서 사용
```bash
# certbot 설치 (Ubuntu/Debian)
sudo apt-get install certbot

# 인증서 발급
sudo certbot certonly --standalone -d mcp.cdnscraper.dev

# 인증서 파일 위치
# /etc/letsencrypt/live/mcp.cdnscraper.dev/fullchain.pem
# /etc/letsencrypt/live/mcp.cdnscraper.dev/privkey.pem
```

#### 자동 갱신 설정
```bash
# crontab에 자동 갱신 추가
sudo crontab -e

# 다음 라인 추가
0 12 * * * /usr/bin/certbot renew --quiet
```

### 8. 모니터링

#### Cloudflare Analytics
- SSL/TLS 탭에서 인증서 상태 모니터링
- 오류 로그에서 525 에러 확인

#### 서버 로그
```bash
# HTTPS MCP 서버 로그 확인
tail -f https_mcp_server.log
```

## 요약

1. **로컬에서 HTTPS MCP 서버 실행**
2. **Cloudflare SSL/TLS 모드를 Full로 설정**
3. **원본 서버에서 유효한 SSL 인증서 사용**
4. **Claude Desktop MCP 설정 업데이트**
5. **테스트 및 모니터링**

이 설정으로 Cloudflare를 통한 Claude AI와 MCP 서버 연결이 정상적으로 작동할 것입니다.
