# MCP (Model Context Protocol) 아카이브

이 폴더는 MCP 관련 파일들을 보관하는 아카이브입니다. 현재 프로젝트에서는 MCP 대신 Gemini API를 직접 사용하고 있지만, 향후 MCP 기능이 필요할 때를 대비해 보관합니다.

## 📁 파일 구조

### 🖥️ GUI 애플리케이션
- `cosmos_gui_v3_mcp.py` - MCP 연동 GUI 애플리케이션 (v3)

### 🚀 MCP 서버들
- `cosmos_mcp_real_server.py` - 실제 Cosmos.so 스크래핑 MCP 서버
- `cosmos_mcp_server.py` - 기본 MCP 서버
- `cosmos_cdn_scraper_server.py` - CDN 스크래퍼 MCP 서버
- `http_mcp_server.py` - HTTP MCP 서버
- `https_mcp_server.py` - HTTPS MCP 서버
- `mcp_http_server.py` - MCP HTTP 서버
- `simple_mcp_server.py` - 간단한 MCP 서버

### 🔧 유틸리티
- `run_mcp_server.py` - MCP 서버 실행 스크립트
- `test_mcp_server.py` - MCP 서버 테스트 스크립트
- `generate_ssl_cert.py` - SSL 인증서 생성 도구

### 📋 설정 파일
- `claude_desktop_config.json` - Claude Desktop MCP 설정

### 📚 문서
- `CLAUDE_DESKTOP_MCP_GUIDE.md` - Claude Desktop MCP 가이드
- `MCP_INTEGRATION_GUIDE.md` - MCP 통합 가이드
- `CLOUDFLARE_MCP_SETUP.md` - Cloudflare MCP 설정 가이드

## 🔄 현재 상태

현재 프로젝트는 MCP 대신 **Gemini API 직접 연동**을 사용합니다:
- `cosmos_gui_v4_gemini.py` - Gemini API 직접 연동 GUI (현재 사용 중)

## 🚀 MCP 복원 방법

필요시 다음 단계로 MCP 기능을 복원할 수 있습니다:

1. **서버 실행**:
   ```bash
   python3 mcp_archive/cosmos_mcp_real_server.py
   ```

2. **GUI 실행**:
   ```bash
   python3 mcp_archive/cosmos_gui_v3_mcp.py
   ```

3. **설정 복원**:
   - `claude_desktop_config.json`을 프로젝트 루트로 복사
   - Claude Desktop에서 MCP 서버 연결 설정

## 📝 참고사항

- 모든 MCP 관련 기능은 현재 비활성화 상태
- Gemini API가 더 안정적이고 빠른 성능을 제공
- 필요시 언제든지 MCP 기능으로 되돌릴 수 있음
