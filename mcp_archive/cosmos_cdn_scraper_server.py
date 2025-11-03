#!/usr/bin/env python3
"""
인증이 포함된 MCP 서버 - Cosmos CDN Link Scraper용
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cosmos CDN Link Scraper MCP Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 인증 설정
security = HTTPBearer(auto_error=False)

# MCP 모델들
class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

# 전역 데이터 저장소
training_data = []

def verify_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """인증 검증 (선택적)"""
    if credentials:
        # 간단한 토큰 검증 (실제로는 더 복잡한 로직 필요)
        if credentials.credentials == "cosmos-token-2024":
            return True
        else:
            raise HTTPException(status_code=401, detail="Invalid token")
    # 인증이 없어도 허용 (개발용)
    return True

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Cosmos CDN Link Scraper MCP Server",
        "version": "1.0.0",
        "status": "running",
        "protocol": "MCP HTTP with Auth",
        "auth_required": False
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "data_count": len(training_data),
        "auth_status": "optional"
    }

@app.post("/mcp")
async def mcp_endpoint(request: MCPRequest, auth_result = Depends(verify_auth)):
    """MCP 프로토콜 엔드포인트 (인증 포함)"""
    try:
        logger.info(f"MCP 요청: {request.method}")
        
        if request.method == "initialize":
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "cosmos-cdn-scraper",
                        "version": "1.0.0"
                    }
                }
            )
        
        elif request.method == "tools/list":
            return MCPResponse(
                id=request.id,
                result={
                    "tools": [
                        {
                            "name": "scrape_cosmos_images",
                            "description": "Cosmos.so에서 이미지 링크를 스크래핑합니다",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "url": {
                                        "type": "string",
                                        "description": "스크래핑할 Cosmos.so 페이지 URL"
                                    },
                                    "max_images": {
                                        "type": "integer",
                                        "description": "최대 이미지 수",
                                        "default": 50
                                    }
                                },
                                "required": ["url"]
                            }
                        },
                        {
                            "name": "analyze_image",
                            "description": "이미지를 분석하고 카테고리를 추천합니다",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "image_url": {
                                        "type": "string",
                                        "description": "분석할 이미지의 URL"
                                    }
                                },
                                "required": ["image_url"]
                            }
                        }
                    ]
                }
            )
        
        elif request.method == "tools/call":
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            
            if tool_name == "scrape_cosmos_images":
                url = arguments.get("url", "")
                max_images = arguments.get("max_images", 50)
                
                # 간단한 스크래핑 시뮬레이션
                sample_images = [
                    "https://cdn.cosmos.so/646ade0c-beff-4003-bcae-c977de3ea7dd?format=webp&w=1080",
                    "https://cdn.cosmos.so/a22716e5-1442-432c-b320-05b3ad24deec?rect=33%2C0%2C528%2C529&format=webp&w=1080",
                    "https://cdn.cosmos.so/458e7583-47f5-4296-9e8b-b4ea9178f093?rect=97%2C0%2C635%2C635&format=webp&w=1080"
                ]
                
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": f"Cosmos.so 스크래핑 완료. {len(sample_images)}개 이미지 발견. URL: {url}"
                            }
                        ]
                    }
                )
            
            elif tool_name == "analyze_image":
                image_url = arguments.get("image_url", "")
                
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": f"이미지 분석 완료. 추천 카테고리: design, creative, art"
                            }
                        ]
                    }
                )
            
            else:
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                )
        
        else:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Unknown method: {request.method}"
                }
            )
    
    except Exception as e:
        logger.error(f"MCP 요청 처리 오류: {e}")
        return MCPResponse(
            id=request.id,
            error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        )

@app.get("/mcp")
async def mcp_get():
    """MCP GET 엔드포인트"""
    return {
        "message": "Cosmos CDN Link Scraper MCP Server is running",
        "protocol": "MCP HTTP with Auth",
        "version": "1.0.0",
        "auth_status": "optional"
    }

@app.options("/mcp")
async def mcp_options():
    """CORS preflight 요청 처리"""
    return {"message": "OK"}

if __name__ == "__main__":
    print("🚀 Cosmos CDN Link Scraper MCP Server 시작 중...")
    print("📡 서버 주소: http://localhost:5001")
    print("📚 MCP 엔드포인트: http://localhost:5001/mcp")
    print("🔐 인증: 선택적 (토큰: cosmos-token-2024)")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5001,
        log_level="info"
    )
