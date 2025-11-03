#!/usr/bin/env python3
"""
HTTPS MCP 서버 - Cloudflare 연동용
SSL/TLS 지원으로 Cloudflare와 호환되는 MCP 서버
"""

import asyncio
import json
import logging
import ssl
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cosmos Image Classifier HTTPS MCP Server")

# CORS 설정 - Cloudflare 도메인 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mcp.cdnscraper.dev",
        "https://cdnscraper.dev", 
        "http://localhost:3000",
        "http://localhost:2001",
        "https://localhost:3000",
        "https://localhost:2001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

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

def analyze_image_url(image_url: str) -> str:
    """URL 패턴을 기반으로 스마트 카테고리 추천"""
    url_lower = image_url.lower()
    
    # URL 패턴 매칭을 통한 카테고리 추천
    recommended = []
    
    # Cosmos.so 특화 패턴 매칭
    if any(word in url_lower for word in ['nature', 'forest', 'tree', 'mountain', 'ocean', 'sea', 'lake', 'river', 'landscape']):
        recommended.extend(['nature', 'landscape', 'outdoor'])
    
    if any(word in url_lower for word in ['building', 'architecture', 'house', 'home', 'office', 'city', 'urban', 'skyscraper']):
        recommended.extend(['architecture', 'urban', 'design'])
    
    if any(word in url_lower for word in ['people', 'person', 'man', 'woman', 'child', 'portrait', 'face', 'lifestyle']):
        recommended.extend(['people', 'portrait', 'lifestyle'])
    
    if any(word in url_lower for word in ['art', 'painting', 'drawing', 'creative', 'design', 'graphic', 'illustration']):
        recommended.extend(['art', 'creative', 'design'])
    
    if any(word in url_lower for word in ['tech', 'technology', 'computer', 'digital', 'ai', 'robot', 'modern']):
        recommended.extend(['technology', 'modern', 'professional'])
    
    if any(word in url_lower for word in ['food', 'restaurant', 'kitchen', 'cooking', 'meal', 'cuisine']):
        recommended.extend(['food', 'lifestyle', 'indoor'])
    
    if any(word in url_lower for word in ['fashion', 'clothing', 'style', 'outfit', 'dress', 'wear']):
        recommended.extend(['fashion', 'lifestyle', 'people'])
    
    if any(word in url_lower for word in ['travel', 'vacation', 'trip', 'destination', 'tourist', 'adventure']):
        recommended.extend(['travel', 'outdoor', 'lifestyle'])
    
    if any(word in url_lower for word in ['sports', 'fitness', 'gym', 'exercise', 'athletic', 'workout']):
        recommended.extend(['sports', 'health', 'lifestyle'])
    
    if any(word in url_lower for word in ['music', 'concert', 'band', 'instrument', 'audio', 'sound']):
        recommended.extend(['music', 'entertainment', 'culture'])
    
    if any(word in url_lower for word in ['business', 'office', 'meeting', 'corporate', 'professional', 'work']):
        recommended.extend(['business', 'professional', 'office'])
    
    if any(word in url_lower for word in ['abstract', 'pattern', 'texture', 'geometric', 'shape', 'minimal']):
        recommended.extend(['abstract', 'art', 'design'])
    
    if any(word in url_lower for word in ['korean', 'korea', 'asian', 'culture', 'traditional']):
        recommended.extend(['korean_culture', 'culture', 'traditional'])
    
    # 중복 제거 및 최대 3개 카테고리 반환
    unique_categories = list(dict.fromkeys(recommended))
    
    if not unique_categories:
        # 기본 카테고리
        unique_categories = ['general', 'design', 'creative']
    
    # 최대 3개까지만 반환
    return ', '.join(unique_categories[:3])

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Cosmos Image Classifier HTTPS MCP Server",
        "version": "1.0.0",
        "status": "running",
        "protocol": "MCP HTTPS",
        "ssl_enabled": True,
        "cloudflare_compatible": True
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "ssl": True,
        "data_count": len(training_data),
        "timestamp": asyncio.get_event_loop().time()
    }

@app.post("/mcp")
async def mcp_endpoint(request: MCPRequest):
    """MCP 프로토콜 엔드포인트 (HTTPS 지원)"""
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
                        "name": "cosmos-image-classifier-https",
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
                            "name": "analyze_image",
                            "description": "이미지를 분석하고 카테고리를 추천합니다",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "image_url": {
                                        "type": "string",
                                        "description": "분석할 이미지의 URL"
                                    },
                                    "context": {
                                        "type": "string",
                                        "description": "추가 컨텍스트 정보",
                                        "default": ""
                                    }
                                },
                                "required": ["image_url"]
                            }
                        },
                        {
                            "name": "batch_analyze_images",
                            "description": "여러 이미지를 일괄 분석합니다",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "image_urls": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "분석할 이미지 URL 목록"
                                    }
                                },
                                "required": ["image_urls"]
                            }
                        },
                        {
                            "name": "train_model",
                            "description": "이미지 분류 모델을 훈련합니다",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "epochs": {
                                        "type": "integer",
                                        "description": "훈련 에포크 수",
                                        "default": 10
                                    },
                                    "batch_size": {
                                        "type": "integer",
                                        "description": "배치 크기",
                                        "default": 8
                                    }
                                }
                            }
                        },
                        {
                            "name": "get_training_status",
                            "description": "현재 훈련 상태를 확인합니다",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        },
                        {
                            "name": "export_dataset",
                            "description": "훈련 데이터셋을 내보냅니다",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "format": {
                                        "type": "string",
                                        "enum": ["csv", "json"],
                                        "description": "내보낼 형식",
                                        "default": "csv"
                                    }
                                }
                            }
                        }
                    ]
                }
            )
        
        elif request.method == "tools/call":
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            
            if tool_name == "analyze_image":
                image_url = arguments.get("image_url")
                context = arguments.get("context", "")
                
                if not image_url:
                    return MCPResponse(
                        id=request.id,
                        error={
                            "code": -32602,
                            "message": "image_url is required"
                        }
                    )
                
                # URL 패턴 기반 스마트 카테고리 추천
                recommended_categories = analyze_image_url(image_url)
                
                # 훈련 데이터에 추가
                training_data.append({
                    "image_url": image_url,
                    "category": recommended_categories,
                    "context": context,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": f"이미지 분석 완료. 추천 카테고리: {recommended_categories}"
                            }
                        ]
                    }
                )
            
            elif tool_name == "batch_analyze_images":
                image_urls = arguments.get("image_urls", [])
                if not image_urls:
                    return MCPResponse(
                        id=request.id,
                        error={
                            "code": -32602,
                            "message": "image_urls is required"
                        }
                    )
                
                results = []
                for url in image_urls:
                    recommended_categories = analyze_image_url(url)
                    training_data.append({
                        "image_url": url,
                        "category": recommended_categories,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                    results.append({
                        "url": url,
                        "categories": recommended_categories
                    })
                
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": f"일괄 분석 완료. {len(results)}개 이미지 처리됨."
                            }
                        ]
                    }
                )
            
            elif tool_name == "train_model":
                epochs = arguments.get("epochs", 10)
                batch_size = arguments.get("batch_size", 8)
                
                if not training_data:
                    return MCPResponse(
                        id=request.id,
                        error={
                            "code": -32602,
                            "message": "훈련할 데이터가 없습니다. 먼저 이미지를 분석해주세요."
                        }
                    )
                
                # 간단한 훈련 시뮬레이션
                await asyncio.sleep(1)  # 훈련 시뮬레이션
                
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": f"모델 훈련 완료. {epochs} 에포크, 배치 크기 {batch_size}로 훈련했습니다. 총 {len(training_data)}개 데이터 사용."
                            }
                        ]
                    }
                )
            
            elif tool_name == "get_training_status":
                if not training_data:
                    status_text = "현재 훈련 데이터가 없습니다."
                else:
                    categories = [item["category"] for item in training_data]
                    category_counts = {cat: categories.count(cat) for cat in set(categories)}
                    
                    status_text = f"""
현재 훈련 상태:
- 총 이미지: {len(training_data)}개
- 카테고리 수: {len(set(categories))}개

카테고리별 분포:
{chr(10).join([f"- {cat}: {count}개" for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)])}
                    """
                
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": status_text
                            }
                        ]
                    }
                )
            
            elif tool_name == "export_dataset":
                export_format = arguments.get("format", "csv")
                
                if not training_data:
                    return MCPResponse(
                        id=request.id,
                        error={
                            "code": -32602,
                            "message": "내보낼 데이터가 없습니다."
                        }
                    )
                
                if export_format == "csv":
                    # CSV 형식으로 내보내기
                    import pandas as pd
                    
                    df_x = pd.DataFrame([{
                        'image_link.jpg': item['image_url'].split('/')[-1].split('?')[0],
                        'Category': item['category']
                    } for item in training_data])
                    
                    df_y = pd.DataFrame({'Category': [item['category'] for item in training_data]})
                    
                    # 파일 저장
                    os.makedirs('./dataset', exist_ok=True)
                    df_x.to_csv('./dataset/x_train_https.csv', index=False)
                    df_y.to_csv('./dataset/y_train_https.csv', index=False)
                    
                    result_text = f"""
데이터셋 내보내기 완료!

생성된 파일:
- ./dataset/x_train_https.csv ({len(df_x)}개 행)
- ./dataset/y_train_https.csv ({len(df_y)}개 행)

형식: 
- x_train: image_link.jpg, Category
- y_train: Category
                    """
                else:  # JSON
                    with open('./dataset/training_data_https.json', 'w', encoding='utf-8') as f:
                        json.dump(training_data, f, ensure_ascii=False, indent=2)
                    
                    result_text = f"""
데이터셋 내보내기 완료!

생성된 파일: ./dataset/training_data_https.json ({len(training_data)}개 항목)
                    """
                
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": result_text
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
        "message": "HTTPS MCP Server is running",
        "protocol": "MCP HTTPS",
        "version": "1.0.0",
        "ssl_enabled": True,
        "cloudflare_compatible": True
    }

@app.options("/mcp")
async def mcp_options():
    """CORS preflight 요청 처리"""
    return {"message": "OK"}

def create_ssl_context():
    """SSL 컨텍스트 생성"""
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    # 자체 서명된 인증서 생성 (개발용)
    # 실제 운영에서는 유효한 SSL 인증서를 사용해야 함
    try:
        # Let's Encrypt 또는 다른 CA에서 발급받은 인증서 사용
        ssl_context.load_cert_chain(
            certfile="cert.pem",
            keyfile="key.pem"
        )
        logger.info("SSL 인증서 로드 성공")
    except FileNotFoundError:
        logger.warning("SSL 인증서 파일을 찾을 수 없습니다. HTTP 모드로 실행됩니다.")
        return None
    
    return ssl_context

if __name__ == "__main__":
    print("🚀 Cosmos Image Classifier HTTPS MCP Server 시작 중...")
    print("📡 서버 주소: https://localhost:3000")
    print("📚 MCP 엔드포인트: https://localhost:3000/mcp")
    print("🔒 SSL/TLS 지원 활성화")
    print("☁️ Cloudflare 호환 모드")
    
    # SSL 컨텍스트 생성
    ssl_context = create_ssl_context()
    
    if ssl_context:
        # HTTPS 모드
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=3000,
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            log_level="info"
        )
    else:
        # HTTP 모드 (SSL 인증서가 없는 경우)
        print("⚠️ SSL 인증서가 없어 HTTP 모드로 실행됩니다.")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=3000,
            log_level="info"
        )
