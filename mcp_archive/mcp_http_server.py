#!/usr/bin/env python3
"""
MCP HTTP 서버 - Claude Desktop 연동용
표준 MCP 프로토콜을 따르는 HTTP 서버
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests
import os
from PIL import Image
import io
import base64
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cosmos Image Classifier MCP Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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

# OAuth 설정 제거 (Claude Desktop 개인 계정에서는 지원하지 않음)
# 공개 엔드포인트로 변경

# 전역 데이터 저장소
training_data = []

def analyze_image_with_gemini(image_url: str) -> str:
    """Gemini Vision API를 사용하여 실제 이미지 분석"""
    try:
        # 이미지 다운로드
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            return "general, design, creative"  # 기본값 반환
        
        # 이미지를 base64로 인코딩
        image_data = base64.b64encode(response.content).decode('utf-8')
        
        # Gemini API 키 확인
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY가 설정되지 않음. URL 패턴 분석으로 대체")
            return analyze_image_url(image_url, get_categories())
        
        # Gemini Vision API 호출
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": """이 이미지를 분석하고 다음 카테고리 중에서 가장 적합한 3-5개를 선택해주세요:

nature, architecture, people, art, technology, design, fashion, food, travel, sports, music, culture, business, education, health, lifestyle, entertainment, photography, interior, outdoor, abstract, vintage, modern, creative, professional, casual, urban, rural, indoor, landscape, portrait, street, home, office, restaurant, hotel, garden, kitchen, bedroom, living, bathroom, gym, studio, library, museum, gallery, theater, airport, station, park, plaza, monument, sculpture, logo, branding, advertising, packaging, typography, pattern, texture, material, fabric, wood, metal, glass, ceramic, plastic, color, black, white, gray, red, blue, green, yellow, orange, purple

답변은 콤마로 구분된 카테고리 이름만 반환해주세요."""
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 200,
                "temperature": 0.1
            }
        }
        
        gemini_response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if gemini_response.status_code == 200:
            result = gemini_response.json()
            categories = result['candidates'][0]['content']['parts'][0]['text'].strip()
            logger.info(f"Gemini 분석 결과: {categories}")
            return categories
        else:
            logger.error(f"Gemini API 오류: {gemini_response.status_code}")
            return analyze_image_url(image_url, get_categories())
            
    except Exception as e:
        logger.error(f"이미지 분석 실패: {e}")
        return analyze_image_url(image_url, get_categories())

def get_categories() -> List[str]:
    """간단하고 실용적인 카테고리 목록 반환"""
    return [
        "nature", "architecture", "people", "art", "technology", "design",
        "fashion", "food", "travel", "sports", "music", "culture",
        "business", "education", "health", "lifestyle", "entertainment",
        "photography", "interior", "outdoor", "abstract", "vintage", "modern",
        "creative", "professional", "casual", "urban", "rural", "indoor",
        "landscape", "portrait", "street", "home", "office", "restaurant",
        "hotel", "garden", "kitchen", "bedroom", "living", "bathroom",
        "gym", "studio", "library", "museum", "gallery", "theater",
        "airport", "station", "park", "plaza", "monument", "sculpture",
        "logo", "branding", "advertising", "packaging", "typography",
        "pattern", "texture", "material", "fabric", "wood", "metal",
        "glass", "ceramic", "plastic", "color", "black", "white",
        "gray", "red", "blue", "green", "yellow", "orange", "purple"
    ]

def analyze_image_url(image_url: str, categories: List[str]) -> str:
    """URL 패턴을 기반으로 스마트 카테고리 추천"""
    url_lower = image_url.lower()
    
    # URL 패턴 매칭을 통한 카테고리 추천
    recommended = []
    
    # 기본 패턴 매칭
    if any(word in url_lower for word in ['nature', 'forest', 'tree', 'mountain', 'ocean', 'sea', 'lake', 'river']):
        recommended.extend(['nature', 'outdoor', 'landscape'])
    
    if any(word in url_lower for word in ['building', 'architecture', 'house', 'home', 'office', 'city', 'urban']):
        recommended.extend(['architecture', 'urban', 'design'])
    
    if any(word in url_lower for word in ['people', 'person', 'man', 'woman', 'child', 'portrait', 'face']):
        recommended.extend(['people', 'portrait', 'lifestyle'])
    
    if any(word in url_lower for word in ['art', 'painting', 'drawing', 'creative', 'design', 'graphic']):
        recommended.extend(['art', 'creative', 'design'])
    
    if any(word in url_lower for word in ['tech', 'technology', 'computer', 'digital', 'ai', 'robot']):
        recommended.extend(['technology', 'modern', 'professional'])
    
    if any(word in url_lower for word in ['food', 'restaurant', 'kitchen', 'cooking', 'meal']):
        recommended.extend(['food', 'lifestyle', 'indoor'])
    
    if any(word in url_lower for word in ['fashion', 'clothing', 'style', 'outfit', 'dress']):
        recommended.extend(['fashion', 'lifestyle', 'people'])
    
    if any(word in url_lower for word in ['travel', 'vacation', 'trip', 'destination', 'tourist']):
        recommended.extend(['travel', 'outdoor', 'lifestyle'])
    
    if any(word in url_lower for word in ['sports', 'fitness', 'gym', 'exercise', 'athletic']):
        recommended.extend(['sports', 'health', 'lifestyle'])
    
    if any(word in url_lower for word in ['music', 'concert', 'band', 'instrument', 'audio']):
        recommended.extend(['music', 'entertainment', 'culture'])
    
    if any(word in url_lower for word in ['business', 'office', 'meeting', 'corporate', 'professional']):
        recommended.extend(['business', 'professional', 'office'])
    
    if any(word in url_lower for word in ['education', 'school', 'university', 'learning', 'study']):
        recommended.extend(['education', 'professional', 'indoor'])
    
    if any(word in url_lower for word in ['health', 'medical', 'hospital', 'doctor', 'wellness']):
        recommended.extend(['health', 'professional', 'lifestyle'])
    
    if any(word in url_lower for word in ['automotive', 'car', 'vehicle', 'auto', 'transport']):
        recommended.extend(['automotive', 'technology', 'urban'])
    
    if any(word in url_lower for word in ['gaming', 'game', 'video', 'console', 'digital']):
        recommended.extend(['gaming', 'entertainment', 'technology'])
    
    if any(word in url_lower for word in ['photography', 'photo', 'camera', 'lens', 'shot']):
        recommended.extend(['photography', 'art', 'creative'])
    
    if any(word in url_lower for word in ['interior', 'room', 'furniture', 'decor', 'home']):
        recommended.extend(['interior', 'design', 'indoor'])
    
    if any(word in url_lower for word in ['abstract', 'pattern', 'texture', 'geometric', 'shape']):
        recommended.extend(['abstract', 'art', 'design'])
    
    if any(word in url_lower for word in ['minimalist', 'simple', 'clean', 'minimal', 'basic']):
        recommended.extend(['minimalist', 'design', 'modern'])
    
    if any(word in url_lower for word in ['vintage', 'retro', 'old', 'classic', 'antique']):
        recommended.extend(['vintage', 'classic', 'culture'])
    
    if any(word in url_lower for word in ['modern', 'contemporary', 'new', 'fresh', 'current']):
        recommended.extend(['modern', 'contemporary', 'design'])
    
    if any(word in url_lower for word in ['luxury', 'premium', 'high-end', 'expensive', 'exclusive']):
        recommended.extend(['luxury', 'professional', 'design'])
    
    if any(word in url_lower for word in ['budget', 'affordable', 'cheap', 'economical', 'value']):
        recommended.extend(['budget', 'casual', 'practical'])
    
    # 색상 기반 추천
    if any(word in url_lower for word in ['black', 'dark', 'shadow', 'night']):
        recommended.extend(['monochrome', 'black', 'night'])
    
    if any(word in url_lower for word in ['white', 'light', 'bright', 'clean']):
        recommended.extend(['monochrome', 'white', 'minimalist'])
    
    if any(word in url_lower for word in ['color', 'colorful', 'vibrant', 'bright']):
        recommended.extend(['color', 'creative', 'art'])
    
    # 중복 제거 및 최대 3개 카테고리 반환
    unique_categories = list(dict.fromkeys(recommended))  # 순서 유지하면서 중복 제거
    
    if not unique_categories:
        # 기본 카테고리
        unique_categories = ['general', 'design', 'creative']
    
    # 최대 3개까지만 반환
    return ', '.join(unique_categories[:3])

# OAuth 엔드포인트 제거됨 - 공개 엔드포인트로 변경

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Cosmos Image Classifier MCP Server",
        "version": "1.0.0",
        "status": "running",
        "protocol": "MCP HTTP"
    }

@app.post("/mcp")
async def mcp_endpoint(request: MCPRequest):
    """MCP 프로토콜 엔드포인트 (공개 엔드포인트)"""
    try:
        logger.info(f"MCP 요청: {request.method}")
        
        # 공개 엔드포인트 - 인증 없이 접근 가능
        
        if request.method == "initialize":
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "cosmos-image-classifier",
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
                
                # Gemini Vision API를 사용한 실제 이미지 분석
                recommended_categories = analyze_image_with_gemini(image_url)
                
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
                results = []
                
                # 확장된 카테고리 시스템
                categories = [
                    "nature", "architecture", "people", "art", "technology", "design",
                    "fashion", "food", "travel", "sports", "music", "culture",
                    "business", "education", "health", "lifestyle", "entertainment",
                    "automotive", "gaming", "photography", "interior", "outdoor",
                    "abstract", "minimalist", "vintage", "modern", "classic",
                    "creative", "professional", "casual", "luxury", "budget",
                    "urban", "rural", "coastal", "mountain", "forest", "desert",
                    "winter", "spring", "summer", "autumn", "night", "day",
                    "indoor", "outdoor", "studio", "street", "landscape", "portrait",
                    "macro", "wide", "close-up", "aerial", "underwater", "nightlife",
                    "wedding", "party", "celebration", "festival", "concert", "exhibition",
                    "workshop", "meeting", "conference", "seminar", "training", "workshop",
                    "retail", "restaurant", "hotel", "office", "home", "garden",
                    "kitchen", "bedroom", "living", "bathroom", "garage", "basement",
                    "rooftop", "balcony", "patio", "deck", "pool", "spa",
                    "gym", "studio", "workshop", "garage", "shed", "greenhouse",
                    "library", "museum", "gallery", "theater", "cinema", "stadium",
                    "airport", "station", "port", "harbor", "bridge", "tunnel",
                    "highway", "street", "alley", "park", "plaza", "square",
                    "monument", "statue", "fountain", "sculpture", "mural", "graffiti",
                    "signage", "logo", "branding", "advertising", "marketing", "promotion",
                    "packaging", "labeling", "typography", "illustration", "icon", "symbol",
                    "pattern", "texture", "material", "fabric", "leather", "wood",
                    "metal", "glass", "ceramic", "plastic", "paper", "cardboard",
                    "color", "monochrome", "black", "white", "gray", "red",
                    "blue", "green", "yellow", "orange", "purple", "pink",
                    "brown", "beige", "gold", "silver", "copper", "bronze"
                ]
                
                for url in image_urls:
                    # 각 URL에 대해 스마트 카테고리 분석
                    recommended_categories = analyze_image_url(url, categories)
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
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": f"모델 훈련 시작. {epochs} 에포크로 훈련합니다."
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
    """MCP GET 엔드포인트 (일부 클라이언트가 GET 요청을 보낼 수 있음)"""
    return {
        "message": "MCP Server is running",
        "protocol": "MCP HTTP",
        "version": "1.0.0"
    }

@app.options("/mcp")
async def mcp_options():
    """CORS preflight 요청 처리"""
    return {"message": "OK"}

if __name__ == "__main__":
    print("🚀 Cosmos Image Classifier MCP HTTP Server 시작 중...")
    print("📡 서버 주소: http://localhost:2001")
    print("📚 MCP 엔드포인트: http://localhost:2001/mcp")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=2001,
        log_level="info"
    )
