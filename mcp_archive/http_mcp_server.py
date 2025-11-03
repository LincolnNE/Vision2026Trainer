#!/usr/bin/env python3
"""
HTTP MCP 서버 - Claude Desktop 연동용
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random

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

# 전역 데이터 저장소
training_data = []

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Cosmos Image Classifier MCP Server",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "data_count": len(training_data)}

@app.post("/analyze_image")
async def analyze_image(request: Dict[str, Any]):
    """이미지 분석"""
    try:
        image_url = request.get("image_url")
        if not image_url:
            raise HTTPException(status_code=400, detail="image_url is required")
        
        # 간단한 분석 시뮬레이션
        categories = ['nature', 'architecture', 'art', 'people', 'objects', 'abstract', 'technology', 'food']
        category = random.choice(categories)
        confidence = random.uniform(0.7, 0.95)
        
        result = {
            "image_url": image_url,
            "category": category,
            "confidence": confidence,
            "analysis": f"이 이미지는 {category} 카테고리에 적합합니다.",
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/claude_auto_categorize")
async def claude_auto_categorize(request: Dict[str, Any]):
    """Claude AI 자동 카테고리 분류"""
    try:
        image_url = request.get("image_url")
        auto_apply = request.get("auto_apply", False)
        
        if not image_url:
            raise HTTPException(status_code=400, detail="image_url is required")
        
        # Claude AI 시뮬레이션
        categories = ['nature', 'architecture', 'art', 'people', 'objects', 'abstract', 'technology', 'food', 'fashion', 'culture']
        category = random.choice(categories)
        confidence = random.uniform(0.8, 0.98)
        
        if auto_apply:
            training_data.append({
                "image_url": image_url,
                "category": category,
                "confidence": confidence,
                "claude_mode": True
            })
            apply_text = "✅ 자동으로 훈련 데이터에 추가되었습니다."
        else:
            apply_text = "💡 자동 적용을 원하시면 auto_apply=true로 설정하세요."
        
        result = {
            "image_url": image_url,
            "category": category,
            "confidence": confidence,
            "analysis": f"Claude AI 분석: 이 이미지는 {category} 카테고리에 가장 적합합니다.",
            "apply_status": apply_text,
            "total_data": len(training_data),
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Claude auto categorize failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_categorize")
async def batch_categorize(request: Dict[str, Any]):
    """일괄 카테고리 분류"""
    try:
        image_urls = request.get("image_urls", [])
        strategy = request.get("strategy", "balanced")
        auto_apply = request.get("auto_apply", True)
        
        if not image_urls:
            raise HTTPException(status_code=400, detail="image_urls is required")
        
        results = []
        applied_count = 0
        
        for i, url in enumerate(image_urls):
            try:
                categories = ['nature', 'architecture', 'art', 'people', 'objects', 'abstract', 'technology', 'food']
                category = random.choice(categories)
                confidence = random.uniform(0.8, 0.95)
                
                if auto_apply:
                    training_data.append({
                        "image_url": url,
                        "category": category,
                        "confidence": confidence,
                        "claude_mode": True
                    })
                    applied_count += 1
                
                results.append({
                    "index": i + 1,
                    "url": url,
                    "category": category,
                    "confidence": confidence
                })
                
            except Exception as e:
                results.append({
                    "index": i + 1,
                    "url": url,
                    "error": str(e)
                })
        
        strategy_text = {
            "conservative": "보수적 전략 (높은 신뢰도 우선)",
            "aggressive": "적극적 전략 (다양한 카테고리 탐색)",
            "balanced": "균형 전략 (정확도와 다양성 균형)"
        }.get(strategy, "균형 전략")
        
        result = {
            "strategy": strategy_text,
            "total_images": len(image_urls),
            "applied_count": applied_count,
            "results": results,
            "total_data": len(training_data),
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Batch categorize failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_status")
async def get_status():
    """상태 확인"""
    try:
        if not training_data:
            return {
                "message": "현재 훈련 데이터가 없습니다.",
                "total_images": 0,
                "categories": [],
                "status": "success"
            }
        
        categories = [item["category"] for item in training_data]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        
        result = {
            "total_images": len(training_data),
            "category_count": len(set(categories)),
            "category_distribution": category_counts,
            "average_confidence": sum(item['confidence'] for item in training_data) / len(training_data),
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Get status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export_data")
async def export_data():
    """데이터 내보내기"""
    try:
        return {
            "training_data": training_data,
            "total_count": len(training_data),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Export data failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Cosmos Image Classifier HTTP MCP Server 시작 중...")
    print("📡 서버 주소: http://localhost:2000")
    print("📚 API 문서: http://localhost:2000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=2000,
        log_level="info"
    )
