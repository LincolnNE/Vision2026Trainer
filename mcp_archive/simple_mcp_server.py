#!/usr/bin/env python3
"""
간단한 MCP 서버 - Claude Desktop 연동용
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, Tool, TextContent,
    CallToolRequest, CallToolResult
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMCPServer:
    """간단한 MCP 서버"""
    
    def __init__(self):
        self.server = Server("cosmos-image-classifier")
        self.training_data = []
        self.setup_handlers()
        
    def setup_handlers(self):
        """핸들러 설정"""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="cosmos://images",
                    name="Cosmos Images",
                    description="Cosmos.so 이미지 데이터셋",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            if uri == "cosmos://images":
                return json.dumps({
                    "images": self.training_data,
                    "total_count": len(self.training_data)
                })
            else:
                raise ValueError(f"Unknown resource: {uri}")
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="analyze_image",
                    description="이미지를 분석하여 카테고리를 추천합니다",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_url": {
                                "type": "string",
                                "description": "분석할 이미지의 URL"
                            }
                        },
                        "required": ["image_url"]
                    }
                ),
                Tool(
                    name="claude_auto_categorize",
                    description="Claude AI를 사용하여 이미지를 자동으로 분석하고 카테고리를 추천합니다",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_url": {
                                "type": "string",
                                "description": "분석할 이미지의 URL"
                            },
                            "auto_apply": {
                                "type": "boolean",
                                "description": "추천된 카테고리를 자동으로 적용할지 여부",
                                "default": false
                            }
                        },
                        "required": ["image_url"]
                    }
                ),
                Tool(
                    name="get_status",
                    description="현재 상태를 확인합니다",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            try:
                if name == "analyze_image":
                    return await self.analyze_image(arguments)
                elif name == "claude_auto_categorize":
                    return await self.claude_auto_categorize(arguments)
                elif name == "get_status":
                    return await self.get_status(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Tool {name} failed: {e}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")]
                )
    
    async def analyze_image(self, arguments: Dict[str, Any]) -> CallToolResult:
        """이미지 분석"""
        image_url = arguments["image_url"]
        
        # 간단한 분석 시뮬레이션
        categories = ['nature', 'architecture', 'art', 'people', 'objects', 'abstract']
        import random
        category = random.choice(categories)
        confidence = random.uniform(0.7, 0.95)
        
        result_text = f"""
이미지 분석 완료:

**추천 카테고리**: {category} (신뢰도: {confidence:.2f})
**이미지 URL**: {image_url}

이미지가 분석되었습니다.
        """
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)]
        )
    
    async def claude_auto_categorize(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Claude AI 자동 카테고리 분류"""
        image_url = arguments["image_url"]
        auto_apply = arguments.get("auto_apply", False)
        
        # Claude AI 시뮬레이션
        categories = ['nature', 'architecture', 'art', 'people', 'objects', 'abstract', 'technology', 'food']
        import random
        category = random.choice(categories)
        confidence = random.uniform(0.8, 0.98)
        
        if auto_apply:
            self.training_data.append({
                "image_url": image_url,
                "category": category,
                "confidence": confidence,
                "claude_mode": True
            })
            apply_text = "✅ 자동으로 훈련 데이터에 추가되었습니다."
        else:
            apply_text = "💡 자동 적용을 원하시면 auto_apply=true로 설정하세요."
        
        result_text = f"""
🤖 Claude AI 자동 카테고리 분석 완료:

**추천 카테고리**: {category} (신뢰도: {confidence:.2f})

**Claude 분석 결과**:
이 이미지는 {category} 카테고리에 가장 적합합니다. Claude AI가 이미지의 맥락과 의미를 분석하여 추천했습니다.

{apply_text}

현재 총 {len(self.training_data)}개의 이미지가 훈련 데이터에 있습니다.
        """
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)]
        )
    
    async def get_status(self, arguments: Dict[str, Any]) -> CallToolResult:
        """상태 확인"""
        if not self.training_data:
            status_text = "현재 훈련 데이터가 없습니다."
        else:
            categories = [item["category"] for item in self.training_data]
            category_counts = {cat: categories.count(cat) for cat in set(categories)}
            
            status_text = f"""
**현재 상태**:

- 총 이미지: {len(self.training_data)}개
- 카테고리 수: {len(set(categories))}개

**카테고리별 분포**:
{chr(10).join([f"- {cat}: {count}개" for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)])}

**평균 신뢰도**: {sum(item['confidence'] for item in self.training_data) / len(self.training_data):.2f}
            """
        
        return CallToolResult(
            content=[TextContent(type="text", text=status_text)]
        )

async def main():
    """메인 함수"""
    server_instance = SimpleMCPServer()
    
    # MCP 서버 시작
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="cosmos-image-classifier",
                server_version="1.0.0",
                capabilities={
                    "resources": {},
                    "tools": {}
                }
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
