#!/usr/bin/env python3
"""
MCP 서버 테스트 스크립트
로컬 및 Cloudflare 연결 테스트
"""

import requests
import json
import time
import sys
from typing import Dict, Any

class MCPTester:
    """MCP 서버 테스트 클래스"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.verify = False  # SSL 인증서 검증 비활성화 (개발용)
        
        # SSL 경고 억제
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def test_health(self) -> bool:
        """헬스 체크 테스트"""
        try:
            print(f"🔍 헬스 체크 테스트: {self.base_url}/health")
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 헬스 체크 성공: {data}")
                return True
            else:
                print(f"❌ 헬스 체크 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 헬스 체크 오류: {e}")
            return False
    
    def test_mcp_initialize(self) -> bool:
        """MCP 초기화 테스트"""
        try:
            print(f"🔍 MCP 초기화 테스트: {self.base_url}/mcp")
            
            payload = {
                "jsonrpc": "2.0",
                "id": "test-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"}
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/mcp",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data:
                    print(f"✅ MCP 초기화 성공: {data['result']}")
                    return True
                else:
                    print(f"❌ MCP 초기화 실패: {data}")
                    return False
            else:
                print(f"❌ MCP 초기화 HTTP 오류: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ MCP 초기화 오류: {e}")
            return False
    
    def test_mcp_tools_list(self) -> bool:
        """MCP 도구 목록 테스트"""
        try:
            print(f"🔍 MCP 도구 목록 테스트: {self.base_url}/mcp")
            
            payload = {
                "jsonrpc": "2.0",
                "id": "test-2",
                "method": "tools/list",
                "params": {}
            }
            
            response = self.session.post(
                f"{self.base_url}/mcp",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data and "tools" in data["result"]:
                    tools = data["result"]["tools"]
                    print(f"✅ MCP 도구 목록 성공: {len(tools)}개 도구")
                    for tool in tools:
                        print(f"   - {tool['name']}: {tool['description']}")
                    return True
                else:
                    print(f"❌ MCP 도구 목록 실패: {data}")
                    return False
            else:
                print(f"❌ MCP 도구 목록 HTTP 오류: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ MCP 도구 목록 오류: {e}")
            return False
    
    def test_mcp_analyze_image(self) -> bool:
        """MCP 이미지 분석 테스트"""
        try:
            print(f"🔍 MCP 이미지 분석 테스트: {self.base_url}/mcp")
            
            # 테스트용 이미지 URL
            test_image_url = "https://cdn.cosmos.so/646ade0c-beff-4003-bcae-c977de3ea7dd?format=webp&w=1080"
            
            payload = {
                "jsonrpc": "2.0",
                "id": "test-3",
                "method": "tools/call",
                "params": {
                    "name": "analyze_image",
                    "arguments": {
                        "image_url": test_image_url,
                        "context": "테스트 이미지 분석"
                    }
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/mcp",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data and "content" in data["result"]:
                    content = data["result"]["content"][0]["text"]
                    print(f"✅ MCP 이미지 분석 성공: {content}")
                    return True
                else:
                    print(f"❌ MCP 이미지 분석 실패: {data}")
                    return False
            else:
                print(f"❌ MCP 이미지 분석 HTTP 오류: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ MCP 이미지 분석 오류: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """모든 테스트 실행"""
        print(f"🚀 MCP 서버 테스트 시작: {self.base_url}")
        print("=" * 60)
        
        tests = {
            "헬스 체크": self.test_health,
            "MCP 초기화": self.test_mcp_initialize,
            "MCP 도구 목록": self.test_mcp_tools_list,
            "MCP 이미지 분석": self.test_mcp_analyze_image
        }
        
        results = {}
        
        for test_name, test_func in tests.items():
            print(f"\n📋 {test_name} 테스트 중...")
            try:
                result = test_func()
                results[test_name] = result
                time.sleep(1)  # 테스트 간 간격
            except Exception as e:
                print(f"❌ {test_name} 테스트 중 오류: {e}")
                results[test_name] = False
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """테스트 결과 요약"""
        print("\n" + "=" * 60)
        print("📊 테스트 결과 요약")
        print("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ 통과" if result else "❌ 실패"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 모든 테스트가 통과했습니다!")
        else:
            print("⚠️ 일부 테스트가 실패했습니다. 설정을 확인해주세요.")

def main():
    """메인 함수"""
    print("🔧 MCP 서버 테스트 도구")
    print("=" * 60)
    
    # 테스트할 서버 URL들
    test_urls = [
        "https://localhost:3000",  # 로컬 HTTPS 서버
        "http://localhost:2001",   # 로컬 HTTP 서버
        "https://mcp.cdnscraper.dev"  # Cloudflare 서버
    ]
    
    print("테스트할 서버를 선택하세요:")
    for i, url in enumerate(test_urls, 1):
        print(f"{i}. {url}")
    print("4. 모든 서버 테스트")
    
    try:
        choice = int(input("\n선택 (1-4): "))
        
        if choice == 4:
            # 모든 서버 테스트
            for url in test_urls:
                print(f"\n{'='*80}")
                print(f"🌐 서버 테스트: {url}")
                print(f"{'='*80}")
                
                tester = MCPTester(url)
                results = tester.run_all_tests()
                tester.print_summary(results)
                
                if url != test_urls[-1]:  # 마지막이 아니면 대기
                    input("\n다음 서버 테스트를 계속하려면 Enter를 누르세요...")
        else:
            # 선택된 서버 테스트
            if 1 <= choice <= len(test_urls):
                url = test_urls[choice - 1]
                tester = MCPTester(url)
                results = tester.run_all_tests()
                tester.print_summary(results)
            else:
                print("잘못된 선택입니다.")
                
    except ValueError:
        print("잘못된 입력입니다.")
    except KeyboardInterrupt:
        print("\n테스트가 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
