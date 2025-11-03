#!/usr/bin/env python3
"""
MCP 서버 실행 가이드
Cloudflare 연동을 위한 단계별 실행 스크립트
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_requirements():
    """필요한 패키지 확인"""
    print("🔍 필요한 패키지 확인 중...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "requests",
        "pandas",
        "numpy",
        "torch",
        "torchvision",
        "scikit-learn",
        "matplotlib",
        "pillow"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (설치 필요)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 다음 패키지들을 설치해주세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 모든 필요한 패키지가 설치되어 있습니다.")
    return True

def generate_ssl_certificate():
    """SSL 인증서 생성"""
    print("\n🔐 SSL 인증서 생성 중...")
    
    cert_file = Path("cert.pem")
    key_file = Path("key.pem")
    
    if cert_file.exists() and key_file.exists():
        print("✅ SSL 인증서가 이미 존재합니다.")
        return True
    
    try:
        # 자체 서명된 인증서 생성
        subprocess.run([
            "openssl", "genrsa", 
            "-out", "key.pem", 
            "2048"
        ], check=True, capture_output=True)
        
        subprocess.run([
            "openssl", "req", 
            "-new", "-x509", 
            "-key", "key.pem", 
            "-out", "cert.pem", 
            "-days", "365",
            "-subj", "/C=KR/ST=Seoul/L=Seoul/O=Cosmos/OU=IT/CN=mcp.cdnscraper.dev"
        ], check=True, capture_output=True)
        
        print("✅ SSL 인증서 생성 완료!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ SSL 인증서 생성 실패: {e}")
        print("OpenSSL이 설치되어 있는지 확인해주세요.")
        return False
    except FileNotFoundError:
        print("❌ OpenSSL이 설치되지 않았습니다.")
        print("macOS: brew install openssl")
        print("Ubuntu: sudo apt-get install openssl")
        return False

def start_server(server_type: str):
    """서버 시작"""
    print(f"\n🚀 {server_type} 서버 시작 중...")
    
    server_scripts = {
        "http": "mcp_http_server.py",
        "https": "https_mcp_server.py",
        "real": "cosmos_mcp_real_server.py",
        "basic": "cosmos_mcp_server.py"
    }
    
    script = server_scripts.get(server_type)
    if not script:
        print(f"❌ 알 수 없는 서버 타입: {server_type}")
        return False
    
    script_path = Path(script)
    if not script_path.exists():
        print(f"❌ 서버 스크립트를 찾을 수 없습니다: {script}")
        return False
    
    try:
        # 서버 실행
        process = subprocess.Popen([
            sys.executable, script
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print(f"✅ {server_type} 서버가 시작되었습니다.")
        print(f"프로세스 ID: {process.pid}")
        
        # 서버 상태 확인
        time.sleep(2)
        
        if process.poll() is None:
            print("✅ 서버가 정상적으로 실행 중입니다.")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ 서버 시작 실패:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 서버 시작 오류: {e}")
        return False

def test_server(server_url: str):
    """서버 테스트"""
    print(f"\n🧪 서버 테스트: {server_url}")
    
    try:
        import requests
        import urllib3
        
        # SSL 경고 억제
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 헬스 체크
        response = requests.get(f"{server_url}/health", verify=False, timeout=5)
        
        if response.status_code == 200:
            print("✅ 서버 헬스 체크 성공")
            return True
        else:
            print(f"❌ 서버 헬스 체크 실패: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 서버 테스트 오류: {e}")
        return False

def main():
    """메인 함수"""
    print("🚀 MCP 서버 실행 가이드")
    print("=" * 50)
    
    # 1. 요구사항 확인
    if not check_requirements():
        print("\n❌ 필요한 패키지를 먼저 설치해주세요.")
        return
    
    # 2. 서버 타입 선택
    print("\n📋 서버 타입을 선택하세요:")
    print("1. HTTP 서버 (포트 2001)")
    print("2. HTTPS 서버 (포트 3000)")
    print("3. 실제 Claude API 서버")
    print("4. 기본 MCP 서버")
    print("5. 모든 서버 테스트")
    
    try:
        choice = int(input("\n선택 (1-5): "))
        
        if choice == 1:
            # HTTP 서버
            if start_server("http"):
                test_server("http://localhost:2001")
            
        elif choice == 2:
            # HTTPS 서버
            if generate_ssl_certificate():
                if start_server("https"):
                    test_server("https://localhost:3000")
            
        elif choice == 3:
            # 실제 Claude API 서버
            api_key = input("Claude API 키를 입력하세요 (선택사항): ")
            if api_key:
                os.environ["CLAUDE_API_KEY"] = api_key
            
            if start_server("real"):
                print("✅ 실제 Claude API 서버가 시작되었습니다.")
            
        elif choice == 4:
            # 기본 MCP 서버
            if start_server("basic"):
                print("✅ 기본 MCP 서버가 시작되었습니다.")
            
        elif choice == 5:
            # 모든 서버 테스트
            print("\n🧪 모든 서버 테스트 중...")
            
            servers = [
                ("HTTP", "http://localhost:2001"),
                ("HTTPS", "https://localhost:3000"),
                ("Cloudflare", "https://mcp.cdnscraper.dev")
            ]
            
            for name, url in servers:
                print(f"\n{name} 서버 테스트:")
                test_server(url)
        
        else:
            print("잘못된 선택입니다.")
            
    except ValueError:
        print("잘못된 입력입니다.")
    except KeyboardInterrupt:
        print("\n실행이 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
