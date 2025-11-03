#!/usr/bin/env python3
"""
SSL 인증서 생성 스크립트
개발용 자체 서명된 인증서 생성
"""

import subprocess
import os
import sys
from pathlib import Path

def generate_self_signed_cert():
    """자체 서명된 SSL 인증서 생성"""
    try:
        # OpenSSL이 설치되어 있는지 확인
        subprocess.run(["openssl", "version"], check=True, capture_output=True)
        
        print("🔐 SSL 인증서 생성 중...")
        
        # 개인 키 생성
        subprocess.run([
            "openssl", "genrsa", 
            "-out", "key.pem", 
            "2048"
        ], check=True)
        
        # 인증서 생성
        subprocess.run([
            "openssl", "req", 
            "-new", "-x509", 
            "-key", "key.pem", 
            "-out", "cert.pem", 
            "-days", "365",
            "-subj", "/C=KR/ST=Seoul/L=Seoul/O=Cosmos/OU=IT/CN=mcp.cdnscraper.dev"
        ], check=True)
        
        print("✅ SSL 인증서 생성 완료!")
        print("📁 생성된 파일:")
        print("   - cert.pem (인증서)")
        print("   - key.pem (개인 키)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ SSL 인증서 생성 실패: {e}")
        return False
    except FileNotFoundError:
        print("❌ OpenSSL이 설치되지 않았습니다.")
        print("   macOS: brew install openssl")
        print("   Ubuntu: sudo apt-get install openssl")
        return False

def generate_letsencrypt_cert():
    """Let's Encrypt 인증서 생성 (실제 도메인용)"""
    print("🌐 Let's Encrypt 인증서 생성 (실제 도메인용)")
    print("   이 방법은 실제 도메인에서만 작동합니다.")
    print("   certbot을 사용하여 인증서를 발급받으세요:")
    print("   sudo certbot certonly --standalone -d mcp.cdnscraper.dev")

def main():
    """메인 함수"""
    print("🔒 SSL 인증서 생성 도구")
    print("=" * 40)
    
    # 현재 디렉토리 확인
    current_dir = Path.cwd()
    print(f"📂 작업 디렉토리: {current_dir}")
    
    # 기존 인증서 파일 확인
    cert_file = Path("cert.pem")
    key_file = Path("key.pem")
    
    if cert_file.exists() and key_file.exists():
        print("⚠️ 기존 SSL 인증서가 발견되었습니다.")
        response = input("새로 생성하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("인증서 생성을 취소했습니다.")
            return
    
    print("\n인증서 생성 방법을 선택하세요:")
    print("1. 자체 서명된 인증서 (개발용)")
    print("2. Let's Encrypt 인증서 (실제 도메인용)")
    print("3. 취소")
    
    choice = input("선택 (1-3): ")
    
    if choice == "1":
        if generate_self_signed_cert():
            print("\n🎉 개발용 SSL 인증서가 생성되었습니다!")
            print("   이제 HTTPS MCP 서버를 실행할 수 있습니다.")
    elif choice == "2":
        generate_letsencrypt_cert()
    elif choice == "3":
        print("인증서 생성을 취소했습니다.")
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()
