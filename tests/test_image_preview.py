#!/usr/bin/env python3
"""
이미지 미리보기 테스트 스크립트
"""

import requests
from PIL import Image
import io
import tkinter as tk
from tkinter import ttk
import threading
import time

def test_image_loading():
    """이미지 로딩 테스트"""
    # 실제 Cosmos CDN 이미지 URL
    test_urls = [
        "https://cdn.cosmos.so/646ade0c-beff-4003-bcae-c977de3ea7dd?format=webp&w=1080",
        "https://cdn.cosmos.so/a22716e5-1442-432c-b320-05b3ad24deec?rect=33%2C0%2C528%2C529&format=webp&w=1080",
        "https://cdn.cosmos.so/f85e4901-04d7-4a73-8e47-ac812eef354e?format=webp&w=1080"
    ]
    
    print("이미지 로딩 테스트 시작...")
    
    for i, url in enumerate(test_urls):
        try:
            print(f"\n테스트 {i+1}: {url}")
            
            # 이미지 다운로드
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            print(f"  ✅ 다운로드 성공: {len(response.content)} bytes")
            
            # PIL Image로 변환
            image = Image.open(io.BytesIO(response.content))
            print(f"  ✅ 이미지 변환 성공: {image.size} ({image.mode})")
            
            # 크기 조정
            max_size = (400, 300)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            print(f"  ✅ 크기 조정 완료: {image.size}")
            
            # 저장 (테스트용)
            filename = f"test_image_{i+1}.jpg"
            image.save(filename, "JPEG")
            print(f"  ✅ 저장 완료: {filename}")
            
        except Exception as e:
            print(f"  ❌ 오류: {str(e)}")
    
    print("\n이미지 로딩 테스트 완료!")

def test_gui_preview():
    """GUI 미리보기 테스트"""
    root = tk.Tk()
    root.title("이미지 미리보기 테스트")
    root.geometry("500x400")
    
    # 이미지 라벨
    image_label = ttk.Label(root, text="이미지를 선택하세요", 
                           background='white', relief=tk.SUNKEN)
    image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 버튼들
    button_frame = ttk.Frame(root)
    button_frame.pack(fill=tk.X, padx=10, pady=5)
    
    def load_test_image():
        """테스트 이미지 로딩"""
        url = "https://cdn.cosmos.so/646ade0c-beff-4003-bcae-c977de3ea7dd?format=webp&w=1080"
        
        def load_thread():
            try:
                image_label.config(text="로딩 중...")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content))
                image.thumbnail((400, 300), Image.Resampling.LANCZOS)
                
                photo = tk.PhotoImage(image)
                
                root.after(0, lambda: update_display(photo))
                
            except Exception as e:
                root.after(0, lambda: show_error(str(e)))
        
        def update_display(photo):
            image_label.config(image=photo, text="")
            image_label.image = photo
        
        def show_error(error_msg):
            image_label.config(image="", text=f"오류: {error_msg}")
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    ttk.Button(button_frame, text="테스트 이미지 로딩", command=load_test_image).pack(side=tk.LEFT)
    ttk.Button(button_frame, text="종료", command=root.quit).pack(side=tk.RIGHT)
    
    root.mainloop()

if __name__ == "__main__":
    print("이미지 미리보기 테스트")
    print("1. 콘솔 테스트")
    print("2. GUI 테스트")
    
    choice = input("선택하세요 (1 또는 2): ")
    
    if choice == "1":
        test_image_loading()
    elif choice == "2":
        test_gui_preview()
    else:
        print("잘못된 선택입니다.")
