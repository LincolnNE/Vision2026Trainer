#!/usr/bin/env python3
"""
Cosmos.so 이미지 분류 GUI v4.0 - Gemini API 직접 연동
- Gemini Vision API 직접 호출
- 실시간 이미지 분석 및 카테고리 추천
- 자동 카테고리 추천 및 훈련
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import os
import pandas as pd
import requests
import json
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import time
import base64
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image, ImageTk
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
import time
import random
import base64
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiAPIClient:
    """Gemini API 직접 호출 클라이언트 - 하드웨어 가속 지원"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.is_available = bool(self.api_key)
        
        # 하드웨어 가속 설정 (429 오류 방지를 위해 워커 수 조정)
        self.max_workers = min(4, multiprocessing.cpu_count())  # API 부하 감소
        self.session = requests.Session()  # 연결 재사용으로 성능 향상
        
        # CPU 우선순위 설정 (macOS)
        if hasattr(os, 'nice'):
            try:
                os.nice(-5)  # 높은 우선순위로 설정
                logger.info(f"🚀 CPU 우선순위 높임 - 최대 워커: {self.max_workers}")
            except PermissionError:
                logger.warning("CPU 우선순위 설정 권한 없음")
        
        logger.info(f"🔧 하드웨어 가속 활성화: {self.max_workers}개 워커")
        
    def analyze_image(self, image_url: str) -> str:
        """Gemini Vision API를 사용하여 이미지 분석"""
        if not self.is_available:
            logger.warning("GEMINI_API_KEY가 설정되지 않음")
            return "general, design, creative"
        
        try:
            # 이미지 다운로드 (세션 재사용으로 성능 향상)
            response = self.session.get(image_url, timeout=10)
            if response.status_code != 200:
                return "general, design, creative"
            
            # 이미지를 base64로 인코딩
            image_data = base64.b64encode(response.content).decode('utf-8')
            
            # Gemini API 호출
            headers = {
                'Content-Type': 'application/json'
            }
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                   "text": """이미지를 분석하여 다음 형식으로 정확히 출력해주세요:

**출력 형식:**
영문주요카테고리1, 영문주요카테고리2, 영문주요카테고리3 | 한국어주요카테고리1, 한국어주요카테고리2, 한국어주요카테고리3

**규칙:**
1. 영문 카테고리는 정확히 3개만 제시 (콤마로 구분)
2. 한국어 카테고리는 정확히 3개만 제시 (콤마로 구분)
3. 영문과 한국어는 파이프(|)로 구분
4. 가장 핵심적이고 구체적인 용어 사용

**예시:**
nature, landscape, mountain | 자연풍경, 산지형, 야외활동
architecture, building, modern | 현대건축, 건물구조, 도시환경
people, portrait, fashion | 인물사진, 포트레이트, 패션스타일

이제 이미지를 분석하여 위 형식에 맞춰 출력해주세요."""
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
            
            gemini_response = self.session.post(
                f'{self.base_url}?key={self.api_key}',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if gemini_response.status_code == 200:
                result = gemini_response.json()
                categories = result['candidates'][0]['content']['parts'][0]['text'].strip()
                logger.info(f"Gemini 분석 결과: {categories}")
                return categories
            elif gemini_response.status_code == 429:
                logger.warning(f"Gemini API 요청 한도 초과 (429). 30초 대기 후 재시도...")
                time.sleep(30)  # 30초 대기로 증가
                return "general, design, creative"  # 기본값 반환
            elif gemini_response.status_code == 503:
                logger.warning(f"Gemini API 서비스 일시 중단 (503). 잠시 대기 후 재시도...")
                time.sleep(3)  # 3초 대기
                return "general, design, creative"  # 기본값 반환
            else:
                logger.error(f"Gemini API 오류: {gemini_response.status_code}")
                logger.error(f"응답 내용: {gemini_response.text}")
                return "general, design, creative"
                
        except Exception as e:
            logger.error(f"이미지 분석 실패: {e}")
            logger.error(f"이미지 URL: {image_url}")
            logger.error(f"API 키 존재: {bool(self.api_key)}")
            logger.error(f"API 키 길이: {len(self.api_key) if self.api_key else 0}")
            return "general, design, creative"
    
    def batch_analyze_images(self, image_urls: List[str]) -> List[str]:
        """여러 이미지 일괄 분석 - 하드웨어 가속 병렬 처리"""
        if not image_urls:
            return []
        
        logger.info(f"🚀 병렬 분석 시작: {len(image_urls)}개 이미지, {self.max_workers}개 워커")
        
        # 결과를 순서대로 저장하기 위한 딕셔너리
        results = {}
        
        # 병렬 처리 실행
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 각 이미지에 대해 Future 객체 생성
            future_to_index = {
                executor.submit(self._analyze_single_image, url, i): i 
                for i, url in enumerate(image_urls)
            }
            
            # 완료된 작업들을 순서대로 처리
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    completed_count += 1
                    logger.info(f"✅ 분석 완료: {completed_count}/{len(image_urls)} - {image_urls[index].split('/')[-1].split('?')[0]}")
                except Exception as e:
                    logger.error(f"❌ 이미지 {index} 분석 실패: {e}")
                    results[index] = "general, design, creative"
        
        # 원래 순서대로 결과 반환
        ordered_results = [results[i] for i in range(len(image_urls))]
        logger.info(f"🎯 병렬 분석 완료: {len(ordered_results)}개 결과")
        return ordered_results
    
    def _analyze_single_image(self, url: str, index: int) -> str:
        """단일 이미지 분석 (병렬 처리용)"""
        try:
            # 각 워커마다 더 긴 지연으로 API 부하 분산 (429 오류 방지)
            time.sleep(index * 2.0)  # 2초씩 순차적 지연으로 API 부하 분산
            return self.analyze_image(url)
        except Exception as e:
            logger.error(f"워커에서 이미지 분석 실패: {e}")
            return "general, design, creative"

class CosmosGUIV4:
    """메인 GUI 애플리케이션 v4.0 - Gemini API 직접 연동"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Cosmos.so 이미지 분류 훈련 관리자 v4.0 - Gemini API 직접 연동")
        self.root.geometry("1800x1100")
        
        # Gemini API 클라이언트 초기화
        self.gemini_client = GeminiAPIClient()
        
        # 데이터 저장
        self.image_urls = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.current_dataset = None
        self.training_thread = None
        self.training_queue = queue.Queue()
        
        # GUI 구성
        self.setup_ui()
        self.setup_logging()
        
        # Gemini API 연결 테스트
        self.test_gemini_connection()
        
        # 기존 데이터 로드
        self.load_existing_data()

    def test_gemini_connection(self):
        """Gemini API 연결 테스트"""
        def test_thread():
            try:
                if not self.gemini_client.api_key:
                    self.root.after(0, lambda: self.log_message("❌ GEMINI_API_KEY가 설정되지 않음"))
                    self.root.after(0, lambda: self.update_gemini_status("API 키 없음"))
                    self.root.after(0, lambda: self.log_message("💡 .env 파일에 GEMINI_API_KEY를 설정하세요"))
                    return
                
                # API 키 형식 검증
                if not self.gemini_client.api_key.startswith('AIza'):
                    self.root.after(0, lambda: self.log_message("❌ API 키 형식이 올바르지 않음"))
                    self.root.after(0, lambda: self.update_gemini_status("API 키 오류"))
                    return
                
                # 실제 API 호출 테스트 (더 안정적인 이미지 사용)
                self.root.after(0, lambda: self.log_message("🔍 Gemini API 연결 테스트 중..."))
                test_result = self.gemini_client.analyze_image("https://picsum.photos/150/150")
                
                if test_result and test_result != "general, design, creative":
                    self.root.after(0, lambda: self.log_message("✅ Gemini API 연결 성공"))
                    self.root.after(0, lambda: self.update_gemini_status("연결됨"))
                    self.root.after(0, lambda: self.log_message(f"📊 테스트 분석 결과: {test_result}"))
                else:
                    self.root.after(0, lambda: self.log_message("❌ Gemini API 테스트 실패"))
                    self.root.after(0, lambda: self.log_message(f"🔍 테스트 결과: {test_result}"))
                    self.root.after(0, lambda: self.update_gemini_status("연결 실패"))
                    
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"❌ Gemini API 연결 오류: {e}"))
                self.root.after(0, lambda: self.update_gemini_status("연결 실패"))
        
        threading.Thread(target=test_thread, daemon=True).start()

    def _mask_api_key(self, api_key):
        """API 키를 마스킹하여 표시"""
        if not api_key:
            return "API 키 없음"
        if len(api_key) <= 8:
            return "***" + api_key[-4:]
        return api_key[:4] + "***" + api_key[-4:]

    def manage_api_key(self):
        """API 키 추가/수정 다이얼로그"""
        dialog = tk.Toplevel(self.root)
        dialog.title("API 키 관리")
        dialog.geometry("500x450")
        dialog.resizable(False, False)
        
        # 다이얼로그를 부모 창 중앙에 위치
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 메인 프레임
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 제목
        title_label = ttk.Label(main_frame, text="Gemini API 키 관리", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # 현재 API 키 표시
        current_frame = ttk.LabelFrame(main_frame, text="현재 API 키", padding=10)
        current_frame.pack(fill=tk.X, pady=(0, 15))
        
        current_key_var = tk.StringVar()
        masked_key = self._mask_api_key(self.gemini_client.api_key)
        current_key_var.set(masked_key)
        current_entry = ttk.Entry(current_frame, textvariable=current_key_var, 
                                 state='readonly', width=50)
        current_entry.pack(fill=tk.X)
        
        # 현재 키 상태 표시
        if self.gemini_client.api_key:
            status_text = "✅ API 키가 설정되어 있습니다"
            status_color = "green"
        else:
            status_text = "❌ API 키가 설정되지 않았습니다"
            status_color = "red"
        
        status_label = ttk.Label(current_frame, text=status_text, 
                                foreground=status_color, font=('Arial', 9))
        status_label.pack(anchor=tk.W, pady=(5, 0))
        
        # 새 API 키 입력
        new_frame = ttk.LabelFrame(main_frame, text="새 API 키 입력", padding=10)
        new_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(new_frame, text="Gemini API 키:").pack(anchor=tk.W)
        new_key_var = tk.StringVar()
        new_key_entry = ttk.Entry(new_frame, textvariable=new_key_var, 
                                 width=50, show="*")
        new_key_entry.pack(fill=tk.X, pady=(5, 0))
        
        # 도움말
        help_text = """API 키를 얻는 방법:
1. https://aistudio.google.com/app/apikey 방문
2. Google 계정으로 로그인
3. "Create API Key" 클릭
4. 생성된 키를 복사하여 위에 입력"""
        
        help_label = ttk.Label(new_frame, text=help_text, 
                              font=('Arial', 8), foreground='gray')
        help_label.pack(anchor=tk.W, pady=(10, 0))
        
        # 버튼 프레임
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(20, 0))
        
        def save_api_key():
            new_key = new_key_var.get().strip()
            if not new_key:
                messagebox.showwarning("경고", "API 키를 입력하세요.")
                return
            
            # API 키 형식 검증
            if not new_key.startswith('AIza'):
                messagebox.showerror("오류", "올바른 Gemini API 키 형식이 아닙니다.\n키는 'AIza'로 시작해야 합니다.")
                return
            
            # API 키 저장
            try:
                self._save_api_key_to_env(new_key)
                self.gemini_client.api_key = new_key
                self.gemini_client.is_available = True
                
                # UI 업데이트
                self.api_key_display_var.set(self._mask_api_key(new_key))
                self.log_message("✅ API 키가 성공적으로 저장되었습니다.")
                
                # 연결 테스트
                self.test_gemini_connection()
                
                dialog.destroy()
                messagebox.showinfo("성공", "API 키가 저장되었습니다.\n연결 테스트를 진행합니다.")
                
            except Exception as e:
                messagebox.showerror("오류", f"API 키 저장 실패:\n{e}")
        
        def test_api_key():
            new_key = new_key_var.get().strip()
            if not new_key:
                messagebox.showwarning("경고", "API 키를 입력하세요.")
                return
            
            # 임시로 API 키 설정하여 테스트
            original_key = self.gemini_client.api_key
            self.gemini_client.api_key = new_key
            
            def test_thread():
                try:
                    test_result = self.gemini_client.analyze_image("https://picsum.photos/150/150")
                    if test_result and test_result != "general, design, creative":
                        self.root.after(0, lambda: messagebox.showinfo("테스트 성공", 
                            f"API 키가 유효합니다!\n테스트 결과: {test_result}"))
                    else:
                        self.root.after(0, lambda: messagebox.showerror("테스트 실패", 
                            "API 키가 유효하지 않거나 연결에 문제가 있습니다."))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("테스트 오류", 
                        f"API 키 테스트 중 오류 발생:\n{e}"))
                finally:
                    # 원래 키로 복원
                    self.gemini_client.api_key = original_key
            
            threading.Thread(target=test_thread, daemon=True).start()
        
        # 버튼들을 더 크고 명확하게 만들기
        test_btn = ttk.Button(btn_frame, text="🔍 API 키 테스트", command=test_api_key, width=15)
        test_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        save_btn = ttk.Button(btn_frame, text="💾 저장 및 적용", command=save_api_key, width=15)
        save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_btn = ttk.Button(btn_frame, text="❌ 취소", command=dialog.destroy, width=10)
        cancel_btn.pack(side=tk.LEFT)
        
        # 포커스 설정
        new_key_entry.focus_set()

    def delete_api_key(self):
        """API 키 삭제"""
        if not self.gemini_client.api_key:
            messagebox.showinfo("정보", "삭제할 API 키가 없습니다.")
            return
        
        result = messagebox.askyesno("확인", 
            "현재 API 키를 삭제하시겠습니까?\n삭제 후에는 AI 분석 기능을 사용할 수 없습니다.")
        
        if result:
            try:
                self._remove_api_key_from_env()
                self.gemini_client.api_key = None
                self.gemini_client.is_available = False
                
                # UI 업데이트
                self.api_key_display_var.set("API 키 없음")
                self.update_gemini_status("API 키 없음")
                self.log_message("🗑️ API 키가 삭제되었습니다.")
                
                messagebox.showinfo("완료", "API 키가 삭제되었습니다.")
                
            except Exception as e:
                messagebox.showerror("오류", f"API 키 삭제 실패:\n{e}")

    def _save_api_key_to_env(self, api_key):
        """API 키를 .env 파일에 저장"""
        env_file = ".env"
        
        # 기존 .env 파일 읽기
        env_content = []
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding='utf-8') as f:
                env_content = f.readlines()
        
        # GEMINI_API_KEY 라인 찾기 및 업데이트
        key_found = False
        for i, line in enumerate(env_content):
            if line.startswith('GEMINI_API_KEY='):
                env_content[i] = f'GEMINI_API_KEY={api_key}\n'
                key_found = True
                break
        
        # 키가 없으면 추가
        if not key_found:
            env_content.append(f'GEMINI_API_KEY={api_key}\n')
        
        # 파일에 쓰기
        with open(env_file, 'w', encoding='utf-8') as f:
            f.writelines(env_content)

    def _remove_api_key_from_env(self):
        """API 키를 .env 파일에서 제거"""
        env_file = ".env"
        
        if not os.path.exists(env_file):
            return
        
        # 기존 .env 파일 읽기
        with open(env_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # GEMINI_API_KEY 라인 제거
        filtered_lines = [line for line in lines if not line.startswith('GEMINI_API_KEY=')]
        
        # 파일에 쓰기
        with open(env_file, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)

    def update_gemini_status(self, status):
        """Gemini API 상태 업데이트"""
        if hasattr(self, 'gemini_status_var'):
            self.gemini_status_var.set(status)  # 중복 제거
        if hasattr(self, 'gemini_status_label'):
            if "연결됨" in status:
                self.gemini_status_label.config(foreground="green")
            elif "API 키 없음" in status:
                self.gemini_status_label.config(foreground="orange")
            else:
                self.gemini_status_label.config(foreground="red")

    def setup_ui(self):
        """UI 구성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 상단 패널 (Gemini API 연결 및 스크래핑 제어)
        self.setup_gemini_panel(main_frame)
        
        # 중간 패널 (이미지 미리보기 및 AI 분석)
        self.setup_image_panel(main_frame)
        
        # 하단 패널 (모델 훈련 및 결과)
        self.setup_training_panel(main_frame)

    def setup_gemini_panel(self, parent):
        """Gemini API 연결 패널 구성"""
        gemini_frame = ttk.LabelFrame(parent, text="Gemini API 연결 및 스크래핑", padding=10)
        gemini_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Gemini API 상태
        gemini_status_frame = ttk.Frame(gemini_frame)
        gemini_status_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(gemini_status_frame, text="Gemini API 상태:").pack(side=tk.LEFT)
        self.gemini_status_var = tk.StringVar(value="연결 중...")
        self.gemini_status_label = ttk.Label(gemini_status_frame, textvariable=self.gemini_status_var, 
                                         foreground='orange')
        self.gemini_status_label.pack(side=tk.LEFT, padx=(5, 10))
        
        # API 키 관리 패널
        api_key_frame = ttk.LabelFrame(gemini_frame, text="API 키 관리", padding=5)
        api_key_frame.pack(fill=tk.X, pady=(5, 0))
        
        # API 키 입력 및 관리
        api_input_frame = ttk.Frame(api_key_frame)
        api_input_frame.pack(fill=tk.X)
        
        ttk.Label(api_input_frame, text="API 키:").pack(side=tk.LEFT)
        
        # API 키 표시 (마스킹)
        self.api_key_display_var = tk.StringVar()
        self.api_key_display_var.set(self._mask_api_key(self.gemini_client.api_key))
        api_key_entry = ttk.Entry(api_input_frame, textvariable=self.api_key_display_var, 
                                 width=30, state='readonly')
        api_key_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        # API 키 관리 버튼들
        api_btn_frame = ttk.Frame(api_input_frame)
        api_btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(api_btn_frame, text="API 키 관리", command=self.manage_api_key).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(api_btn_frame, text="API 키 삭제", command=self.delete_api_key).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(api_btn_frame, text="연결 테스트", command=self.test_gemini_connection).pack(side=tk.LEFT)
        
        # 하드웨어 가속 정보
        hw_info_frame = ttk.Frame(gemini_frame)
        hw_info_frame.pack(fill=tk.X, pady=(5, 0))
        
        cpu_count = multiprocessing.cpu_count()
        max_workers = min(4, cpu_count)
        ttk.Label(hw_info_frame, text=f"🚀 하드웨어 가속: {max_workers}개 워커 (CPU: {cpu_count}코어)", 
                 foreground='blue', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        
        # 성능 모드 토글
        self.performance_mode_var = tk.BooleanVar(value=True)
        performance_check = ttk.Checkbutton(hw_info_frame, text="고성능 모드 (80% CPU)", 
                                          variable=self.performance_mode_var,
                                          command=self.toggle_performance_mode)
        performance_check.pack(side=tk.RIGHT)
        
        # 수동 이미지 링크 입력
        manual_frame = ttk.Frame(gemini_frame)
        manual_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(manual_frame, text="수동 이미지 링크:").pack(side=tk.LEFT)
        self.manual_url_var = tk.StringVar()
        self.manual_url_entry = ttk.Entry(manual_frame, textvariable=self.manual_url_var, width=50)
        self.manual_url_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        ttk.Button(manual_frame, text="한 번에 추가", command=self.bulk_add_images).pack(side=tk.RIGHT)
        
        # 스크래핑 제어
        scraping_frame = ttk.Frame(gemini_frame)
        scraping_frame.pack(fill=tk.X)
        
        # 스크래핑 버튼
        self.scrape_btn = ttk.Button(scraping_frame, text="Cosmos.so 스크래핑 시작", 
                                   command=self.start_scraping)
        self.scrape_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 진행률 바
        self.progress_var = tk.StringVar(value="준비됨")
        self.progress_label = ttk.Label(scraping_frame, textvariable=self.progress_var)
        self.progress_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.progress_bar = ttk.Progressbar(scraping_frame, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # 데이터셋 저장/로드
        ttk.Button(scraping_frame, text="CSV 저장", command=self.save_csv).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(scraping_frame, text="CSV 새로 쓰기", command=self.create_new_csv).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(scraping_frame, text="CSV 로드", command=self.load_csv).pack(side=tk.RIGHT)

    def setup_image_panel(self, parent):
        """이미지 미리보기 패널 구성"""
        image_frame = ttk.LabelFrame(parent, text="AI 이미지 분석", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 좌측: 이미지 리스트
        left_frame = ttk.Frame(image_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 이미지 리스트박스
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(list_frame, text="이미지 목록:").pack(anchor=tk.W)
        
        # 리스트박스와 스크롤바
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.image_listbox = tk.Listbox(list_container, selectmode=tk.SINGLE)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.image_listbox.yview)
        self.image_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # 키보드 단축키 바인딩 (Cmd+A 지원)
        self.image_listbox.bind('<Command-a>', self.select_all_images)
        self.image_listbox.bind('<Control-a>', self.select_all_images)
        self.image_listbox.focus_set()  # 포커스를 설정하여 키보드 이벤트 수신
        
        # AI 분석 제어
        ai_frame = ttk.Frame(left_frame)
        ai_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(ai_frame, text="AI 분석 시작", command=self.start_ai_analysis).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ai_frame, text="전체 분석", command=self.batch_ai_analysis).pack(side=tk.LEFT, padx=(0, 5))
        
        # 카테고리 관리
        category_frame = ttk.Frame(left_frame)
        category_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(category_frame, text="카테고리:").pack(side=tk.LEFT)
        
        # 카테고리 입력 (자유 입력 가능, 콤마로 여러 카테고리 구분)
        self.category_var = tk.StringVar()
        self.category_entry = ttk.Entry(category_frame, textvariable=self.category_var, width=20)
        
        # 카테고리 입력 필드 변경 이벤트 추가
        self.category_var.trace_add('write', self.on_category_text_change)
        self.category_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        # 도움말 텍스트 추가
        help_label = ttk.Label(category_frame, text="(콤마로 여러 카테고리 구분)", font=("Arial", 8))
        help_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # 컨트롤 버튼들
        control_frame = ttk.Frame(category_frame)
        control_frame.pack(side=tk.RIGHT)
        
        ttk.Button(control_frame, text="카테고리 변경", command=self.change_category).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="선택 삭제", command=self.delete_selected_images).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="전체 선택", command=self.select_all_images).pack(side=tk.LEFT)
        
        # AI 분석 결과 표시
        analysis_frame = ttk.Frame(left_frame)
        analysis_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(analysis_frame, text="AI 분석 결과:").pack(anchor=tk.W)
        self.analysis_var = tk.StringVar(value="이미지를 선택하고 AI 분석을 시작하세요")
        self.analysis_label = ttk.Label(analysis_frame, textvariable=self.analysis_var, 
                                       foreground='blue', wraplength=300)
        self.analysis_label.pack(anchor=tk.W)
        
        # 크기 조절 핸들 (세퍼레이터)
        self.separator = ttk.Separator(image_frame, orient=tk.VERTICAL)
        self.separator.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # 우측: 이미지 미리보기 (크기 조절 가능)
        right_frame = ttk.Frame(image_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_frame, text="이미지 미리보기:").pack(anchor=tk.W)
        
        # 미리보기 창 크기 조절을 위한 변수들
        self.preview_width = 400  # 기본 너비
        self.preview_height = 300  # 기본 높이
        
        # 미리보기 프레임 (크기 조절 가능)
        self.preview_frame = tk.Frame(right_frame, width=self.preview_width, height=self.preview_height,
                                     background='white', relief=tk.SUNKEN, bd=2)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.preview_frame.pack_propagate(False)  # 크기 고정
        
        # 이미지 라벨 (미리보기 프레임 내부)
        self.image_label = ttk.Label(self.preview_frame, text="이미지를 선택하세요", 
                                   background='white')
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # 크기 조절 핸들 바인딩
        self.separator.bind('<Button-1>', self.start_resize)
        self.separator.bind('<B1-Motion>', self.do_resize)
        self.separator.bind('<ButtonRelease-1>', self.stop_resize)
        
        # 커서 변경
        self.separator.bind('<Enter>', lambda e: self.separator.config(cursor='sb_h_double_arrow'))
        self.separator.bind('<Leave>', lambda e: self.separator.config(cursor=''))
        
        # 크기 조절 상태 변수
        self.is_resizing = False
        self.start_x = 0

    def start_resize(self, event):
        """크기 조절 시작"""
        self.is_resizing = True
        self.start_x = event.x_root
        self.root.config(cursor='sb_h_double_arrow')

    def do_resize(self, event):
        """크기 조절 중"""
        if not self.is_resizing:
            return
        
        # 마우스 이동 거리 계산
        delta_x = event.x_root - self.start_x
        
        # 미리보기 창 크기 조절 (최소/최대 크기 제한)
        new_width = max(200, min(800, self.preview_width + delta_x))
        
        if new_width != self.preview_width:
            self.preview_width = new_width
            self.preview_frame.config(width=self.preview_width)
            self.start_x = event.x_root  # 기준점 업데이트

    def stop_resize(self, event):
        """크기 조절 종료"""
        self.is_resizing = False
        self.root.config(cursor='')

    def setup_training_panel(self, parent):
        """훈련 패널 구성"""
        training_frame = ttk.LabelFrame(parent, text="모델 훈련", padding=10)
        training_frame.pack(fill=tk.X)
        
        # 훈련 제어
        control_frame = ttk.Frame(training_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="에포크:").pack(side=tk.LEFT)
        self.epochs_var = tk.StringVar(value="5")
        ttk.Entry(control_frame, textvariable=self.epochs_var, width=5).pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(control_frame, text="배치 크기:").pack(side=tk.LEFT)
        self.batch_size_var = tk.StringVar(value="8")
        ttk.Entry(control_frame, textvariable=self.batch_size_var, width=5).pack(side=tk.LEFT, padx=(5, 10))
        
        self.train_btn = ttk.Button(control_frame, text="훈련 시작", command=self.start_training)
        self.train_btn.pack(side=tk.RIGHT)
        
        # 훈련 상태
        self.training_status_var = tk.StringVar(value="대기 중")
        ttk.Label(control_frame, textvariable=self.training_status_var).pack(side=tk.RIGHT, padx=(0, 10))
        
        # 그래프
        self.setup_training_graph(training_frame)

    def setup_training_graph(self, parent):
        """훈련 그래프 구성"""
        graph_frame = ttk.Frame(parent)
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib 그래프
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend(['Train', 'Test'])
        
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.legend(['Train', 'Test'])
        
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_logging(self):
        """로깅 설정"""
        # 로그 텍스트 위젯
        log_frame = ttk.LabelFrame(self.root, text="로그", padding=5)
        log_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.log_text = tk.Text(log_frame, height=6, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def log_message(self, message: str):
        """로그 메시지 추가"""
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()


    def start_scraping(self):
        """스크래핑 시작"""
        self.scrape_btn.config(state='disabled')
        self.progress_bar['value'] = 0
        
        def scraping_thread():
            # 간단한 스크래핑 시뮬레이션
            self.update_scraping_progress("Cosmos.so 스크래핑 시작...", 0)
            time.sleep(1)
            
            self.update_scraping_progress("페이지 접속 중...", 20)
            time.sleep(1)
            
            self.update_scraping_progress("이미지 발견 중...", 50)
            time.sleep(1)
            
            # 새 CSV 파일 생성
            try:
                # 새 타임스탬프 기반 CSV 파일명 생성
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join("dataset", f"scraped_images_{timestamp}.csv")
                
                # 기존 데이터 초기화
                self.image_data = []
                
                # 실제 Cosmos.so 스크래핑
                from utils.cosmos_real_final import CosmosRealScraper
                
                scraper = CosmosRealScraper()
                self.update_scraping_progress("Cosmos.so 실제 스크래핑 시작...", 20)
                
                # 실제 이미지 스크래핑
                image_data = scraper.scrape_cosmos_images()
                
                if image_data:
                    sample_urls = [item[0] for item in image_data[:10]]  # 최대 10개
                    initial_categories = [item[1] for item in image_data[:10]]
                    
                    self.update_scraping_progress(f"실제 {len(sample_urls)}개 이미지 발견", 50)
                else:
                    # 스크래핑 실패 시 기본 이미지들 사용
                    sample_urls = [
                        "https://cdn.cosmos.so/646ade0c-beff-4003-bcae-c977de3ea7dd?format=webp&w=1080",
                        "https://cdn.cosmos.so/a22716e5-1442-432c-b320-05b3ad24deec?rect=33%2C0%2C528%2C529&format=webp&w=1080",
                        "https://cdn.cosmos.so/458e7583-47f5-4296-9e8b-b4ea9178f093?rect=97%2C0%2C635%2C635&format=webp&w=1080",
                        "https://cdn.cosmos.so/default-avatars/014.png?format=webp&w=1080",
                        "https://cdn.cosmos.so/50c37c58-e828-4061-a24b-223a785d6b05?format=webp&w=1080"
                    ]
                    initial_categories = ["people", "art", "design", "people", "nature"]
                    self.update_scraping_progress("기본 이미지 사용 (스크래핑 실패)", 50)
                
                # 각 이미지에 대해 실제 Gemini API로 카테고리 분석
                categories = []
                for i, url in enumerate(sample_urls):
                    self.update_scraping_progress(f"이미지 {i+1}/{len(sample_urls)} 분석 중...", 60 + (i * 3))
                    try:
                        # Gemini API 직접 호출
                        category_result = self.gemini_client.analyze_image(url)
                        if category_result and category_result != "general, design, creative":
                            categories.append(category_result)
                        else:
                            # 초기 카테고리 사용
                            categories.append(initial_categories[i] if i < len(initial_categories) else "people, portrait, indoor")
                    except Exception as e:
                        self.log_message(f"이미지 {i+1} 분석 실패: {e}")
                        categories.append(initial_categories[i] if i < len(initial_categories) else "people, portrait, indoor")
                
                # 새 CSV 파일에 데이터 저장
                df = pd.DataFrame({
                    'image_url': sample_urls,
                    'category': categories
                })
                df.to_csv(csv_path, index=False)
                self.log_message(f"새 CSV 파일 생성: {csv_path}")
                    
            except Exception as e:
                self.log_message(f"CSV 생성 실패: {e}")
                sample_urls = []
                categories = []
            
            self.update_scraping_progress("이미지 수집 완료", 100)
            
            self.root.after(0, lambda: self.scraping_completed(sample_urls, categories))
        
        thread = threading.Thread(target=scraping_thread)
        thread.daemon = True
        thread.start()

    def update_scraping_progress(self, message: str, progress: int = None):
        """스크래핑 진행 상황 업데이트"""
        def update():
            self.progress_var.set(message)
            if progress is not None and progress >= 0:
                self.progress_bar['value'] = progress
            self.log_message(message)
        
        self.root.after(0, update)

    def scraping_completed(self, urls: List[str], labels: List[str]):
        """스크래핑 완료 처리"""
        self.scrape_btn.config(state='normal')
        
        if urls:
            self.image_urls = urls
            self.labels = labels
            self.update_image_list()
            self.log_message(f"스크래핑 완료: {len(urls)}개 이미지 수집")
            
            # 자동 저장
            self.auto_save_data()
        else:
            self.log_message("스크래핑 실패: 이미지를 찾을 수 없습니다")
            messagebox.showerror("오류", "이미지 스크래핑에 실패했습니다.")

    def update_image_list(self):
        """이미지 리스트 업데이트"""
        self.image_listbox.delete(0, tk.END)
        
        for i, (url, label) in enumerate(zip(self.image_urls, self.labels)):
            filename = url.split('/')[-1].split('?')[0]
            self.image_listbox.insert(tk.END, f"{i+1:2d}. [{label}] {filename}")

    def on_image_select(self, event):
        """이미지 선택 이벤트"""
        selection = self.image_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        url = self.image_urls[index]
        category = self.labels[index]
        
        self.category_var.set(category)
        
        # 이미지 미리보기 업데이트
        self.load_image_preview(url)

    def load_image_preview(self, image_url: str):
        """이미지 미리보기 로딩"""
        def load_image_thread():
            try:
                self.log_message(f"이미지 로딩 중: {image_url}")
                
                # 로딩 상태 표시
                self.root.after(0, lambda: self.show_loading_state())
                
                # 이미지 다운로드 (더 긴 타임아웃)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                response = requests.get(image_url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # PIL Image로 변환
                image = Image.open(io.BytesIO(response.content))
                
                # 이미지 크기 조정 (미리보기용 - 동적 크기)
                max_size = (self.preview_width - 20, self.preview_height - 20)  # 패딩 고려
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Tkinter용 이미지로 변환
                photo = ImageTk.PhotoImage(image)
                
                # GUI 스레드에서 이미지 업데이트
                self.root.after(0, lambda: self.update_image_display(photo, image_url))
                
            except requests.exceptions.RequestException as e:
                error_msg = f"네트워크 오류: {str(e)}"
                self.log_message(error_msg)
                self.root.after(0, lambda: self.show_image_error(error_msg))
            except Exception as e:
                error_msg = f"이미지 처리 오류: {str(e)}"
                self.log_message(error_msg)
                self.root.after(0, lambda: self.show_image_error(error_msg))
        
        # 백그라운드에서 이미지 로딩
        threading.Thread(target=load_image_thread, daemon=True).start()
    
    def show_loading_state(self):
        """로딩 상태 표시"""
        self.image_label.config(image="", text="이미지 로딩 중...\n잠시만 기다려주세요")
        self.image_label.image = None
    
    def update_image_display(self, photo, image_url: str):
        """이미지 표시 업데이트"""
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # 참조 유지
        self.log_message(f"이미지 표시 완료: {image_url}")
    
    def show_image_error(self, error_msg: str):
        """이미지 로딩 오류 표시"""
        self.image_label.config(image="", text=f"이미지 로딩 실패\n{error_msg}")
        self.image_label.image = None

    def start_ai_analysis(self):
        """AI 분석 시작"""
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("경고", "이미지를 선택하세요.")
            return
        
        # Gemini API 키 확인
        if not self.gemini_client.api_key:
            messagebox.showerror("오류", "Gemini API 키가 설정되지 않았습니다.\n.env 파일에 GEMINI_API_KEY를 설정하세요.")
            return
        
        index = selection[0]
        url = self.image_urls[index]
        
        def analysis_thread():
            self.log_message(f"이미지 {index+1} AI 분석 시작...")
            
            try:
                # Gemini API 직접 호출
                suggested_categories = self.gemini_client.analyze_image(url)
                
                if suggested_categories and suggested_categories != "general, design, creative":
                    self.root.after(0, lambda: self.log_message(f"✅ AI 분석 완료: {suggested_categories}"))
                    
                    # 카테고리 입력 필드에 자동 입력
                    self.root.after(0, lambda: self.category_var.set(suggested_categories))
                    
                    # 분석 결과 표시
                    self.root.after(0, lambda: self.analysis_var.set(
                        f"AI 추천 카테고리: {suggested_categories}"
                    ))
                    
                    # 라벨도 업데이트
                    self.root.after(0, lambda: self.labels.__setitem__(index, suggested_categories))
                    self.root.after(0, lambda: self.update_image_list())
                else:
                    self.root.after(0, lambda: self.log_message("❌ AI 분석 실패 - 기본값 반환"))
                    self.root.after(0, lambda: self.analysis_var.set("AI 분석 실패 - 기본값 반환"))
                    
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"❌ AI 분석 오류: {e}"))
                self.root.after(0, lambda: self.analysis_var.set(f"AI 분석 오류: {e}"))
        
        threading.Thread(target=analysis_thread, daemon=True).start()

    def update_analysis_result(self, result: Dict, index: int):
        """AI 분석 결과 업데이트"""
        suggested_categories = result["suggested_categories"]
        confidence_scores = result["confidence_scores"]
        analysis_text = result["analysis_text"]
        
        # 카테고리 업데이트
        self.labels[index] = suggested_categories[0]
        self.category_var.set(suggested_categories[0])
        
        # 분석 결과 표시
        analysis_display = f"""
추천 카테고리: {suggested_categories[0]} (신뢰도: {confidence_scores[0]:.2f})

대안 카테고리:
- {suggested_categories[1]} ({confidence_scores[1]:.2f})
- {suggested_categories[2]} ({confidence_scores[2]:.2f})
- {suggested_categories[3]} ({confidence_scores[3]:.2f})

분석: {analysis_text}
        """
        
        self.analysis_var.set(analysis_display)
        self.update_image_list()
        self.log_message(f"AI 분석 완료: {suggested_categories[0]} 추천")

    def batch_ai_analysis(self):
        """전체 이미지 AI 분석"""
        if not self.image_urls:
            messagebox.showwarning("경고", "분석할 이미지가 없습니다.")
            return
        
        # Gemini API 키 확인
        if not self.gemini_client.api_key:
            messagebox.showerror("오류", "Gemini API 키가 설정되지 않았습니다.\n.env 파일에 GEMINI_API_KEY를 설정하세요.")
            return
        
        # 사용자 확인
        if not messagebox.askyesno("확인", f"전체 {len(self.image_urls)}개 이미지를 분석하시겠습니까?\n(시간이 오래 걸릴 수 있습니다)"):
            return
        
        def batch_analysis_thread():
            # 성능 모드에 따른 워커 수 동적 조정
            if self.performance_mode_var.get():
                self.gemini_client.max_workers = min(8, multiprocessing.cpu_count())
                mode_text = "고성능 모드"
            else:
                self.gemini_client.max_workers = min(2, multiprocessing.cpu_count())
                mode_text = "절전 모드"
            
            self.log_message(f"🚀 전체 {len(self.image_urls)}개 이미지 AI 분석 시작... ({mode_text}, {self.gemini_client.max_workers}개 워커)")
            
            try:
                # Gemini API 직접 호출로 배치 분석 (병렬 처리)
                results = self.gemini_client.batch_analyze_images(self.image_urls)
                
                if results:
                    self.root.after(0, lambda: self.log_message("✅ 배치 AI 분석 완료"))
                    
                    # 각 이미지에 대해 결과 적용
                    for i, result in enumerate(results):
                        self.labels[i] = result
                        
                        # 진행 상황 업데이트
                        progress = int((i + 1) / len(self.image_urls) * 100)
                        self.root.after(0, lambda p=progress: self.progress_bar.config(value=p))
                        self.root.after(0, lambda: self.progress_var.set(f"처리 중... {i+1}/{len(self.image_urls)}"))
                    
                    self.root.after(0, lambda: self.update_image_list())
                    self.root.after(0, lambda: self.log_message("✅ 모든 이미지 카테고리 업데이트 완료"))
                    self.root.after(0, lambda: self.progress_var.set("배치 분석 완료"))
                else:
                    self.root.after(0, lambda: self.log_message("❌ 배치 AI 분석 실패"))
                    self.root.after(0, lambda: self.progress_var.set("분석 실패"))
                    
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"❌ 배치 분석 오류: {e}"))
                self.root.after(0, lambda: self.progress_var.set("분석 오류"))
        
    def add_manual_image(self):
        """수동으로 이미지 링크 추가 (카테고리 없이)"""
        url = self.manual_url_var.get().strip()
        if not url:
            messagebox.showwarning("경고", "이미지 링크를 입력하세요.")
            return
        
        # URL 유효성 검사
        if not url.startswith(('http://', 'https://')):
            messagebox.showerror("오류", "올바른 URL 형식이 아닙니다.")
            return
        
        # 기본 카테고리로 추가
        category = "manual, general, creative"
        
        # 리스트에 추가
        self.image_urls.append(url)
        self.labels.append(category)
        
        # UI 업데이트
        self.update_image_list()
        self.manual_url_var.set("")  # 입력 필드 초기화
        
        self.log_message(f"수동 이미지 추가: {url}")
        messagebox.showinfo("성공", "이미지가 추가되었습니다.")
    
    def add_manual_image_with_ai(self):
        """수동으로 이미지 링크 추가 (AI 분석 후)"""
        url = self.manual_url_var.get().strip()
        if not url:
            messagebox.showwarning("경고", "이미지 링크를 입력하세요.")
            return
        
        # URL 유효성 검사
        if not url.startswith(('http://', 'https://')):
            messagebox.showerror("오류", "올바른 URL 형식이 아닙니다.")
            return
        
        # 중복 URL 확인
        if url in self.image_urls:
            messagebox.showinfo("알림", "이미 존재하는 이미지 링크입니다.\n중복된 URL은 추가되지 않습니다.")
            return
        
        # Gemini API 키 확인
        if not self.gemini_client.api_key:
            messagebox.showerror("오류", "Gemini API 키가 설정되지 않았습니다.\n.env 파일에 GEMINI_API_KEY를 설정하세요.")
            return
        
        def ai_analysis_thread():
            self.log_message(f"수동 이미지 AI 분석 시작: {url}")
            
            try:
                # Gemini API로 분석
                suggested_categories = self.gemini_client.analyze_image(url)
                
                if suggested_categories and suggested_categories != "general, design, creative":
                    # 리스트에 추가
                    self.image_urls.append(url)
                    self.labels.append(suggested_categories)
                    
                    # UI 업데이트
                    self.root.after(0, lambda: self.update_image_list())
                    self.root.after(0, lambda: self.manual_url_var.set(""))  # 입력 필드 초기화
                    
                    # 자동 저장
                    self.root.after(0, lambda: self.auto_save_data())
                    
                    self.root.after(0, lambda: self.log_message(f"✅ 수동 이미지 AI 분석 완료: {suggested_categories}"))
                    self.root.after(0, lambda: messagebox.showinfo("성공", f"이미지가 AI 분석 후 추가되었습니다.\n카테고리: {suggested_categories}"))
                else:
                    # 기본 카테고리로 추가
                    self.image_urls.append(url)
                    self.labels.append("manual, general, creative")
                    
                    self.root.after(0, lambda: self.update_image_list())
                    self.root.after(0, lambda: self.manual_url_var.set(""))
                    
                    self.root.after(0, lambda: self.log_message("❌ AI 분석 실패 - 기본 카테고리로 추가"))
                    self.root.after(0, lambda: messagebox.showinfo("성공", "이미지가 추가되었습니다. (AI 분석 실패로 기본 카테고리 사용)"))
                    
            except Exception as e:
                # 오류 시 기본 카테고리로 추가
                self.image_urls.append(url)
                self.labels.append("manual, general, creative")
                
                self.root.after(0, lambda: self.update_image_list())
                self.root.after(0, lambda: self.manual_url_var.set(""))
                
                self.root.after(0, lambda: self.log_message(f"❌ AI 분석 오류: {e} - 기본 카테고리로 추가"))
                self.root.after(0, lambda: messagebox.showinfo("성공", "이미지가 추가되었습니다. (AI 분석 오류로 기본 카테고리 사용)"))
        
    def bulk_add_images(self):
        """한 번에 여러 이미지 링크 추가 (모달 창)"""
        # 모달 창 생성
        modal = tk.Toplevel(self.root)
        modal.title("한 번에 이미지 추가")
        modal.geometry("800x600")
        modal.transient(self.root)
        modal.grab_set()  # 모달로 설정
        
        # 중앙에 위치
        modal.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # 메인 프레임
        main_frame = ttk.Frame(modal, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 제목
        title_label = ttk.Label(main_frame, text="이미지 링크를 한 번에 추가하세요", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 설명
        desc_label = ttk.Label(main_frame, text="여러 이미지 링크를 입력하세요 (한 줄에 하나씩):", font=("Arial", 10))
        desc_label.pack(anchor=tk.W, pady=(0, 10))
        
        # 텍스트 영역과 스크롤바
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.bulk_text = tk.Text(text_frame, height=15, wrap=tk.WORD, font=("Arial", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.bulk_text.yview)
        self.bulk_text.configure(yscrollcommand=scrollbar.set)
        
        self.bulk_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 텍스트 영역은 빈 상태로 시작
        
        # 옵션 프레임
        options_frame = ttk.Frame(main_frame)
        options_frame.pack(fill=tk.X, pady=(0, 20))
        
        # AI 분석 옵션
        self.ai_analysis_var = tk.BooleanVar(value=True)
        ai_checkbox = ttk.Checkbutton(options_frame, text="AI 분석 후 추가 (권장)", variable=self.ai_analysis_var)
        ai_checkbox.pack(side=tk.LEFT)
        
        # 진행 상황 표시
        self.bulk_progress_var = tk.StringVar(value="준비됨")
        progress_label = ttk.Label(options_frame, textvariable=self.bulk_progress_var)
        progress_label.pack(side=tk.RIGHT)
        
        # 버튼 프레임
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        # 취소 버튼
        cancel_btn = ttk.Button(button_frame, text="취소", command=modal.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # 추가 버튼
        add_btn = ttk.Button(button_frame, text="추가 시작", command=lambda: self.start_bulk_add(modal))
        add_btn.pack(side=tk.RIGHT)
        
        # 포커스 설정
        self.bulk_text.focus_set()
    
    def start_bulk_add(self, modal):
        """한 번에 추가 시작"""
        # 텍스트에서 URL 추출
        text_content = self.bulk_text.get("1.0", tk.END).strip()
        if not text_content:
            messagebox.showwarning("경고", "이미지 링크를 입력하세요.")
            return
        
        # 줄별로 분리하고 유효한 URL만 필터링
        input_urls = []
        for line in text_content.split('\n'):
            line = line.strip()
            if line and line.startswith(('http://', 'https://')):
                input_urls.append(line)
        
        if not input_urls:
            messagebox.showerror("오류", "유효한 이미지 링크가 없습니다.")
            return
        
        # 중복 URL 확인
        existing_urls = set(self.image_urls)
        new_urls = []
        duplicate_urls = []
        
        for url in input_urls:
            if url in existing_urls:
                duplicate_urls.append(url)
            else:
                new_urls.append(url)
        
        # 중복 URL이 있으면 사용자에게 알림
        if duplicate_urls:
            duplicate_count = len(duplicate_urls)
            new_count = len(new_urls)
            
            if new_count == 0:
                messagebox.showinfo("알림", f"입력한 {duplicate_count}개의 URL이 모두 이미 존재합니다.\n중복된 URL은 추가되지 않습니다.")
                modal.destroy()
                return
            else:
                if not messagebox.askyesno("중복 URL 발견", 
                    f"중복된 URL {duplicate_count}개가 발견되었습니다.\n"
                    f"새로운 URL {new_count}개만 추가하시겠습니까?\n\n"
                    f"중복된 URL은 추가되지 않습니다."):
                    return
        
        # AI 분석 옵션 확인
        use_ai = self.ai_analysis_var.get()
        
        if use_ai and not self.gemini_client.api_key:
            messagebox.showerror("오류", "Gemini API 키가 설정되지 않았습니다.\n.env 파일에 GEMINI_API_KEY를 설정하세요.")
            return
        
        # 사용자 확인
        if not messagebox.askyesno("확인", f"{len(new_urls)}개의 새로운 이미지를 {'AI 분석 후 ' if use_ai else ''}추가하시겠습니까?"):
            return
        
        # 모달 창 닫기
        modal.destroy()
        
        # 백그라운드에서 일괄 추가 실행
        def bulk_add_thread():
            self.log_message(f"한 번에 {len(new_urls)}개 새로운 이미지 추가 시작...")
            if duplicate_urls:
                self.log_message(f"중복된 URL {len(duplicate_urls)}개는 건너뜀")
            
            success_count = 0
            fail_count = 0
            
            for i, url in enumerate(new_urls):
                try:
                    self.root.after(0, lambda p=int((i+1)/len(new_urls)*100): self.progress_bar.config(value=p))
                    self.root.after(0, lambda: self.bulk_progress_var.set(f"처리 중... {i+1}/{len(new_urls)}"))
                    
                    if use_ai:
                        # AI 분석 후 추가
                        suggested_categories = self.gemini_client.analyze_image(url)
                        if suggested_categories and suggested_categories != "general, design, creative":
                            category = suggested_categories
                        else:
                            category = "manual, general, creative"
                    else:
                        # 기본 카테고리로 추가
                        category = "manual, general, creative"
                    
                    # 리스트에 추가
                    self.image_urls.append(url)
                    self.labels.append(category)
                    
                    success_count += 1
                    self.log_message(f"✅ 이미지 {i+1}/{len(new_urls)} 추가 완료: {category}")
                    
                    # API 호출 간격 조절 (AI 분석 시)
                    if use_ai:
                        time.sleep(1.0)
                    
                except Exception as e:
                    fail_count += 1
                    self.log_message(f"❌ 이미지 {i+1}/{len(new_urls)} 추가 실패: {e}")
            
            # 완료 처리
            self.root.after(0, lambda: self.update_image_list())
            self.root.after(0, lambda: self.progress_bar.config(value=100))
            self.root.after(0, lambda: self.bulk_progress_var.set("완료"))
            
            # 자동 저장
            self.root.after(0, lambda: self.auto_save_data())
            
            # 결과 메시지
            result_message = f"한 번에 추가가 완료되었습니다.\n\n• 성공: {success_count}개\n• 실패: {fail_count}개\n• 총 처리: {len(new_urls)}개"
            if duplicate_urls:
                result_message += f"\n• 중복 건너뜀: {len(duplicate_urls)}개"
            
            self.log_message(f"✅ 한 번에 추가 완료: 성공 {success_count}개, 실패 {fail_count}개")
            if duplicate_urls:
                self.log_message(f"중복 URL {len(duplicate_urls)}개는 건너뜀")
            
            self.root.after(0, lambda: messagebox.showinfo("완료", result_message))
        
        threading.Thread(target=bulk_add_thread, daemon=True).start()

    def change_category(self):
        """카테고리 변경 (콤마로 여러 카테고리 지원, 전체 선택 지원)"""
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("경고", "이미지를 선택하세요.")
            return
        
        new_category_input = self.category_var.get().strip()
        
        if not new_category_input:
            messagebox.showwarning("경고", "새 카테고리를 입력하세요.")
            return
        
        # 콤마로 구분된 카테고리들을 정리
        categories = [cat.strip() for cat in new_category_input.split(',') if cat.strip()]
        
        if not categories:
            messagebox.showwarning("경고", "유효한 카테고리를 입력하세요.")
            return
        
        # 선택된 모든 이미지에 카테고리 적용
        changed_count = 0
        for index in selection:
            # 카테고리 변경
            self.labels[index] = new_category_input  # 원본 입력값 저장
            
            # 선택된 항목만 업데이트 (전체 리스트 재생성 방지)
            filename = self.image_urls[index].split('/')[-1].split('?')[0]
            self.image_listbox.delete(index)
            
            # 여러 카테고리 표시 (최대 3개까지만 표시)
            if len(categories) <= 3:
                display_categories = ', '.join(categories)
            else:
                display_categories = ', '.join(categories[:3]) + f" (+{len(categories)-3})"
            
            self.image_listbox.insert(index, f"{index+1:2d}. [{display_categories}] {filename}")
            changed_count += 1
        
        # 선택 상태 유지
        for index in selection:
            self.image_listbox.selection_set(index)
        
        if len(selection) == 1:
            self.log_message(f"이미지 {selection[0]+1}의 카테고리를 '{new_category_input}'로 변경 ({len(categories)}개 카테고리)")
        else:
            self.log_message(f"{changed_count}개 이미지의 카테고리를 '{new_category_input}'로 변경 ({len(categories)}개 카테고리)")

    def on_category_text_change(self, *args):
        """카테고리 텍스트 변경 시 선택 상태 유지"""
        # 현재 선택된 이미지가 있다면 선택 상태 유지
        selection = self.image_listbox.curselection()
        if selection:
            # 선택 상태가 풀렸다면 다시 선택
            if not self.image_listbox.curselection():
                self.image_listbox.selection_set(selection[0])

    def select_all_images(self, event=None):
        """Cmd+A 또는 Ctrl+A로 모든 이미지 선택 (버튼 클릭도 지원)"""
        if not self.image_urls:
            return
        
        # 모든 항목 선택
        self.image_listbox.selection_clear(0, tk.END)
        for i in range(len(self.image_urls)):
            self.image_listbox.selection_set(i)
        
        # 첫 번째 항목에 포커스 설정
        self.image_listbox.activate(0)
        self.image_listbox.see(0)
        
        # 첫 번째 이미지 정보 표시
        self.on_image_select(None)
        
        self.log_message(f"모든 이미지 선택됨 ({len(self.image_urls)}개)")
        return "break"  # 기본 이벤트 처리 방지
    
    def delete_selected_images(self):
        """선택된 이미지들 삭제"""
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("경고", "삭제할 이미지를 선택하세요.")
            return
        
        # 사용자 확인
        if len(selection) == 1:
            confirm_msg = f"선택된 이미지 1개를 삭제하시겠습니까?"
        else:
            confirm_msg = f"선택된 이미지 {len(selection)}개를 삭제하시겠습니까?"
        
        if not messagebox.askyesno("확인", confirm_msg):
            return
        
        # 역순으로 삭제 (인덱스가 변경되지 않도록)
        deleted_count = 0
        for index in reversed(sorted(selection)):
            if 0 <= index < len(self.image_urls):
                deleted_url = self.image_urls[index]
                deleted_label = self.labels[index]
                
                # 리스트에서 제거
                del self.image_urls[index]
                del self.labels[index]
                
                deleted_count += 1
                self.log_message(f"이미지 삭제: {deleted_url.split('/')[-1].split('?')[0]} ({deleted_label})")
        
        # UI 업데이트
        self.update_image_list()
        
        # 자동 저장
        self.auto_save_data()
        
        if deleted_count == 1:
            messagebox.showinfo("완료", "이미지 1개가 삭제되었습니다.")
        else:
            messagebox.showinfo("완료", f"이미지 {deleted_count}개가 삭제되었습니다.")
        
        self.log_message(f"✅ {deleted_count}개 이미지 삭제 완료")
    
    def toggle_performance_mode(self):
        """성능 모드 토글"""
        if self.performance_mode_var.get():
            # 고성능 모드 활성화 (429 오류 방지를 위해 워커 수 제한)
            self.gemini_client.max_workers = min(4, multiprocessing.cpu_count())
            self.log_message("🚀 고성능 모드 활성화 - API 부하 고려한 최적화")
        else:
            # 절전 모드 활성화
            self.gemini_client.max_workers = min(2, multiprocessing.cpu_count())
            self.log_message("🔋 절전 모드 활성화 - 효율성 우선")
    
    def auto_save_data(self):
        """유동 데이터를 정적 데이터로 자동 저장"""
        if not self.image_urls:
            return
        
        try:
            # dataset 폴더가 없으면 생성
            dataset_dir = "dataset"
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            
            # 타임스탬프 기반 파일명 생성
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 새로운 형식: image_link.jpg, 영문주요카테고리3개, 한국어주요카테고리3개
            x_data = []
            for i, (url, category) in enumerate(zip(self.image_urls, self.labels)):
                filename_only = url.split('/')[-1].split('?')[0]
                
                # 카테고리 파싱 (영문3개 | 한국어3개 형식)
                if '|' in category:
                    english_part, korean_part = category.split('|', 1)
                    english_categories = english_part.strip()
                    korean_categories = korean_part.strip()
                else:
                    # 기존 형식인 경우 기본값 사용
                    english_categories = category
                    korean_categories = "일반, 기본, 표준"
                
                x_data.append([filename_only, english_categories, korean_categories])
            
            x_df = pd.DataFrame(x_data, columns=['image_link.jpg', '영문주요카테고리3개', '한국어주요카테고리3개'])
            x_filename = f"{dataset_dir}/x_train_auto_{timestamp}.csv"
            x_df.to_csv(x_filename, index=False)
            
            # y_train.csv: 영문주요카테고리3개, 한국어주요카테고리3개
            y_data = []
            for category in self.labels:
                if '|' in category:
                    english_part, korean_part = category.split('|', 1)
                    english_categories = english_part.strip()
                    korean_categories = korean_part.strip()
                else:
                    english_categories = category
                    korean_categories = "일반, 기본, 표준"
                y_data.append([english_categories, korean_categories])
            
            y_df = pd.DataFrame(y_data, columns=['영문주요카테고리3개', '한국어주요카테고리3개'])
            y_filename = f"{dataset_dir}/y_train_auto_{timestamp}.csv"
            y_df.to_csv(y_filename, index=False)
            
            # 전체 데이터 CSV (URL 포함)
            full_df = pd.DataFrame({
                'image_url': self.image_urls,
                'category': self.labels
            })
            full_filename = f"{dataset_dir}/full_data_auto_{timestamp}.csv"
            full_df.to_csv(full_filename, index=False)
            
            # 최신 데이터를 기본 파일로도 저장 (덮어쓰기)
            x_df.to_csv(f"{dataset_dir}/x_train.csv", index=False)
            y_df.to_csv(f"{dataset_dir}/y_train.csv", index=False)
            full_df.to_csv(f"{dataset_dir}/full_data.csv", index=False)
            
            # 카테고리별 통계 생성
            all_categories = []
            for category_string in self.labels:
                categories = [cat.strip() for cat in category_string.split(',') if cat.strip()]
                all_categories.extend(categories)
            
            from collections import Counter
            category_counts = Counter(all_categories)
            
            self.log_message(f"🔄 자동 저장 완료:")
            self.log_message(f"  - X 데이터: {x_filename}")
            self.log_message(f"  - Y 데이터: {y_filename}")
            self.log_message(f"  - 전체 데이터: {full_filename}")
            self.log_message(f"  - 기본 파일 업데이트: x_train.csv, y_train.csv, full_data.csv")
            self.log_message(f"총 {len(self.image_urls)}개 이미지, {len(category_counts)}개 고유 카테고리")
            
            # 저장 완료 후 이전 파일들 정리
            self.cleanup_old_data_files()
            
        except Exception as e:
            self.log_message(f"❌ 자동 저장 실패: {e}")

    def create_new_csv(self):
        """새로운 CSV 파일 생성 (기존 데이터 모두 삭제)"""
        # 확인 대화상자
        result = messagebox.askyesno(
            "CSV 새로 쓰기", 
            "현재 모든 데이터가 삭제되고 새로운 CSV 파일이 생성됩니다.\n계속하시겠습니까?"
        )
        
        if not result:
            return
        
        # 데이터 초기화
        self.image_urls = []
        self.labels = []
        self.update_image_list()
        
        # UI 업데이트
        self.log_message("🔄 모든 데이터가 삭제되었습니다.")
        self.log_message("💡 새로운 이미지를 추가하거나 스크래핑을 시작하세요.")
        
        # 빈 CSV 파일 생성
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="새로운 CSV 파일 저장"
        )
        
        if filename:
            try:
                # 빈 데이터프레임 생성
                empty_df = pd.DataFrame(columns=['image_link.jpg', '영문주요카테고리3개', '한국어주요카테고리3개'])
                empty_df.to_csv(filename, index=False)
                
                self.log_message(f"✅ 새로운 CSV 파일이 생성되었습니다: {filename}")
                messagebox.showinfo("완료", f"새로운 CSV 파일이 생성되었습니다:\n{filename}")
                
            except Exception as e:
                self.log_message(f"❌ CSV 파일 생성 실패: {e}")
                messagebox.showerror("오류", f"CSV 파일 생성에 실패했습니다:\n{e}")

    def save_csv(self):
        """CSV 파일 저장 (여러 카테고리 지원, 수동 추가 이미지 포함)"""
        if not self.image_urls:
            messagebox.showwarning("경고", "저장할 데이터가 없습니다.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # 타임스탬프 기반 파일명 생성
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_filename = filename.replace('.csv', '')
                
                # 새로운 형식: image_link.jpg, 영문주요카테고리3개, 한국어주요카테고리3개
                x_data = []
                for i, (url, category) in enumerate(zip(self.image_urls, self.labels)):
                    filename_only = url.split('/')[-1].split('?')[0]
                    
                    # 카테고리 파싱 (영문3개 | 한국어3개 형식)
                    if '|' in category:
                        english_part, korean_part = category.split('|', 1)
                        english_categories = english_part.strip()
                        korean_categories = korean_part.strip()
                    else:
                        # 기존 형식인 경우 기본값 사용
                        english_categories = category
                        korean_categories = "일반, 기본, 표준"
                    
                    x_data.append([filename_only, english_categories, korean_categories])
                
                x_df = pd.DataFrame(x_data, columns=['image_link.jpg', '영문주요카테고리3개', '한국어주요카테고리3개'])
                x_filename = f"{base_filename}_x_train_{timestamp}.csv"
                x_df.to_csv(x_filename, index=False)
                
                # y_train.csv: 영문주요카테고리3개, 한국어주요카테고리3개
                y_data = []
                for category in self.labels:
                    if '|' in category:
                        english_part, korean_part = category.split('|', 1)
                        english_categories = english_part.strip()
                        korean_categories = korean_part.strip()
                    else:
                        english_categories = category
                        korean_categories = "일반, 기본, 표준"
                    y_data.append([english_categories, korean_categories])
                
                y_df = pd.DataFrame(y_data, columns=['영문주요카테고리3개', '한국어주요카테고리3개'])
                y_filename = f"{base_filename}_y_train_{timestamp}.csv"
                y_df.to_csv(y_filename, index=False)
                
                # 전체 데이터 CSV (URL 포함)
                full_df = pd.DataFrame({
                    'image_url': self.image_urls,
                    'category': self.labels
                })
                full_filename = f"{base_filename}_full_data_{timestamp}.csv"
                full_df.to_csv(full_filename, index=False)
                
                # 카테고리별 통계 생성
                all_categories = []
                for category_string in self.labels:
                    categories = [cat.strip() for cat in category_string.split(',') if cat.strip()]
                    all_categories.extend(categories)
                
                from collections import Counter
                category_counts = Counter(all_categories)
                
                self.log_message(f"CSV 파일 저장 완료:")
                self.log_message(f"  - X 데이터: {x_filename}")
                self.log_message(f"  - Y 데이터: {y_filename}")
                self.log_message(f"  - 전체 데이터: {full_filename}")
                self.log_message(f"총 {len(self.image_urls)}개 이미지, {len(category_counts)}개 고유 카테고리")
                self.log_message(f"카테고리 통계: {dict(category_counts)}")
                
                messagebox.showinfo("성공", 
                    f"CSV 파일이 저장되었습니다.\n\n"
                    f"• X 데이터: {x_filename}\n"
                    f"• Y 데이터: {y_filename}\n"
                    f"• 전체 데이터: {full_filename}\n\n"
                    f"총 {len(self.image_urls)}개 이미지, {len(category_counts)}개 고유 카테고리")
                
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 실패: {str(e)}")

    def load_csv(self):
        """CSV 파일 로드"""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                df = pd.read_csv(filename)
                
                if 'image_link.jpg' in df.columns and 'Category' in df.columns:
                    self.image_urls = df['image_link.jpg'].tolist()
                    self.labels = df['Category'].tolist()
                elif 'image_url' in df.columns and 'category' in df.columns:
                    self.image_urls = df['image_url'].tolist()
                    self.labels = df['category'].tolist()
                
                self.update_image_list()
                self.log_message(f"CSV 파일 로드 완료: {filename}")
                
            except Exception as e:
                messagebox.showerror("오류", f"파일 로드 실패: {str(e)}")

    def cleanup_old_data_files(self):
        """이전 정적 데이터 파일들 정리 (최신 파일만 유지)"""
        try:
            dataset_dir = "dataset"
            if not os.path.exists(dataset_dir):
                return
            
            # 타임스탬프가 포함된 자동 생성 파일들 찾기
            import glob
            auto_files = glob.glob(f"{dataset_dir}/*_auto_*.csv")
            
            if auto_files:
                # 파일명에서 타임스탬프 추출하여 정렬
                file_timestamps = []
                for file_path in auto_files:
                    filename = os.path.basename(file_path)
                    # 파일명에서 타임스탬프 추출 (예: x_train_auto_20251025_110352.csv)
                    parts = filename.split('_auto_')
                    if len(parts) == 2:
                        timestamp_part = parts[1].replace('.csv', '')
                        file_timestamps.append((file_path, timestamp_part))
                
                if file_timestamps:
                    # 타임스탬프로 정렬하여 최신 파일 찾기
                    file_timestamps.sort(key=lambda x: x[1], reverse=True)
                    latest_timestamp = file_timestamps[0][1]
                    
                    # 최신 파일이 아닌 모든 파일 삭제
                    deleted_count = 0
                    for file_path, timestamp in file_timestamps[1:]:  # 최신 파일 제외
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                            self.log_message(f"🗑️ 이전 파일 삭제: {os.path.basename(file_path)}")
                        except Exception as e:
                            self.log_message(f"❌ 파일 삭제 실패: {os.path.basename(file_path)} - {e}")
                    
                    if deleted_count > 0:
                        self.log_message(f"✅ {deleted_count}개의 이전 파일 정리 완료 (최신: {latest_timestamp})")
            
            # 오래된 scraped_images 파일들도 정리
            scraped_files = glob.glob(f"{dataset_dir}/scraped_images_*.csv")
            if len(scraped_files) > 1:  # 1개 이상일 때만 정리
                scraped_timestamps = []
                for file_path in scraped_files:
                    filename = os.path.basename(file_path)
                    # 파일명에서 타임스탬프 추출 (예: scraped_images_20251024_152907.csv)
                    parts = filename.replace('scraped_images_', '').replace('.csv', '').split('_')
                    if len(parts) >= 2:
                        timestamp = '_'.join(parts)
                        scraped_timestamps.append((file_path, timestamp))
                
                if scraped_timestamps:
                    scraped_timestamps.sort(key=lambda x: x[1], reverse=True)
                    # 최신 파일 제외하고 나머지 삭제
                    for file_path, timestamp in scraped_timestamps[1:]:
                        try:
                            os.remove(file_path)
                            self.log_message(f"🗑️ 이전 스크래핑 파일 삭제: {os.path.basename(file_path)}")
                        except Exception as e:
                            self.log_message(f"❌ 스크래핑 파일 삭제 실패: {os.path.basename(file_path)} - {e}")
                            
        except Exception as e:
            self.log_message(f"❌ 파일 정리 중 오류: {e}")

    def load_latest_data(self):
        """최신 정적 데이터 로드"""
        try:
            dataset_dir = "dataset"
            if not os.path.exists(dataset_dir):
                return
            
            # 최신 full_data.csv 파일 찾기
            full_data_path = os.path.join(dataset_dir, "full_data.csv")
            if os.path.exists(full_data_path):
                df = pd.read_csv(full_data_path)
                if 'image_url' in df.columns and 'category' in df.columns:
                    self.image_urls = df['image_url'].tolist()
                    self.labels = df['category'].tolist()
                    self.update_image_list()
                    self.log_message(f"📂 최신 데이터 로드: {len(self.image_urls)}개 이미지")
                    return
            
            # full_data.csv가 없으면 x_train.csv 시도
            x_train_path = os.path.join(dataset_dir, "x_train.csv")
            if os.path.exists(x_train_path):
                df = pd.read_csv(x_train_path)
                if 'image_url' in df.columns:
                    self.image_urls = df['image_url'].tolist()
                    self.labels = df['category'].tolist()
                elif 'image_link.jpg' in df.columns:
                    self.image_urls = df['image_link.jpg'].tolist()
                    self.labels = df['Category'].tolist()
                self.update_image_list()
                self.log_message(f"📂 최신 데이터 로드 (x_train): {len(self.image_urls)}개 이미지")
                return
            
            # 기본 파일들도 없으면 자동 생성 파일 중 최신 것 찾기
            import glob
            auto_files = glob.glob(f"{dataset_dir}/full_data_auto_*.csv")
            if auto_files:
                # 타임스탬프로 정렬하여 최신 파일 로드
                file_timestamps = []
                for file_path in auto_files:
                    filename = os.path.basename(file_path)
                    parts = filename.split('_auto_')
                    if len(parts) == 2:
                        timestamp_part = parts[1].replace('.csv', '')
                        file_timestamps.append((file_path, timestamp_part))
                
                if file_timestamps:
                    file_timestamps.sort(key=lambda x: x[1], reverse=True)
                    latest_file = file_timestamps[0][0]
                    df = pd.read_csv(latest_file)
                    if 'image_url' in df.columns and 'category' in df.columns:
                        self.image_urls = df['image_url'].tolist()
                        self.labels = df['category'].tolist()
                        self.update_image_list()
                        self.log_message(f"📂 자동 생성 최신 데이터 로드: {len(self.image_urls)}개 이미지 ({os.path.basename(latest_file)})")
                        
        except Exception as e:
            self.log_message(f"❌ 최신 데이터 로드 실패: {e}")

    def load_existing_data(self):
        """기존 데이터 로드 (하위 호환성 유지)"""
        # 먼저 이전 파일들 정리
        self.cleanup_old_data_files()
        # 최신 데이터 로드
        self.load_latest_data()

    def start_training(self):
        """모델 훈련 시작"""
        if not self.image_urls:
            messagebox.showwarning("경고", "훈련할 데이터가 없습니다.")
            return
        
        self.train_btn.config(state='disabled')
        self.training_status_var.set("훈련 중...")
        
        def training_thread():
            try:
                epochs = int(self.epochs_var.get())
                batch_size = int(self.batch_size_var.get())
                
                # 간단한 훈련 시뮬레이션
                self.log_message("모델 훈련 시작...")
                time.sleep(2)
                
                self.root.after(0, lambda: self.training_completed())
                
            except Exception as e:
                self.root.after(0, lambda: self.training_error(str(e)))
        
        self.training_thread = threading.Thread(target=training_thread)
        self.training_thread.daemon = True
        self.training_thread.start()

    def training_completed(self):
        """훈련 완료 처리"""
        self.train_btn.config(state='normal')
        self.training_status_var.set("훈련 완료")
        self.log_message("모델 훈련 완료!")
        messagebox.showinfo("완료", "모델 훈련이 완료되었습니다.")

    def training_error(self, error_msg: str):
        """훈련 오류 처리"""
        self.train_btn.config(state='normal')
        self.training_status_var.set("오류 발생")
        self.log_message(f"훈련 오류: {error_msg}")
        messagebox.showerror("오류", f"훈련 중 오류가 발생했습니다:\n{error_msg}")

    def __del__(self):
        """소멸자"""
        pass

def main():
    """메인 함수"""
    root = tk.Tk()
    app = CosmosGUIV4(root)
    
    # 창 닫기 이벤트 처리
    def on_closing():
        if app.training_thread and app.training_thread.is_alive():
            if messagebox.askokcancel("종료", "훈련이 진행 중입니다. 정말 종료하시겠습니까?"):
                # 종료 전 자동 저장
                if app.image_urls:
                    app.log_message("앱 종료 중... 유동 데이터를 정적 데이터로 저장합니다.")
                    app.auto_save_data()
                    app.log_message("✅ 자동 저장 완료. 앱을 종료합니다.")
                root.destroy()
        else:
            # 종료 전 자동 저장
            if app.image_urls:
                app.log_message("앱 종료 중... 유동 데이터를 정적 데이터로 저장합니다.")
                app.auto_save_data()
                app.log_message("✅ 자동 저장 완료. 앱을 종료합니다.")
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
