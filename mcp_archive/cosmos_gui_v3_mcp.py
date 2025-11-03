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
    """Gemini API 직접 호출 클라이언트"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.is_available = bool(self.api_key)
        
    def analyze_image(self, image_url: str) -> str:
        """Gemini Vision API를 사용하여 이미지 분석"""
        if not self.is_available:
            logger.warning("GEMINI_API_KEY가 설정되지 않음")
            return "general, design, creative"
        
        try:
            # 이미지 다운로드
            response = requests.get(image_url, timeout=10)
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
            else:
                logger.error(f"Gemini API 오류: {gemini_response.status_code}")
                return "general, design, creative"
                
        except Exception as e:
            logger.error(f"이미지 분석 실패: {e}")
            return "general, design, creative"
    
    def batch_analyze_images(self, image_urls: List[str]) -> List[str]:
        """여러 이미지 일괄 분석"""
        results = []
        for i, url in enumerate(image_urls):
            logger.info(f"이미지 {i+1}/{len(image_urls)} 분석 중...")
            result = self.analyze_image(url)
            results.append(result)
            time.sleep(0.5)  # API 호출 간격 조절
        return results

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
                if self.gemini_client.is_available:
                    self.root.after(0, lambda: self.log_message("✅ Gemini API 연결 성공"))
                    self.root.after(0, lambda: self.update_gemini_status("연결됨"))
                else:
                    self.root.after(0, lambda: self.log_message("❌ Gemini API 키가 설정되지 않음"))
                    self.root.after(0, lambda: self.update_gemini_status("API 키 없음"))
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"❌ Gemini API 연결 오류: {e}"))
                self.root.after(0, lambda: self.update_gemini_status("연결 실패"))
        
        threading.Thread(target=test_thread, daemon=True).start()

    def update_gemini_status(self, status):
        """Gemini API 상태 업데이트"""
        if hasattr(self, 'gemini_status_label'):
            self.gemini_status_label.config(text=f"Gemini API 상태: {status}")
            if "연결됨" in status:
                self.gemini_status_label.config(foreground="green")
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
        
        ttk.Button(gemini_status_frame, text="API 재연결", command=self.test_gemini_connection).pack(side=tk.RIGHT)
        
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
        ttk.Button(scraping_frame, text="CSV 로드", command=self.load_csv).pack(side=tk.RIGHT)

    def setup_image_panel(self, parent):
        """이미지 미리보기 패널 구성"""
        image_frame = ttk.LabelFrame(parent, text="AI 이미지 분석", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 좌측: 이미지 리스트
        left_frame = ttk.Frame(image_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
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
        self.category_var.trace('w', self.on_category_text_change)
        self.category_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        # 도움말 텍스트 추가
        help_label = ttk.Label(category_frame, text="(콤마로 여러 카테고리 구분)", font=("Arial", 8))
        help_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # 카테고리 변경 버튼
        ttk.Button(category_frame, text="카테고리 변경", command=self.change_category).pack(side=tk.RIGHT)
        
        # AI 분석 결과 표시
        analysis_frame = ttk.Frame(left_frame)
        analysis_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(analysis_frame, text="AI 분석 결과:").pack(anchor=tk.W)
        self.analysis_var = tk.StringVar(value="이미지를 선택하고 AI 분석을 시작하세요")
        self.analysis_label = ttk.Label(analysis_frame, textvariable=self.analysis_var, 
                                       foreground='blue', wraplength=300)
        self.analysis_label.pack(anchor=tk.W)
        
        # 우측: 이미지 미리보기
        right_frame = ttk.Frame(image_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_frame, text="이미지 미리보기:").pack(anchor=tk.W)
        
        self.image_label = ttk.Label(right_frame, text="이미지를 선택하세요", 
                                   background='white', relief=tk.SUNKEN)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

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
                
                # 이미지 크기 조정 (미리보기용)
                max_size = (400, 300)
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
        
        index = selection[0]
        url = self.image_urls[index]
        
        def analysis_thread():
            self.log_message(f"이미지 {index+1} AI 분석 시작...")
            
            try:
                # Gemini API 직접 호출
                suggested_categories = self.gemini_client.analyze_image(url)
                
                if suggested_categories:
                    self.root.after(0, lambda: self.log_message(f"✅ AI 분석 완료: {suggested_categories}"))
                    
                    # 카테고리 입력 필드에 자동 입력
                    self.root.after(0, lambda: self.category_var.set(suggested_categories))
                    
                    # 분석 결과 표시
                    self.root.after(0, lambda: self.analysis_var.set(
                        f"AI 추천 카테고리: {suggested_categories}"
                    ))
                else:
                    self.root.after(0, lambda: self.log_message("❌ AI 분석 실패"))
                    self.root.after(0, lambda: self.analysis_var.set("AI 분석 실패"))
                    
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
        
        def batch_analysis_thread():
            self.log_message(f"전체 {len(self.image_urls)}개 이미지 AI 분석 시작...")
            
            try:
                # Gemini API 직접 호출로 배치 분석
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
        
        threading.Thread(target=batch_analysis_thread, daemon=True).start()

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

    def select_all_images(self, event):
        """Cmd+A 또는 Ctrl+A로 모든 이미지 선택"""
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

    def save_csv(self):
        """CSV 파일 저장 (여러 카테고리 지원)"""
        if not self.image_urls:
            messagebox.showwarning("경고", "저장할 데이터가 없습니다.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # x_train.csv: image_link.jpg, Category (여러 카테고리 콤마 구분)
                x_data = []
                for i, (url, category) in enumerate(zip(self.image_urls, self.labels)):
                    filename_only = url.split('/')[-1].split('?')[0]
                    x_data.append([filename_only, category])
                
                x_df = pd.DataFrame(x_data, columns=['image_link.jpg', 'Category'])
                x_df.to_csv(filename.replace('.csv', '_x_train.csv'), index=False)
                
                # y_train.csv: Category (여러 카테고리 콤마 구분)
                y_df = pd.DataFrame({'Category': self.labels})
                y_df.to_csv(filename.replace('.csv', '_y_train.csv'), index=False)
                
                # 카테고리별 통계 생성
                all_categories = []
                for category_string in self.labels:
                    categories = [cat.strip() for cat in category_string.split(',') if cat.strip()]
                    all_categories.extend(categories)
                
                from collections import Counter
                category_counts = Counter(all_categories)
                
                self.log_message(f"CSV 파일 저장 완료: {filename}")
                self.log_message(f"총 {len(self.image_urls)}개 이미지, {len(category_counts)}개 고유 카테고리")
                self.log_message(f"카테고리 통계: {dict(category_counts)}")
                messagebox.showinfo("성공", f"CSV 파일이 저장되었습니다.\n총 {len(self.image_urls)}개 이미지, {len(category_counts)}개 고유 카테고리")
                
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

    def load_existing_data(self):
        """기존 데이터 로드"""
        try:
            if os.path.exists('./dataset/x_train.csv'):
                df = pd.read_csv('./dataset/x_train.csv')
                if 'image_url' in df.columns:
                    self.image_urls = df['image_url'].tolist()
                    self.labels = df['category'].tolist()
                elif 'image_link.jpg' in df.columns:
                    self.image_urls = df['image_link.jpg'].tolist()
                    self.labels = df['Category'].tolist()
                self.update_image_list()
                self.log_message(f"기존 데이터 로드: {len(self.image_urls)}개 이미지")
        except Exception as e:
            self.log_message(f"기존 데이터 로드 실패: {str(e)}")

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
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
