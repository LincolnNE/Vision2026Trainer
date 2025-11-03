#!/usr/bin/env python3
"""
Cosmos.so 이미지 분류 GUI 애플리케이션 v2.0
- 동적 카테고리 시스템 (words0.csv, words1.csv 기반)
- 자유 카테고리 입력
- 개선된 출력 형식
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import io
import re
import time
import json
from urllib.parse import urljoin, urlparse
from collections import Counter
import random

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicCategorySystem:
    """동적 카테고리 시스템"""
    
    def __init__(self):
        self.word_categories = {}
        self.category_keywords = {}
        self.load_word_categories()
        
    def load_word_categories(self):
        """words0.csv, words1.csv에서 단어 카테고리 로딩"""
        try:
            # words0.csv 로딩
            if os.path.exists('words0.csv'):
                df0 = pd.read_csv('words0.csv', header=None)
                words0 = df0[0].tolist()
                logger.info(f"words0.csv에서 {len(words0)}개 단어 로딩")
            else:
                words0 = []
                
            # words1.csv 로딩  
            if os.path.exists('words1.csv'):
                df1 = pd.read_csv('words1.csv', header=None)
                words1 = df1[0].tolist()
                logger.info(f"words1.csv에서 {len(words1)}개 단어 로딩")
            else:
                words1 = []
                
            all_words = words0 + words1
            
            # 동적 카테고리 매핑
            self.category_keywords = {
                'nature': [
                    'tree', 'forest', 'mountain', 'ocean', 'sky', 'flower', 'leaf', 'grass', 'water', 
                    'sun', 'moon', 'star', 'cloud', 'rain', 'snow', 'wind', 'earth', 'nature', 
                    'landscape', 'park', 'garden', 'beach', 'river', 'lake', 'sea', 'hill', 'valley',
                    'plant', 'vegetation', 'flora', 'fauna', 'bloom', 'petal', 'stem', 'root', 'branch',
                    'volcano', 'canyon', 'cliff', 'cave', 'waterfall', 'stream', 'pond', 'marsh'
                ],
                'animals': [
                    'cat', 'dog', 'horse', 'cow', 'pig', 'sheep', 'goat', 'deer', 'bear', 'lion', 
                    'tiger', 'elephant', 'wolf', 'fox', 'rabbit', 'mouse', 'rat', 'squirrel', 'bat',
                    'whale', 'dolphin', 'seal', 'walrus', 'penguin', 'kangaroo', 'koala', 'panda',
                    'bird', 'eagle', 'hawk', 'owl', 'parrot', 'canary', 'robin', 'sparrow', 'crow',
                    'fish', 'shark', 'salmon', 'tuna', 'cod', 'trout', 'bass', 'carp', 'goldfish',
                    'snake', 'lizard', 'gecko', 'iguana', 'turtle', 'tortoise', 'crocodile', 'alligator',
                    'butterfly', 'bee', 'wasp', 'ant', 'spider', 'beetle', 'dragonfly', 'mosquito',
                    'animal', 'pet', 'wildlife', 'mammal', 'reptile', 'amphibian', 'insect'
                ],
                'food': [
                    'food', 'meal', 'dish', 'cuisine', 'recipe', 'cooking', 'kitchen', 'restaurant', 
                    'cafe', 'bakery', 'diner', 'buffet', 'catering', 'delivery', 'takeout',
                    'pizza', 'burger', 'sandwich', 'salad', 'soup', 'pasta', 'spaghetti', 'lasagna',
                    'bread', 'cake', 'cookie', 'pie', 'tart', 'muffin', 'croissant', 'bagel', 'donut',
                    'fruit', 'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'cherry',
                    'vegetable', 'carrot', 'potato', 'onion', 'garlic', 'pepper', 'cucumber', 'lettuce',
                    'meat', 'beef', 'pork', 'chicken', 'turkey', 'lamb', 'duck', 'fish', 'seafood',
                    'drink', 'beverage', 'water', 'juice', 'soda', 'coffee', 'tea', 'wine', 'beer',
                    'dessert', 'sweet', 'candy', 'chocolate', 'ice', 'cream', 'pudding', 'jelly'
                ],
                'architecture': [
                    'building', 'house', 'home', 'apartment', 'condo', 'mansion', 'villa', 'cottage',
                    'office', 'tower', 'skyscraper', 'mall', 'shopping', 'center', 'market', 'store',
                    'school', 'university', 'college', 'hospital', 'clinic', 'library', 'museum',
                    'church', 'cathedral', 'chapel', 'temple', 'mosque', 'synagogue', 'shrine',
                    'castle', 'palace', 'fortress', 'citadel', 'monument', 'memorial', 'statue',
                    'roof', 'wall', 'door', 'window', 'floor', 'ceiling', 'stair', 'staircase',
                    'modern', 'contemporary', 'classical', 'traditional', 'gothic', 'baroque',
                    'architecture', 'design', 'construction', 'structure', 'foundation', 'framework',
                    'bridge', 'tunnel', 'landmark', 'city', 'urban', 'downtown'
                ],
                'technology': [
                    'computer', 'laptop', 'desktop', 'pc', 'mac', 'server', 'workstation', 'tablet',
                    'phone', 'mobile', 'smartphone', 'iphone', 'android', 'device', 'gadget',
                    'keyboard', 'mouse', 'monitor', 'screen', 'display', 'printer', 'scanner',
                    'software', 'app', 'application', 'program', 'code', 'programming', 'coding',
                    'website', 'web', 'internet', 'browser', 'search', 'engine', 'email', 'social',
                    'ai', 'artificial', 'intelligence', 'machine', 'learning', 'deep', 'neural',
                    'network', 'algorithm', 'data', 'science', 'analytics', 'blockchain',
                    'vr', 'virtual', 'reality', 'ar', 'augmented', 'reality', 'iot', 'automation',
                    'robot', 'robotics', 'cybersecurity', 'privacy', 'security', 'encryption',
                    'wifi', 'bluetooth', 'ethernet', 'router', 'modem', 'signal', 'connection'
                ],
                'art': [
                    'art', 'painting', 'drawing', 'sketch', 'illustration', 'portrait', 'landscape',
                    'still', 'life', 'abstract', 'realistic', 'impressionist', 'expressionist',
                    'surreal', 'pop', 'art', 'contemporary', 'modern', 'classical', 'renaissance',
                    'design', 'graphic', 'logo', 'poster', 'banner', 'advertisement', 'marketing',
                    'brand', 'identity', 'typography', 'layout', 'composition', 'color', 'palette',
                    'sculpture', 'statue', 'monument', 'installation', 'ceramic', 'bronze',
                    'photography', 'photo', 'picture', 'image', 'camera', 'lens', 'exposure',
                    'music', 'song', 'melody', 'rhythm', 'beat', 'harmony', 'instrument', 'piano',
                    'theater', 'drama', 'play', 'musical', 'opera', 'ballet', 'dance', 'performance',
                    'literature', 'book', 'novel', 'story', 'poem', 'poetry', 'writing', 'author',
                    'gallery', 'museum', 'exhibition', 'cinema', 'movie', 'film', 'video', 'animation'
                ],
                'people': [
                    'doctor', 'nurse', 'teacher', 'professor', 'student', 'lawyer', 'judge',
                    'police', 'officer', 'firefighter', 'soldier', 'pilot', 'captain', 'chef',
                    'artist', 'painter', 'sculptor', 'photographer', 'musician', 'singer', 'actor',
                    'athlete', 'player', 'coach', 'referee', 'fan', 'spectator', 'team', 'sport',
                    'family', 'parent', 'father', 'mother', 'dad', 'mom', 'child', 'baby', 'kid',
                    'person', 'people', 'human', 'man', 'woman', 'boy', 'girl', 'adult', 'teenager',
                    'friend', 'neighbor', 'stranger', 'visitor', 'guest', 'customer', 'client',
                    'portrait', 'face', 'smile', 'happy', 'professional', 'group', 'crowd'
                ],
                'objects': [
                    'furniture', 'chair', 'table', 'desk', 'sofa', 'couch', 'bed', 'mattress',
                    'television', 'tv', 'radio', 'stereo', 'cd', 'dvd', 'bluray', 'remote',
                    'tool', 'hammer', 'screwdriver', 'wrench', 'pliers', 'saw', 'drill', 'knife',
                    'clothing', 'clothes', 'shirt', 'pants', 'jeans', 'dress', 'skirt', 'jacket',
                    'jewelry', 'ring', 'necklace', 'bracelet', 'earrings', 'watch', 'sunglasses',
                    'utensil', 'fork', 'knife', 'spoon', 'plate', 'bowl', 'cup', 'glass', 'mug',
                    'toy', 'doll', 'teddy', 'bear', 'ball', 'game', 'puzzle', 'block', 'lego',
                    'book', 'car', 'bike', 'bicycle', 'skateboard', 'object', 'item', 'product'
                ],
                'abstract': [
                    'concept', 'idea', 'thought', 'theory', 'principle', 'philosophy', 'logic',
                    'beauty', 'truth', 'justice', 'freedom', 'liberty', 'equality', 'democracy',
                    'peace', 'harmony', 'balance', 'order', 'chaos', 'random', 'pattern',
                    'time', 'space', 'dimension', 'universe', 'world', 'reality', 'existence',
                    'number', 'mathematics', 'geometry', 'algebra', 'calculus', 'statistics',
                    'pattern', 'texture', 'shape', 'minimal', 'geometric', 'line', 'color', 'form',
                    'quote', 'spirituality', 'meditation', 'zen', 'mindfulness', 'consciousness'
                ],
                'korean_culture': [
                    'korea', 'korean', 'seoul', 'busan', 'daegu', 'incheon', 'gwangju', 'daejeon',
                    'hangul', '한글', 'kimchi', '김치', 'bulgogi', '불고기', 'bibimbap', '비빔밥',
                    'k-pop', '케이팝', 'idol', '아이돌', 'singer', '가수', 'actor', '배우',
                    'samsung', '삼성', 'lg', '엘지', 'hyundai', '현대', 'kia', '기아',
                    'hanbok', '한복', 'palace', '궁궐', 'temple', '절', 'church', '교회',
                    'traditional', '전통', 'culture', '문화', 'korean food', '한국 음식'
                ],
                'fashion': [
                    'fashion', 'style', 'clothing', 'clothes', 'outfit', 'dress', 'suit', 'jacket',
                    'shirt', 'blouse', 'pants', 'jeans', 'skirt', 'shorts', 'coat', 'sweater',
                    'shoes', 'boots', 'sneakers', 'sandals', 'heels', 'flats', 'socks',
                    'hat', 'cap', 'gloves', 'scarf', 'belt', 'tie', 'bow', 'accessory',
                    'jewelry', 'ring', 'necklace', 'bracelet', 'earrings', 'watch', 'sunglasses',
                    'bag', 'purse', 'backpack', 'wallet', 'handbag', 'clutch', 'tote',
                    'makeup', 'cosmetics', 'lipstick', 'mascara', 'foundation', 'eyeshadow',
                    'runway', 'model', 'designer', 'brand', 'luxury', 'trend', 'vintage'
                ],
                'culture': [
                    'culture', 'tradition', 'heritage', 'custom', 'ritual', 'ceremony', 'festival',
                    'holiday', 'celebration', 'party', 'event', 'wedding', 'birthday', 'anniversary',
                    'religion', 'spiritual', 'faith', 'belief', 'worship', 'prayer', 'meditation',
                    'music', 'dance', 'theater', 'performance', 'show', 'concert', 'festival',
                    'art', 'craft', 'handmade', 'traditional', 'folk', 'ethnic', 'indigenous',
                    'language', 'literature', 'poetry', 'story', 'myth', 'legend', 'folklore',
                    'history', 'ancient', 'medieval', 'renaissance', 'modern', 'contemporary',
                    'halloween', 'christmas', 'easter', 'thanksgiving', 'new year', 'valentine'
                ]
            }
            
            # 단어별 카테고리 매핑 생성
            for word in all_words:
                word_lower = word.lower()
                self.word_categories[word] = self._categorize_word(word_lower)
                
        except Exception as e:
            logger.error(f"단어 카테고리 로딩 실패: {e}")
            
    def _categorize_word(self, word: str) -> str:
        """단어를 카테고리로 분류"""
        # 한국어 문화 관련 단어 우선 분류
        korean_keywords = ['korea', 'korean', 'seoul', '한국', '서울', '한복', '김치', '궁궐', '절', '전통']
        if any(keyword in word for keyword in korean_keywords):
            return 'korean_culture'
            
        # 다른 카테고리 분류
        for category, keywords in self.category_keywords.items():
            if any(keyword in word for keyword in keywords):
                return category
                
        return 'general'
    
    def suggest_category(self, image_url: str, alt_text: str = "") -> List[str]:
        """이미지에 대한 카테고리 추천"""
        suggestions = []
        
        # URL 기반 추천
        url_lower = image_url.lower()
        for category, keywords in self.category_keywords.items():
            if any(keyword in url_lower for keyword in keywords):
                suggestions.append(category)
        
        # Alt 텍스트 기반 추천
        if alt_text:
            alt_lower = alt_text.lower()
            for category, keywords in self.category_keywords.items():
                if any(keyword in alt_lower for keyword in keywords):
                    suggestions.append(category)
        
        # 랜덤 추천 (동적성 추가)
        if not suggestions:
            suggestions = random.sample(list(self.category_keywords.keys()), 3)
        else:
            # 기존 추천에 랜덤 카테고리 추가
            additional = random.sample(list(self.category_keywords.keys()), 2)
            suggestions.extend(additional)
        
        # 중복 제거 및 순서 섞기
        suggestions = list(set(suggestions))
        random.shuffle(suggestions)
        
        return suggestions[:5]  # 최대 5개 추천

class CosmosScraper:
    """Cosmos.so 스크래퍼 클래스"""
    
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.category_system = DynamicCategorySystem()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    def _update_progress(self, message: str, progress: int = None):
        """진행 상황 업데이트"""
        if self.progress_callback:
            self.progress_callback(message, progress)

    def scrape_images(self) -> Tuple[List[str], List[str]]:
        """이미지 스크래핑"""
        self._update_progress("Cosmos.so 스크래핑 시작...", 0)
        
        image_data = []
        
        try:
            self._update_progress("cosmos.so/discover 페이지 접속 중...", 10)
            response = requests.get("https://www.cosmos.so/discover", headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                self._update_progress("페이지 파싱 중...", 30)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                img_tags = soup.find_all('img')
                total_images = len([img for img in img_tags if img.get('src') and 'cdn.cosmos.so' in img.get('src', '')])
                
                self._update_progress(f"총 {total_images}개 이미지 발견", 50)
                
                processed = 0
                for img in img_tags:
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src and 'cdn.cosmos.so' in src:
                        alt_text = img.get('alt', '')
                        
                        # 동적 카테고리 추천
                        suggested_categories = self.category_system.suggest_category(src, alt_text)
                        category = suggested_categories[0] if suggested_categories else 'featured'
                        
                        high_res_url = self._get_high_resolution_url(src)
                        
                        if high_res_url and self._validate_image_url(high_res_url):
                            image_data.append((high_res_url, category))
                            self._update_progress(f"이미지 수집: {category} ({len(image_data)}/{total_images})", 
                                                50 + int(40 * processed / total_images))
                        
                        processed += 1
                        time.sleep(0.1)  # 요청 간격 조절
                
                self._update_progress(f"스크래핑 완료: {len(image_data)}개 이미지 수집", 100)
                
        except Exception as e:
            self._update_progress(f"스크래핑 실패: {str(e)}", -1)
            return [], []
        
        # 중복 제거
        unique_pairs = list(set(zip([url for url, _ in image_data], [cat for _, cat in image_data])))
        if unique_pairs:
            all_image_urls, all_labels = zip(*unique_pairs)
            return list(all_image_urls), list(all_labels)
        
        return [], []

    def _get_high_resolution_url(self, src: str) -> Optional[str]:
        """고해상도 URL 변환"""
        try:
            if 'cdn.cosmos.so' in src and 'w=' in src:
                return re.sub(r'w=\d+', 'w=1080', src)
            elif 'cdn.cosmos.so' in src:
                if '?' in src:
                    return src + '&w=1080'
                else:
                    return src + '?w=1080'
        except:
            pass
        return None

    def _validate_image_url(self, url: str) -> bool:
        """이미지 URL 유효성 검증"""
        try:
            response = requests.head(url, headers=self.headers, timeout=5)
            return response.status_code == 200 and 'image' in response.headers.get('content-type', '').lower()
        except:
            return False

class ImageDataset(Dataset):
    """이미지 데이터셋"""
    
    def __init__(self, image_urls: List[str], labels: List[int], transform=None):
        self.image_urls = image_urls
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, idx):
        image_url = self.image_urls[idx]
        label = self.labels[idx]
        
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            image.thumbnail((224, 224), Image.Resampling.LANCZOS)
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            logger.warning(f"이미지 로딩 실패: {e}")
            dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label

class SimpleCNN(nn.Module):
    """간단한 CNN 모델"""
    
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CosmosGUI:
    """메인 GUI 애플리케이션 v2.0"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Cosmos.so 이미지 분류 훈련 관리자 v2.0")
        self.root.geometry("1600x1000")
        
        # 데이터 저장
        self.image_urls = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.current_dataset = None
        self.training_thread = None
        self.training_queue = queue.Queue()
        self.category_system = DynamicCategorySystem()
        
        # GUI 구성
        self.setup_ui()
        self.setup_logging()
        
        # 기존 데이터 로드
        self.load_existing_data()

    def setup_ui(self):
        """UI 구성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 상단 패널 (스크래핑 제어)
        self.setup_scraping_panel(main_frame)
        
        # 중간 패널 (이미지 미리보기 및 카테고리 관리)
        self.setup_image_panel(main_frame)
        
        # 하단 패널 (모델 훈련 및 결과)
        self.setup_training_panel(main_frame)

    def setup_scraping_panel(self, parent):
        """스크래핑 패널 구성"""
        scraping_frame = ttk.LabelFrame(parent, text="스크래핑 제어", padding=10)
        scraping_frame.pack(fill=tk.X, pady=(0, 10))
        
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
        image_frame = ttk.LabelFrame(parent, text="이미지 관리", padding=10)
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
        
        # 카테고리 관리
        category_frame = ttk.Frame(left_frame)
        category_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(category_frame, text="카테고리:").pack(side=tk.LEFT)
        
        # 카테고리 입력 (자유 입력 가능)
        self.category_var = tk.StringVar()
        self.category_entry = ttk.Entry(category_frame, textvariable=self.category_var, width=20)
        self.category_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        # AI 추천 카테고리 버튼
        ttk.Button(category_frame, text="AI 추천", command=self.get_ai_suggestions).pack(side=tk.LEFT, padx=(0, 5))
        
        # 카테고리 변경 버튼
        ttk.Button(category_frame, text="카테고리 변경", command=self.change_category).pack(side=tk.RIGHT)
        
        # AI 추천 카테고리 표시
        suggestion_frame = ttk.Frame(left_frame)
        suggestion_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(suggestion_frame, text="AI 추천 카테고리:").pack(anchor=tk.W)
        self.suggestion_var = tk.StringVar(value="이미지를 선택하면 AI 추천이 표시됩니다")
        self.suggestion_label = ttk.Label(suggestion_frame, textvariable=self.suggestion_var, 
                                         foreground='blue', wraplength=300)
        self.suggestion_label.pack(anchor=tk.W)
        
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
            scraper = CosmosScraper(progress_callback=self.update_scraping_progress)
            urls, labels = scraper.scrape_images()
            
            self.root.after(0, lambda: self.scraping_completed(urls, labels))
        
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
        
        # AI 추천 카테고리 표시
        suggestions = self.category_system.suggest_category(url)
        suggestion_text = f"추천: {', '.join(suggestions)}"
        self.suggestion_var.set(suggestion_text)
        
        # 이미지 로드 및 표시
        def load_image():
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content))
                image.thumbnail((300, 300), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo  # 참조 유지
                
            except Exception as e:
                self.image_label.config(image="", text=f"이미지 로드 실패:\n{str(e)}")
        
        threading.Thread(target=load_image, daemon=True).start()

    def get_ai_suggestions(self):
        """AI 추천 카테고리 가져오기"""
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("경고", "이미지를 선택하세요.")
            return
        
        index = selection[0]
        url = self.image_urls[index]
        
        suggestions = self.category_system.suggest_category(url)
        
        # 추천 카테고리 선택 다이얼로그
        suggestion_window = tk.Toplevel(self.root)
        suggestion_window.title("AI 카테고리 추천")
        suggestion_window.geometry("400x300")
        
        ttk.Label(suggestion_window, text="AI가 추천하는 카테고리:", font=('Arial', 12, 'bold')).pack(pady=10)
        
        for i, suggestion in enumerate(suggestions):
            btn = ttk.Button(suggestion_window, text=suggestion, 
                           command=lambda s=suggestion: self.apply_suggestion(s, suggestion_window))
            btn.pack(pady=5, padx=20, fill=tk.X)

    def apply_suggestion(self, suggestion: str, window):
        """추천 카테고리 적용"""
        self.category_var.set(suggestion)
        window.destroy()

    def change_category(self):
        """카테고리 변경"""
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("경고", "이미지를 선택하세요.")
            return
        
        index = selection[0]
        new_category = self.category_var.get().strip()
        
        if not new_category:
            messagebox.showwarning("경고", "새 카테고리를 입력하세요.")
            return
        
        self.labels[index] = new_category
        self.update_image_list()
        self.log_message(f"이미지 {index+1}의 카테고리를 '{new_category}'로 변경")

    def save_csv(self):
        """CSV 파일 저장 (새로운 형식)"""
        if not self.image_urls:
            messagebox.showwarning("경고", "저장할 데이터가 없습니다.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # x_train.csv: image_link.jpg, Category
                x_data = []
                for i, (url, category) in enumerate(zip(self.image_urls, self.labels)):
                    filename_only = url.split('/')[-1].split('?')[0]
                    x_data.append([filename_only, category])
                
                x_df = pd.DataFrame(x_data, columns=['image_link.jpg', 'Category'])
                x_df.to_csv(filename.replace('.csv', '_x_train.csv'), index=False)
                
                # y_train.csv: Category
                y_df = pd.DataFrame({'Category': self.labels})
                y_df.to_csv(filename.replace('.csv', '_y_train.csv'), index=False)
                
                self.log_message(f"CSV 파일 저장 완료: {filename}")
                messagebox.showinfo("성공", "CSV 파일이 저장되었습니다.")
                
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
                    # x_train.csv 형식
                    self.image_urls = df['image_link.jpg'].tolist()
                    self.labels = df['Category'].tolist()
                elif 'image_url' in df.columns and 'category' in df.columns:
                    # 기존 형식
                    self.image_urls = df['image_url'].tolist()
                    self.labels = df['category'].tolist()
                elif 'Category' in df.columns:
                    # y_train.csv 형식
                    self.labels = df['Category'].tolist()
                    if not self.image_urls:
                        messagebox.showwarning("경고", "이미지 URL이 없습니다. x_train.csv를 먼저 로드하세요.")
                        return
                
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
                
                self.train_model(epochs, batch_size)
                
            except Exception as e:
                self.root.after(0, lambda: self.training_error(str(e)))
        
        self.training_thread = threading.Thread(target=training_thread)
        self.training_thread.daemon = True
        self.training_thread.start()

    def train_model(self, epochs: int, batch_size: int):
        """모델 훈련"""
        try:
            # 데이터 전처리
            self.root.after(0, lambda: self.log_message("데이터 전처리 중..."))
            
            unique_labels = np.unique(self.labels)
            num_classes = len(unique_labels)
            encoded_labels = self.label_encoder.fit_transform(self.labels)
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = ImageDataset(self.image_urls, encoded_labels, transform)
            
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # 모델 초기화
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SimpleCNN(num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 훈련 루프
            train_losses = []
            train_accuracies = []
            test_losses = []
            test_accuracies = []
            
            for epoch in range(epochs):
                # 훈련
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    train_total += target.size(0)
                    train_correct += (predicted == target).sum().item()
                
                # 테스트
                model.eval()
                test_loss = 0.0
                test_correct = 0
                test_total = 0
                
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        loss = criterion(output, target)
                        
                        test_loss += loss.item()
                        _, predicted = torch.max(output.data, 1)
                        test_total += target.size(0)
                        test_correct += (predicted == target).sum().item()
                
                # 메트릭 계산
                train_loss_avg = train_loss / len(train_loader)
                train_acc = 100. * train_correct / train_total
                test_loss_avg = test_loss / len(test_loader)
                test_acc = 100. * test_correct / test_total
                
                train_losses.append(train_loss_avg)
                train_accuracies.append(train_acc)
                test_losses.append(test_loss_avg)
                test_accuracies.append(test_acc)
                
                # UI 업데이트
                self.root.after(0, lambda e=epoch+1, tl=train_loss_avg, ta=train_acc, 
                              vl=test_loss_avg, va=test_acc: 
                              self.update_training_progress(e, epochs, tl, ta, vl, va))
                
                # 그래프 업데이트
                self.root.after(0, lambda: self.update_training_graph(train_losses, train_accuracies, 
                                                                     test_losses, test_accuracies))
            
            # 모델 저장
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'label_encoder': self.label_encoder
            }, './models/cosmos_gui_model_v2.pt')
            
            self.root.after(0, lambda: self.training_completed())
            
        except Exception as e:
            self.root.after(0, lambda: self.training_error(str(e)))

    def update_training_progress(self, epoch: int, total_epochs: int, 
                               train_loss: float, train_acc: float,
                               test_loss: float, test_acc: float):
        """훈련 진행 상황 업데이트"""
        self.training_status_var.set(f"Epoch {epoch}/{total_epochs}")
        self.log_message(f"Epoch {epoch}/{total_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    def update_training_graph(self, train_losses: List[float], train_accuracies: List[float],
                            test_losses: List[float], test_accuracies: List[float]):
        """훈련 그래프 업데이트"""
        self.ax1.clear()
        self.ax1.plot(train_losses, label='Train Loss')
        self.ax1.plot(test_losses, label='Test Loss')
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        
        self.ax2.clear()
        self.ax2.plot(train_accuracies, label='Train Accuracy')
        self.ax2.plot(test_accuracies, label='Test Accuracy')
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.legend()
        
        self.canvas.draw()

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

def main():
    """메인 함수"""
    root = tk.Tk()
    app = CosmosGUI(root)
    
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
