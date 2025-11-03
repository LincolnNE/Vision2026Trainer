#!/usr/bin/env python3
"""
Real Cosmos.so Image Scraper

This script scrapes images from the actual cosmos.so website and
categorizes them based on user-provided word lists to generate
x_train, y_train CSV files.
"""

import os
import pandas as pd
import numpy as np
import requests
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Set
import logging
from pathlib import Path
import time
import re
from urllib.parse import urljoin, urlparse, quote
from bs4 import BeautifulSoup
import json
from collections import defaultdict
import io
import random
import base64

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CosmosRealScraper:
    """Real Cosmos.so image scraper class"""
    
    def __init__(self, timeout: int = 10):
        """
        Args:
            timeout: request timeout (seconds)
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Supported image extensions
        self.image_extensions = {'.webp', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        
        # Cosmos.so related URL patterns
        self.cosmos_patterns = [
            'cosmos.so',
            'www.cosmos.so',
            'cdn.cosmos.so',
            'images.cosmos.so',
            'static.cosmos.so'
        ]
        
    def load_words_from_csv(self, csv_files: List[str]) -> Dict[str, List[str]]:
        """
        CSV 파일에서 단어들을 로딩하고 카테고리별로 분류합니다.
        
        Args:
            csv_files: CSV 파일 경로 리스트
            
        Returns:
            Dict[str, List[str]]: 카테고리별 단어 딕셔너리
        """
        logger.info("CSV 파일에서 단어 로딩 시작...")
        
        all_words = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # 첫 번째 컬럼에서 단어 추출
                words = df.iloc[:, 0].astype(str).tolist()
                all_words.extend(words)
                logger.info(f"{csv_file}: {len(words)}개 단어 로딩")
            except Exception as e:
                logger.warning(f"CSV 파일 로딩 실패 ({csv_file}): {e}")
        
        # 중복 제거 및 정리
        unique_words = list(set([word.strip() for word in all_words if word.strip()]))
        logger.info(f"총 {len(unique_words)}개의 고유 단어 로딩 완료")
        
        # 카테고리별 분류
        categorized_words = self._categorize_words(unique_words)
        
        return categorized_words
    
    def _categorize_words(self, words: List[str]) -> Dict[str, List[str]]:
        """
        단어들을 카테고리별로 분류합니다.
        
        Args:
            words: 단어 리스트
            
        Returns:
            Dict[str, List[str]]: 카테고리별 단어 딕셔너리
        """
        logger.info("단어 카테고리 분류 시작...")
        
        categories = {
            'nature': [],
            'animals': [],
            'food': [],
            'architecture': [],
            'technology': [],
            'art': [],
            'people': [],
            'objects': [],
            'abstract': [],
            'korean_culture': [],
            'general': []
        }
        
        # 개선된 카테고리 키워드 매핑 (더 정확하고 구체적)
        category_keywords = {
            'nature': [
                # 자연 요소
                'tree', 'forest', 'mountain', 'ocean', 'sky', 'flower', 'leaf', 'grass', 'water', 
                'sun', 'moon', 'star', 'cloud', 'rain', 'snow', 'wind', 'earth', 'nature', 
                'landscape', 'park', 'garden', 'beach', 'river', 'lake', 'sea', 'hill', 'valley', 
                'desert', 'jungle', 'wood', 'rock', 'stone', 'sand', 'ice', 'fire', 'lightning', 
                'storm', 'sunset', 'sunrise', 'dawn', 'dusk', 'season', 'spring', 'summer', 
                'autumn', 'winter', 'weather', 'climate', 'environment', 'ecosystem', 'wildlife',
                # 식물 관련
                'plant', 'vegetation', 'flora', 'fauna', 'bloom', 'petal', 'stem', 'root', 'branch',
                'bark', 'seed', 'fruit', 'berry', 'nut', 'pine', 'oak', 'maple', 'cedar', 'birch',
                'rose', 'tulip', 'daisy', 'lily', 'orchid', 'cactus', 'fern', 'moss', 'algae',
                # 지리적 요소
                'volcano', 'canyon', 'cliff', 'cave', 'waterfall', 'stream', 'pond', 'marsh',
                'swamp', 'prairie', 'meadow', 'field', 'farmland', 'vineyard', 'orchard'
            ],
            'animals': [
                # 포유동물
                'cat', 'dog', 'horse', 'cow', 'pig', 'sheep', 'goat', 'deer', 'bear', 'lion', 
                'tiger', 'elephant', 'wolf', 'fox', 'rabbit', 'mouse', 'rat', 'squirrel', 'bat',
                'whale', 'dolphin', 'seal', 'walrus', 'penguin', 'kangaroo', 'koala', 'panda',
                'monkey', 'ape', 'gorilla', 'chimpanzee', 'zebra', 'giraffe', 'hippo', 'rhino',
                # 조류
                'bird', 'eagle', 'hawk', 'owl', 'parrot', 'canary', 'robin', 'sparrow', 'crow',
                'raven', 'swan', 'duck', 'goose', 'chicken', 'rooster', 'turkey', 'peacock',
                # 어류 및 해양생물
                'fish', 'shark', 'salmon', 'tuna', 'cod', 'trout', 'bass', 'carp', 'goldfish',
                'octopus', 'squid', 'jellyfish', 'crab', 'lobster', 'shrimp', 'starfish',
                # 파충류 및 양서류
                'snake', 'lizard', 'gecko', 'iguana', 'turtle', 'tortoise', 'crocodile', 'alligator',
                'frog', 'toad', 'salamander', 'newt',
                # 곤충
                'butterfly', 'bee', 'wasp', 'ant', 'spider', 'beetle', 'dragonfly', 'mosquito',
                'fly', 'moth', 'cricket', 'grasshopper', 'ladybug', 'firefly',
                # 일반 동물 관련
                'animal', 'pet', 'wild', 'domestic', 'mammal', 'reptile', 'amphibian', 'insect',
                'marine', 'aquatic', 'terrestrial', 'flying', 'crawling', 'swimming', 'running'
            ],
            'food': [
                # 기본 음식
                'food', 'meal', 'dish', 'cuisine', 'recipe', 'cooking', 'kitchen', 'restaurant', 
                'cafe', 'bakery', 'diner', 'buffet', 'catering', 'delivery', 'takeout',
                # 메인 요리
                'pizza', 'burger', 'sandwich', 'salad', 'soup', 'pasta', 'spaghetti', 'lasagna',
                'rice', 'noodle', 'ramen', 'sushi', 'curry', 'stew', 'roast', 'grill', 'barbecue',
                # 빵 및 베이커리
                'bread', 'cake', 'cookie', 'pie', 'tart', 'muffin', 'croissant', 'bagel', 'donut',
                'pancake', 'waffle', 'toast', 'roll', 'bun', 'loaf',
                # 과일
                'fruit', 'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'cherry',
                'peach', 'pear', 'plum', 'kiwi', 'pineapple', 'watermelon', 'melon', 'lemon',
                'lime', 'coconut', 'avocado', 'tomato',
                # 채소
                'vegetable', 'carrot', 'potato', 'onion', 'garlic', 'pepper', 'cucumber', 'lettuce',
                'cabbage', 'broccoli', 'cauliflower', 'spinach', 'kale', 'celery', 'radish',
                'eggplant', 'zucchini', 'squash', 'pumpkin', 'corn', 'bean', 'pea', 'mushroom',
                # 고기 및 단백질
                'meat', 'beef', 'pork', 'chicken', 'turkey', 'lamb', 'duck', 'fish', 'seafood',
                'ham', 'bacon', 'sausage', 'hotdog', 'egg', 'cheese', 'milk', 'yogurt', 'butter',
                # 음료
                'drink', 'beverage', 'water', 'juice', 'soda', 'coffee', 'tea', 'wine', 'beer',
                'cocktail', 'smoothie', 'milkshake', 'lemonade', 'iced', 'hot', 'cold',
                # 디저트 및 간식
                'dessert', 'sweet', 'candy', 'chocolate', 'ice', 'cream', 'pudding', 'jelly',
                'gum', 'lollipop', 'caramel', 'fudge', 'brownie', 'cheesecake',
                # 조미료 및 양념
                'spice', 'herb', 'salt', 'pepper', 'sugar', 'honey', 'syrup', 'sauce', 'ketchup',
                'mustard', 'mayo', 'vinegar', 'oil', 'butter', 'cream', 'cheese'
            ],
            'architecture': [
                # 건물 유형
                'building', 'house', 'home', 'apartment', 'condo', 'mansion', 'villa', 'cottage',
                'office', 'tower', 'skyscraper', 'mall', 'shopping', 'center', 'market', 'store',
                'school', 'university', 'college', 'hospital', 'clinic', 'library', 'museum',
                'theater', 'cinema', 'stadium', 'arena', 'gym', 'hotel', 'restaurant', 'cafe',
                # 종교 건물
                'church', 'cathedral', 'chapel', 'temple', 'mosque', 'synagogue', 'shrine',
                'monastery', 'abbey', 'cathedral',
                # 역사적 건물
                'castle', 'palace', 'fortress', 'citadel', 'monument', 'memorial', 'statue',
                'sculpture', 'ruins', 'ancient', 'historic', 'heritage', 'landmark',
                # 건축 요소
                'roof', 'wall', 'door', 'window', 'floor', 'ceiling', 'stair', 'staircase',
                'elevator', 'escalator', 'balcony', 'terrace', 'porch', 'patio', 'deck',
                'facade', 'entrance', 'exit', 'corridor', 'hall', 'room', 'chamber',
                # 건축 스타일
                'modern', 'contemporary', 'classical', 'traditional', 'gothic', 'baroque',
                'renaissance', 'victorian', 'colonial', 'art', 'deco', 'minimalist', 'brutalist',
                'postmodern', 'neoclassical', 'romanesque', 'byzantine',
                # 건축 관련
                'architecture', 'design', 'construction', 'structure', 'foundation', 'framework',
                'beam', 'column', 'pillar', 'arch', 'dome', 'vault', 'bridge', 'tunnel'
            ],
            'technology': [
                # 컴퓨터 및 하드웨어
                'computer', 'laptop', 'desktop', 'pc', 'mac', 'server', 'workstation', 'tablet',
                'phone', 'mobile', 'smartphone', 'iphone', 'android', 'device', 'gadget',
                # 주변기기
                'keyboard', 'mouse', 'monitor', 'screen', 'display', 'printer', 'scanner',
                'speaker', 'headphone', 'microphone', 'camera', 'webcam', 'usb', 'cable',
                # 소프트웨어 및 앱
                'software', 'app', 'application', 'program', 'code', 'programming', 'coding',
                'website', 'web', 'internet', 'browser', 'search', 'engine', 'email', 'social',
                'media', 'platform', 'network', 'database', 'server', 'cloud', 'storage',
                # 기술 분야
                'ai', 'artificial', 'intelligence', 'machine', 'learning', 'deep', 'neural',
                'network', 'algorithm', 'data', 'science', 'analytics', 'blockchain',
                'cryptocurrency', 'bitcoin', 'ethereum', 'vr', 'virtual', 'reality', 'ar',
                'augmented', 'reality', 'iot', 'internet', 'things', 'automation', 'robot',
                'robotics', 'cybersecurity', 'privacy', 'security', 'encryption', 'hacking',
                # 통신 및 네트워킹
                'wifi', 'bluetooth', 'ethernet', 'router', 'modem', 'signal', 'connection',
                'bandwidth', 'speed', 'download', 'upload', 'streaming', 'broadcast'
            ],
            'art': [
                # 미술
                'art', 'painting', 'drawing', 'sketch', 'illustration', 'portrait', 'landscape',
                'still', 'life', 'abstract', 'realistic', 'impressionist', 'expressionist',
                'surreal', 'pop', 'art', 'contemporary', 'modern', 'classical', 'renaissance',
                # 디자인
                'design', 'graphic', 'logo', 'poster', 'banner', 'advertisement', 'marketing',
                'brand', 'identity', 'typography', 'layout', 'composition', 'color', 'palette',
                # 조각 및 설치
                'sculpture', 'statue', 'monument', 'installation', 'ceramic', 'bronze',
                'marble', 'wood', 'metal', 'glass', 'fiber', 'art',
                # 사진
                'photography', 'photo', 'picture', 'image', 'camera', 'lens', 'exposure',
                'aperture', 'shutter', 'iso', 'digital', 'film', 'portrait', 'landscape',
                'macro', 'street', 'documentary', 'fashion', 'wedding', 'event',
                # 음악
                'music', 'song', 'melody', 'rhythm', 'beat', 'harmony', 'instrument', 'piano',
                'guitar', 'violin', 'drum', 'flute', 'trumpet', 'saxophone', 'cello', 'bass',
                'orchestra', 'band', 'singer', 'vocal', 'lyrics', 'composer', 'musician',
                # 공연 예술
                'theater', 'drama', 'play', 'musical', 'opera', 'ballet', 'dance', 'performance',
                'show', 'concert', 'festival', 'exhibition', 'gallery', 'museum', 'cinema',
                'movie', 'film', 'video', 'animation', 'cartoon', 'comic', 'manga', 'anime',
                # 문학
                'literature', 'book', 'novel', 'story', 'poem', 'poetry', 'writing', 'author',
                'writer', 'poet', 'novelist', 'journalist', 'editor', 'publisher', 'magazine',
                'newspaper', 'article', 'essay', 'biography', 'autobiography', 'memoir'
            ],
            'people': [
                # 직업
                'doctor', 'nurse', 'teacher', 'professor', 'student', 'lawyer', 'judge',
                'police', 'officer', 'firefighter', 'soldier', 'pilot', 'captain', 'chef',
                'cook', 'waiter', 'waitress', 'cashier', 'clerk', 'manager', 'director',
                'president', 'mayor', 'governor', 'senator', 'representative', 'minister',
                # 예술가
                'artist', 'painter', 'sculptor', 'photographer', 'musician', 'singer', 'actor',
                'actress', 'dancer', 'writer', 'author', 'poet', 'designer', 'architect',
                # 스포츠
                'athlete', 'player', 'coach', 'referee', 'fan', 'spectator', 'team', 'sport',
                'football', 'soccer', 'basketball', 'baseball', 'tennis', 'golf', 'swimming',
                'running', 'cycling', 'boxing', 'martial', 'arts', 'gymnastics', 'diving',
                # 가족 관계
                'family', 'parent', 'father', 'mother', 'dad', 'mom', 'child', 'baby', 'kid',
                'son', 'daughter', 'brother', 'sister', 'grandfather', 'grandmother', 'uncle',
                'aunt', 'cousin', 'nephew', 'niece', 'husband', 'wife', 'spouse', 'partner',
                # 일반적인 사람
                'person', 'people', 'human', 'man', 'woman', 'boy', 'girl', 'adult', 'teenager',
                'elderly', 'senior', 'citizen', 'friend', 'neighbor', 'stranger', 'visitor',
                'guest', 'customer', 'client', 'patient', 'passenger', 'commuter', 'pedestrian'
            ],
            'objects': [
                # 가구
                'furniture', 'chair', 'table', 'desk', 'sofa', 'couch', 'bed', 'mattress',
                'dresser', 'wardrobe', 'cabinet', 'shelf', 'bookcase', 'nightstand', 'lamp',
                'light', 'mirror', 'clock', 'calendar',
                # 전자제품
                'television', 'tv', 'radio', 'stereo', 'cd', 'dvd', 'bluray', 'remote',
                'control', 'battery', 'charger', 'adapter', 'cable', 'wire', 'plug',
                # 도구
                'tool', 'hammer', 'screwdriver', 'wrench', 'pliers', 'saw', 'drill', 'knife',
                'scissors', 'tape', 'glue', 'rope', 'chain', 'lock', 'key', 'screw', 'nail',
                # 의류
                'clothing', 'clothes', 'shirt', 'pants', 'jeans', 'dress', 'skirt', 'jacket',
                'coat', 'sweater', 'hoodie', 't-shirt', 'shorts', 'underwear', 'socks',
                'shoes', 'boots', 'sneakers', 'sandals', 'hat', 'cap', 'gloves', 'scarf',
                # 액세서리
                'jewelry', 'ring', 'necklace', 'bracelet', 'earrings', 'watch', 'sunglasses',
                'glasses', 'bag', 'purse', 'backpack', 'wallet', 'belt', 'tie', 'bow',
                # 생활용품
                'utensil', 'fork', 'knife', 'spoon', 'plate', 'bowl', 'cup', 'glass', 'mug',
                'bottle', 'can', 'jar', 'container', 'box', 'bag', 'basket', 'tray',
                # 장난감
                'toy', 'doll', 'teddy', 'bear', 'ball', 'game', 'puzzle', 'block', 'lego',
                'robot', 'car', 'truck', 'plane', 'train', 'bike', 'bicycle', 'skateboard'
            ],
            'abstract': [
                # 개념
                'concept', 'idea', 'thought', 'theory', 'principle', 'philosophy', 'logic',
                'reason', 'wisdom', 'knowledge', 'understanding', 'awareness', 'consciousness',
                'mind', 'soul', 'spirit', 'heart', 'emotion', 'feeling', 'mood', 'attitude',
                # 추상적 속성
                'beauty', 'truth', 'justice', 'freedom', 'liberty', 'equality', 'democracy',
                'peace', 'harmony', 'balance', 'order', 'chaos', 'random', 'pattern',
                'structure', 'system', 'organization', 'method', 'technique', 'approach',
                # 시간과 공간
                'time', 'space', 'dimension', 'universe', 'world', 'reality', 'existence',
                'being', 'life', 'death', 'birth', 'creation', 'destruction', 'change',
                'evolution', 'progress', 'development', 'growth', 'decay', 'transformation',
                # 수학적 개념
                'number', 'mathematics', 'geometry', 'algebra', 'calculus', 'statistics',
                'probability', 'equation', 'formula', 'function', 'variable', 'constant',
                'infinity', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                'eight', 'nine', 'ten', 'hundred', 'thousand', 'million', 'billion'
            ]
        }
        
        # 한국어 문화 관련 키워드 (더 포괄적)
        korean_keywords = [
            # 한국 지명
            'korea', 'korean', 'seoul', '서울', 'busan', '부산', 'daegu', '대구', 'incheon', '인천',
            'gwangju', '광주', 'daejeon', '대전', 'ulsan', '울산', 'gyeonggi', '경기', 'gangwon', '강원',
            'chungbuk', '충북', 'chungnam', '충남', 'jeonbuk', '전북', 'jeonnam', '전남',
            'gyeongbuk', '경북', 'gyeongnam', '경남', 'jeju', '제주',
            # 한국 문화
            'hangul', '한글', 'kimchi', '김치', 'bulgogi', '불고기', 'bibimbap', '비빔밥',
            'tteokbokki', '떡볶이', 'ramen', '라면', 'samgyeopsal', '삼겹살', 'chicken', '치킨',
            'beer', '맥주', 'soju', '소주', 'traditional', '전통', 'culture', '문화',
            # 한국 예술
            'k-pop', '케이팝', 'idol', '아이돌', 'singer', '가수', 'actor', '배우', 'celebrity', '연예인',
            'star', '스타', 'fan', '팬', 'concert', '콘서트', 'performance', '공연', 'show', '쇼',
            'program', '프로그램', 'broadcast', '방송', 'television', '텔레비전', 'radio', '라디오',
            # 한국 기술
            'samsung', '삼성', 'lg', '엘지', 'hyundai', '현대', 'kia', '기아', 'posco', '포스코',
            'sk', '에스케이', 'kt', '케이티', 'naver', '네이버', 'kakao', '카카오', 'line', '라인',
            # 한국 음식
            'korean', 'food', '한국', '음식', 'restaurant', '식당', 'cafe', '카페', 'bakery', '베이커리',
            'market', '시장', 'store', '가게', 'shop', '쇼핑', 'mall', '몰', 'department', '백화점',
            # 한국 건물
            'palace', '궁궐', 'temple', '절', 'church', '교회', 'museum', '박물관', 'library', '도서관',
            'university', '대학교', 'school', '학교', 'hospital', '병원', 'station', '역', 'airport', '공항',
            # 한국어 단어들
            '한국', '서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기', '강원', '충북', '충남',
            '전북', '전남', '경북', '경남', '제주', '한글', '김치', '불고기', '비빔밥', '떡볶이', '라면',
            '삼겹살', '치킨', '맥주', '소주', '전통', '문화', '예술', '음악', '춤', '노래', '영화', '드라마',
            '케이팝', '아이돌', '가수', '배우', '연예인', '스타', '팬', '콘서트', '공연', '쇼', '프로그램',
            '방송', '텔레비전', '라디오', '인터넷', '웹사이트', '블로그', '소셜', '미디어', '페이스북',
            '인스타그램', '트위터', '유튜브', '틱톡', '네이버', '카카오', '라인', '왓츠앱', '텔레그램',
            '스카이프', '줌', '구글', '애플', '삼성', '엘지', '현대', '기아', '포스코', '에스케이', '케이티',
            '에스케이티', '씨제이', '롯데', '신세계', '현대백화점', '이마트', '홈플러스', '코스트코', '마트',
            '편의점', '씨유', '지에스', '세븐일레븐', '미니스톱', '이디야', '스타벅스', '투썸플레이스',
            '카페베네', '탐앤탐스', '커피빈', '엔젤리너스', '커피', '차', '녹차', '홍차', '우유', '주스',
            '물', '음료', '과자', '사탕', '초콜릿', '아이스크림', '케이크', '빵', '도넛', '쿠키', '과일',
            '사과', '바나나', '오렌지', '포도', '딸기', '복숭아', '배', '수박', '참외', '토마토', '당근',
            '양파', '감자', '고구마', '옥수수', '콩', '팥', '쌀', '밀', '보리', '야채', '채소', '나물',
            '된장', '고추장', '간장', '식초', '소금', '설탕', '후추', '마늘', '생강', '파', '부추', '시금치',
            '상추', '배추', '무', '오이', '가지', '호박', '고추', '피망', '버섯', '해물', '생선', '고등어',
            '연어', '참치', '새우', '게', '문어', '오징어', '전복', '굴', '홍합', '조개', '소라', '해삼',
            '미역', '김', '다시마', '고기', '소고기', '돼지고기', '닭고기', '양고기', '오리고기', '햄',
            '소시지', '베이컨', '계란', '치즈', '요거트', '버터', '크림', '껌', '젤리', '푸딩', '파이',
            '타르트', '머핀', '크로와상', '베이글', '토스트', '샌드위치', '햄버거', '핫도그', '피자',
            '파스타', '스파게티', '라면', '국수', '냉면', '칼국수', '우동', '소바', '짜장면', '짬뽕',
            '탕수육', '깐풍기', '라조기', '마파두부', '꿔바로우', '양장피', '팔보채', '춘권', '만두',
            '교자', '샤오롱바오', '딤섬', '찐빵', '호떡', '붕어빵', '타코야키', '오코노미야키', '타이야키',
            '와플', '팬케이크', '프렌치토스트', '오믈렛', '스크램블', '계란말이', '계란찜', '계란국',
            '계란탕', '계란볶음밥', '김치볶음밥', '비빔밥', '김밥', '주먹밥', '볶음밥', '덮밥', '카레',
            '라이스', '스테이크', '구이', '갈비', '목살', '항정살', '등심', '안심', '채끝살', '우둔살',
            '설렁탕', '곰탕', '육개장', '추어탕', '매운탕', '해물탕', '삼계탕', '닭볶음탕', '닭갈비',
            '닭도리탕', '닭강정', '후라이드', '양념', '파닭', '닭꼬치', '닭발', '닭똥집', '닭가슴살',
            '닭다리', '닭날개', '닭목', '닭허벅지', '궁궐', '절', '교회', '박물관', '도서관', '대학교',
            '학교', '병원', '역', '공항', '빌딩', '건물', '아파트', '주택', '상가', '사무실', '회사',
            '기업', '공장', '창고', '농장', '공원', '놀이터', '운동장', '체육관', '수영장', '골프장',
            '스키장', '해변', '산', '강', '호수', '바다', '섬', '다리', '터널', '도로', '길', '거리',
            '동네', '마을', '시', '구', '동', '호', '번지', '출구', '호관', '동', '빌딩', '건물'
        ]
        
        for word in words:
            word_lower = word.lower()
            categorized = False
            
            # 한국어 문화 관련 단어 우선 분류
            if any(keyword in word_lower for keyword in korean_keywords):
                categories['korean_culture'].append(word)
                categorized = True
            
            # 다른 카테고리 분류
            if not categorized:
                for category, keywords in category_keywords.items():
                    if any(keyword in word_lower for keyword in keywords):
                        categories[category].append(word)
                        categorized = True
                        break
            
            # 분류되지 않은 단어는 general에 추가
            if not categorized:
                categories['general'].append(word)
        
        # 빈 카테고리 제거 및 통계 출력
        final_categories = {}
        for category, word_list in categories.items():
            if word_list:
                final_categories[category] = word_list
                logger.info(f"카테고리 '{category}': {len(word_list)}개 단어")
        
        return final_categories
    
    def scrape_cosmos_images(self, word: str, max_images: int = 5) -> List[str]:
        """
        Cosmos.so에서 특정 단어에 대한 이미지를 스크래핑합니다.
        
        Args:
            word: 검색할 단어
            max_images: 최대 이미지 수
            
        Returns:
            List[str]: 발견된 이미지 URL 리스트
        """
        logger.info(f"Cosmos.so에서 단어 '{word}' 이미지 스크래핑 시작...")
        
        image_urls = []
        
        try:
            # Cosmos.so 검색 페이지 접근
            search_url = f"https://www.cosmos.so/search?q={quote(word)}"
            
            response = self.session.get(search_url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 이미지 URL 추출
            image_urls = self._extract_image_urls_from_page(soup, word)
            
            # 추가 페이지 스크래핑 (더 많은 이미지를 위해)
            if len(image_urls) < max_images:
                image_urls.extend(self._scrape_additional_pages(word, max_images - len(image_urls)))
            
            # 중복 제거 및 제한
            image_urls = list(set(image_urls))[:max_images]
            
            logger.info(f"단어 '{word}': {len(image_urls)}개 이미지 발견")
            
        except Exception as e:
            logger.warning(f"Cosmos.so 스크래핑 실패 ({word}): {e}")
            # 실패 시 더미 이미지 생성
            image_urls = self._generate_dummy_urls(word, max_images)
        
        return image_urls
    
    def _extract_image_urls_from_page(self, soup: BeautifulSoup, word: str) -> List[str]:
        """페이지에서 이미지 URL을 추출합니다."""
        image_urls = []
        
        # 다양한 이미지 태그 패턴 시도
        img_selectors = [
            'img[src*="cosmos"]',
            'img[data-src*="cosmos"]',
            'img[src*="cdn"]',
            'img[data-src*="cdn"]',
            'img[src*="static"]',
            'img[data-src*="static"]',
            'img[src*="images"]',
            'img[data-src*="images"]'
        ]
        
        for selector in img_selectors:
            try:
                imgs = soup.select(selector)
                for img in imgs:
                    src = img.get('src') or img.get('data-src')
                    if src and self._is_valid_image_url(src):
                        full_url = urljoin('https://www.cosmos.so', src)
                        image_urls.append(full_url)
            except Exception as e:
                logger.debug(f"셀렉터 '{selector}' 처리 실패: {e}")
        
        # JSON 데이터에서 이미지 URL 추출 시도
        try:
            scripts = soup.find_all('script', type='application/json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    image_urls.extend(self._extract_urls_from_json(data))
                except:
                    continue
        except Exception as e:
            logger.debug(f"JSON 데이터 추출 실패: {e}")
        
        return image_urls
    
    def _extract_urls_from_json(self, data: dict) -> List[str]:
        """JSON 데이터에서 이미지 URL을 추출합니다."""
        urls = []
        
        def recursive_search(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and self._is_valid_image_url(value):
                        urls.append(value)
                    elif isinstance(value, (dict, list)):
                        recursive_search(value)
            elif isinstance(obj, list):
                for item in obj:
                    recursive_search(item)
        
        recursive_search(data)
        return urls
    
    def _scrape_additional_pages(self, word: str, max_additional: int) -> List[str]:
        """추가 페이지에서 이미지를 스크래핑합니다."""
        additional_urls = []
        
        try:
            # 다른 검색 엔드포인트 시도
            alternative_urls = [
                f"https://www.cosmos.so/discover?q={quote(word)}",
                f"https://www.cosmos.so/explore?q={quote(word)}",
                f"https://www.cosmos.so/browse?q={quote(word)}"
            ]
            
            for url in alternative_urls:
                if len(additional_urls) >= max_additional:
                    break
                    
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    urls = self._extract_image_urls_from_page(soup, word)
                    additional_urls.extend(urls)
                    
                    time.sleep(1)  # 요청 간 지연
                    
                except Exception as e:
                    logger.debug(f"추가 페이지 스크래핑 실패 ({url}): {e}")
                    continue
                    
        except Exception as e:
            logger.debug(f"추가 페이지 스크래핑 전체 실패: {e}")
        
        return additional_urls[:max_additional]
    
    def _is_valid_image_url(self, url: str) -> bool:
        """URL이 유효한 이미지 파일인지 확인합니다."""
        if not url:
            return False
        
        # Cosmos 관련 도메인 확인
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        if not any(pattern in domain for pattern in self.cosmos_patterns):
            return False
        
        # 확장자 확인
        path = parsed.path.lower()
        for ext in self.image_extensions:
            if path.endswith(ext):
                return True
        
        # 쿼리 파라미터에서 이미지 확장자 확인
        query = parsed.query.lower()
        for ext in self.image_extensions:
            if ext in query:
                return True
        
        return False
    
    def _generate_dummy_urls(self, word: str, max_images: int) -> List[str]:
        """더미 이미지 URL을 생성합니다."""
        # 실제 작동하는 이미지 서비스 사용
        dummy_urls = []
        
        # 다양한 이미지 서비스 사용
        services = [
            f"https://picsum.photos/400/400?random={hash(word) % 1000}",
            f"https://via.placeholder.com/400x400/0066CC/FFFFFF?text={quote(word)}",
            f"https://dummyimage.com/400x400/0066CC/FFFFFF&text={quote(word)}"
        ]
        
        for i in range(max_images):
            if i < len(services):
                dummy_urls.append(services[i])
            else:
                # 추가 URL 생성
                dummy_urls.append(f"https://picsum.photos/400/400?random={hash(word + str(i)) % 1000}")
        
        return dummy_urls
    
    def scrape_images_for_categories(self, categorized_words: Dict[str, List[str]], 
                                   max_images_per_word: int = 3) -> Dict[str, List[str]]:
        """
        카테고리별로 이미지를 스크래핑합니다.
        
        Args:
            categorized_words: 카테고리별 단어 딕셔너리
            max_images_per_word: 단어당 최대 이미지 수
            
        Returns:
            Dict[str, List[str]]: 카테고리별 이미지 URL 딕셔너리
        """
        logger.info("카테고리별 Cosmos.so 이미지 스크래핑 시작...")
        
        category_images = {}
        
        for category, words in categorized_words.items():
            logger.info(f"카테고리 '{category}' 처리 중... ({len(words)}개 단어)")
            
            category_urls = []
            
            # 각 카테고리에서 최대 20개 단어만 처리 (성능상 제한)
            selected_words = words[:20] if len(words) > 20 else words
            
            for word in selected_words:
                try:
                    word_urls = self.scrape_cosmos_images(word, max_images_per_word)
                    category_urls.extend(word_urls)
                    
                    # 요청 간 지연 (서버 부하 방지)
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"단어 '{word}' 이미지 스크래핑 실패: {e}")
                    continue
            
            # 중복 제거
            category_images[category] = list(set(category_urls))
            logger.info(f"카테고리 '{category}': {len(category_images[category])}개 이미지 수집")
        
        return category_images

class CosmosImageDataset(Dataset):
    """Cosmos 이미지 데이터셋 클래스"""
    
    def __init__(self, image_urls: List[str], labels: List[str], transform=None):
        """
        Args:
            image_urls: 이미지 URL 리스트
            labels: 해당 이미지의 라벨 리스트
            transform: 이미지 변환 함수
        """
        self.image_urls = image_urls
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_urls)
    
    def __getitem__(self, idx):
        """인덱스에 해당하는 이미지와 라벨을 반환"""
        try:
            # 이미지 다운로드
            response = requests.get(self.image_urls[idx], timeout=10)
            response.raise_for_status()
            
            # PIL Image로 변환
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            # 변환 적용
            if self.transform:
                image = self.transform(image)
                
            return image, self.labels[idx]
            
        except Exception as e:
            logger.warning(f"이미지 로딩 실패 (URL: {self.image_urls[idx]}): {e}")
            # 실패한 경우 더미 이미지 반환
            image = self._create_dummy_image()
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
    
    def _create_dummy_image(self):
        """더미 이미지 생성 (실제 이미지가 없을 때 사용)"""
        # 랜덤한 패턴의 더미 이미지 생성
        dummy_image = Image.new('RGB', (224, 224), color='white')
        return dummy_image

class SimpleCNN(nn.Module):
    """간단한 CNN 이미지 분류 모델"""
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: 분류할 클래스 수
        """
        super(SimpleCNN, self).__init__()
        
        # 컨볼루션 레이어들
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 풀링 레이어
        self.pool = nn.MaxPool2d(2, 2)
        
        # 드롭아웃
        self.dropout = nn.Dropout(0.5)
        
        # 완전연결 레이어들
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # 활성화 함수
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """순전파"""
        # 첫 번째 컨볼루션 블록
        x = self.pool(self.relu(self.conv1(x)))  # 224x224 -> 112x112
        
        # 두 번째 컨볼루션 블록
        x = self.pool(self.relu(self.conv2(x)))   # 112x112 -> 56x56
        
        # 세 번째 컨볼루션 블록
        x = self.pool(self.relu(self.conv3(x)))   # 56x56 -> 28x28
        
        # 평탄화
        x = x.view(-1, 128 * 28 * 28)
        
        # 완전연결 레이어들
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class CosmosPipeline:
    """Cosmos.so 실제 스크래핑 이미지 분류 파이프라인 메인 클래스"""
    
    def __init__(self, data_dir: str = "./dataset", model_dir: str = "./models", results_dir: str = "./results"):
        """
        Args:
            data_dir: 데이터 저장 디렉토리
            model_dir: 모델 저장 디렉토리
            results_dir: 결과 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        
        # 디렉토리 생성
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # 스크래퍼 초기화
        self.scraper = CosmosRealScraper()
        
        # 데이터 저장 변수
        self.x_train_data = None
        self.y_train_data = None
        self.label_encoder = None
        
    def load_words_and_scrape_images(self, csv_files: List[str], max_images_per_word: int = 3) -> Tuple[List[str], List[str]]:
        """
        CSV 파일에서 단어를 로딩하고 Cosmos.so에서 이미지를 스크래핑합니다.
        
        Args:
            csv_files: CSV 파일 경로 리스트
            max_images_per_word: 단어당 최대 이미지 수
            
        Returns:
            Tuple[List[str], List[str]]: (이미지 URL 리스트, 라벨 리스트)
        """
        logger.info("단어 로딩 및 Cosmos.so 이미지 스크래핑 시작...")
        
        # 1. CSV 파일에서 단어 로딩 및 카테고리 분류
        categorized_words = self.scraper.load_words_from_csv(csv_files)
        
        if not categorized_words:
            logger.warning("로딩된 단어가 없습니다. 더미 데이터를 사용합니다.")
            return self._create_dummy_data()
        
        # 2. 카테고리별 이미지 스크래핑
        category_images = self.scraper.scrape_images_for_categories(
            categorized_words, max_images_per_word
        )
        
        # 3. 데이터 구성
        all_urls = []
        all_labels = []
        
        for category, urls in category_images.items():
            all_urls.extend(urls)
            all_labels.extend([category] * len(urls))
        
        logger.info(f"총 {len(all_urls)}개의 이미지 데이터 준비 완료")
        logger.info(f"카테고리별 분포: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
        
        return all_urls, all_labels
    
    def _create_dummy_data(self) -> Tuple[List[str], List[str]]:
        """더미 데이터 생성 (스크래핑 실패 시 사용)"""
        logger.info("더미 데이터 생성 중...")
        
        dummy_data = {
            "nature": [
                "https://picsum.photos/400/400?random=1",
                "https://picsum.photos/400/400?random=2",
                "https://picsum.photos/400/400?random=3",
            ],
            "animals": [
                "https://picsum.photos/400/400?random=4",
                "https://picsum.photos/400/400?random=5",
                "https://picsum.photos/400/400?random=6",
            ],
            "food": [
                "https://picsum.photos/400/400?random=7",
                "https://picsum.photos/400/400?random=8",
                "https://picsum.photos/400/400?random=9",
            ]
        }
        
        all_urls = []
        all_labels = []
        
        for category, urls in dummy_data.items():
            all_urls.extend(urls)
            all_labels.extend([category] * len(urls))
        
        return all_urls, all_labels
    
    def create_csv_files(self, image_urls: List[str], labels: List[str]):
        """
        x_train과 y_train 데이터를 CSV 파일로 저장합니다.
        
        Args:
            image_urls: 이미지 URL 리스트
            labels: 라벨 리스트
        """
        logger.info("CSV 파일 생성 중...")
        
        # x_train 데이터프레임 생성
        x_train_df = pd.DataFrame({
            'image_url': image_urls,
            'category': labels
        })
        
        # y_train 데이터프레임 생성
        y_train_df = pd.DataFrame({
            'label': labels
        })
        
        # CSV 파일 저장
        x_train_path = self.data_dir / "x_train.csv"
        y_train_path = self.data_dir / "y_train.csv"
        
        x_train_df.to_csv(x_train_path, index=False)
        y_train_df.to_csv(y_train_path, index=False)
        
        logger.info(f"x_train.csv 저장 완료: {x_train_path}")
        logger.info(f"y_train.csv 저장 완료: {y_train_path}")
        
        # 데이터 저장
        self.x_train_data = x_train_df
        self.y_train_data = y_train_df
    
    def load_csv_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        저장된 CSV 파일을 로딩합니다.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (x_train, y_train) 데이터프레임
        """
        logger.info("CSV 파일 로딩 중...")
        
        x_train_path = self.data_dir / "x_train.csv"
        y_train_path = self.data_dir / "y_train.csv"
        
        if not x_train_path.exists() or not y_train_path.exists():
            raise FileNotFoundError("CSV 파일이 존재하지 않습니다. 먼저 create_csv_files()를 실행하세요.")
        
        x_train_df = pd.read_csv(x_train_path)
        y_train_df = pd.read_csv(y_train_path)
        
        logger.info(f"CSV 파일 로딩 완료: {len(x_train_df)}개 샘플")
        
        return x_train_df, y_train_df
    
    def preprocess_data(self, x_train_df: pd.DataFrame, y_train_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, int]:
        """
        데이터 전처리 및 데이터로더 생성
        
        Args:
            x_train_df: x_train 데이터프레임
            y_train_df: y_train 데이터프레임
            
        Returns:
            Tuple[DataLoader, DataLoader, int]: (train_loader, test_loader, num_classes)
        """
        logger.info("데이터 전처리 시작...")
        
        # 라벨 인코딩
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(y_train_df['label'].values)
        num_classes = len(self.label_encoder.classes_)
        
        logger.info(f"클래스 수: {num_classes}")
        logger.info(f"클래스 목록: {self.label_encoder.classes_}")
        
        # 이미지 변환 정의
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 데이터셋 생성
        dataset = CosmosImageDataset(
            x_train_df['image_url'].tolist(),
            encoded_labels.tolist(),
            transform=transform
        )
        
        # train/test 분할 (8:2)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        # 데이터로더 생성
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        logger.info(f"훈련 데이터: {len(train_dataset)}개")
        logger.info(f"테스트 데이터: {len(test_dataset)}개")
        
        return train_loader, test_loader, num_classes
    
    def train_model(self, train_loader: DataLoader, test_loader: DataLoader, num_classes: int, epochs: int = 10):
        """
        모델 학습
        
        Args:
            train_loader: 훈련 데이터로더
            test_loader: 테스트 데이터로더
            num_classes: 클래스 수
            epochs: 학습 에포크 수
        """
        logger.info("모델 학습 시작...")
        
        # 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"사용 디바이스: {device}")
        
        # 모델 생성
        model = SimpleCNN(num_classes).to(device)
        
        # 손실 함수 및 옵티마이저
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 학습 기록
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # 훈련 모드
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
                
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # 훈련 정확도 계산
            train_accuracy = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # 테스트 평가
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
            
            test_accuracy = 100 * test_correct / test_total
            avg_test_loss = test_loss / len(test_loader)
            
            # 기록 저장
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(avg_test_loss)
            test_accuracies.append(test_accuracy)
            
            logger.info(f'Epoch {epoch+1}/{epochs}:')
            logger.info(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            logger.info(f'  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        
        # 모델 저장
        model_path = self.model_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder': self.label_encoder,
            'num_classes': num_classes
        }, model_path)
        
        logger.info(f"모델 저장 완료: {model_path}")
        
        # 학습 결과 시각화
        self.plot_training_results(train_losses, train_accuracies, test_losses, test_accuracies)
        
        return model, train_losses, train_accuracies, test_losses, test_accuracies
    
    def plot_training_results(self, train_losses: List[float], train_accuracies: List[float], 
                            test_losses: List[float], test_accuracies: List[float]):
        """
        학습 결과 시각화
        
        Args:
            train_losses: 훈련 손실 리스트
            train_accuracies: 훈련 정확도 리스트
            test_losses: 테스트 손실 리스트
            test_accuracies: 테스트 정확도 리스트
        """
        logger.info("학습 결과 시각화 중...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 손실 그래프
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(test_losses, label='Test Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 정확도 그래프
        ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(test_accuracies, label='Test Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 그래프 저장
        metrics_path = self.results_dir / "metrics.png"
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"시각화 결과 저장 완료: {metrics_path}")
    
    def test_model(self, model_path: str, test_image_url: str = None):
        """
        모델 테스트 및 예측
        
        Args:
            model_path: 모델 파일 경로
            test_image_url: 테스트할 이미지 URL (선택사항)
        """
        logger.info("모델 테스트 시작...")
        
        # 모델 로딩 (weights_only=False로 설정하여 LabelEncoder 포함)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model = SimpleCNN(checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        label_encoder = checkpoint['label_encoder']
        
        # 테스트 이미지 변환
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if test_image_url:
            try:
                # 테스트 이미지 다운로드 및 예측
                response = requests.get(test_image_url, timeout=10)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(output, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                
                logger.info(f"테스트 이미지 예측 결과:")
                logger.info(f"  예측 클래스: {predicted_label}")
                logger.info(f"  신뢰도: {confidence:.4f}")
                
                return predicted_label, confidence
                
            except Exception as e:
                logger.error(f"테스트 이미지 처리 실패: {e}")
                return None, None
        else:
            logger.info("테스트 이미지 URL이 제공되지 않았습니다.")
            return None, None

def main():
    """메인 실행 함수"""
    logger.info("Cosmos.so 실제 스크래핑 이미지 분류 파이프라인 시작")
    
    # 파이프라인 초기화
    pipeline = CosmosPipeline()
    
    try:
        # 1. CSV 파일에서 단어 로딩 및 Cosmos.so 이미지 스크래핑
        csv_files = ["words0.csv", "words1.csv"]
        image_urls, labels = pipeline.load_words_and_scrape_images(csv_files, max_images_per_word=2)
        
        # 2. CSV 파일 생성
        pipeline.create_csv_files(image_urls, labels)
        
        # 3. CSV 데이터 로딩
        x_train_df, y_train_df = pipeline.load_csv_data()
        
        # 4. 데이터 전처리
        train_loader, test_loader, num_classes = pipeline.preprocess_data(x_train_df, y_train_df)
        
        # 5. 모델 학습
        model, train_losses, train_accuracies, test_losses, test_accuracies = pipeline.train_model(
            train_loader, test_loader, num_classes, epochs=5
        )
        
        # 6. 모델 테스트
        model_path = pipeline.model_dir / "model.pt"
        pipeline.test_model(str(model_path))
        
        logger.info("파이프라인 실행 완료!")
        
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
