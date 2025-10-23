#!/usr/bin/env python3
"""
Cosmos CDN 자동 스크래핑 및 이미지 분류 모델 학습 파이프라인

이 스크립트는 cosmos.so CDN에서 이미지를 자동으로 스크래핑하고
이미지 분류 모델을 학습하는 완전 자동화된 파이프라인을 제공합니다.

주요 기능:
- cosmos.so CDN 자동 스크래핑
- URL 경로 기반 자동 카테고리 라벨링
- 이미지 다운로드 및 전처리
- CNN 모델 학습 및 평가
- 결과 시각화 및 모델 저장
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
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
from collections import defaultdict
import io

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CosmosScraper:
    """cosmos.so CDN 이미지 스크래퍼 클래스"""
    
    def __init__(self, base_url: str = "https://cdn.cosmos.so", timeout: int = 10):
        """
        Args:
            base_url: cosmos CDN 기본 URL
            timeout: 요청 타임아웃 (초)
        """
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 지원하는 이미지 확장자
        self.image_extensions = {'.webp', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        
    def scrape_image_urls(self, max_pages: int = 10) -> List[str]:
        """
        cosmos CDN에서 이미지 URL을 스크래핑합니다.
        
        Args:
            max_pages: 최대 탐색 페이지 수
            
        Returns:
            List[str]: 발견된 이미지 URL 리스트
        """
        logger.info(f"cosmos CDN 스크래핑 시작: {self.base_url}")
        
        image_urls = []
        visited_urls = set()
        
        try:
            # 메인 페이지에서 시작
            urls_to_visit = [self.base_url]
            
            for page_num in range(max_pages):
                if not urls_to_visit:
                    break
                    
                current_url = urls_to_visit.pop(0)
                if current_url in visited_urls:
                    continue
                    
                visited_urls.add(current_url)
                logger.info(f"페이지 스크래핑 중: {current_url}")
                
                try:
                    response = self.session.get(current_url, timeout=self.timeout)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # 이미지 링크 찾기
                    page_image_urls = self._extract_image_urls(soup, current_url)
                    image_urls.extend(page_image_urls)
                    
                    # 추가 페이지 링크 찾기
                    new_urls = self._extract_page_urls(soup, current_url)
                    for url in new_urls:
                        if url not in visited_urls and url not in urls_to_visit:
                            urls_to_visit.append(url)
                            
                except Exception as e:
                    logger.warning(f"페이지 스크래핑 실패 ({current_url}): {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"스크래핑 중 오류 발생: {e}")
            
        # 중복 제거
        image_urls = list(set(image_urls))
        logger.info(f"총 {len(image_urls)}개의 이미지 URL 발견")
        
        return image_urls
    
    def _extract_image_urls(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """HTML에서 이미지 URL을 추출합니다."""
        image_urls = []
        
        # img 태그에서 이미지 URL 추출
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                full_url = urljoin(base_url, src)
                if self._is_image_url(full_url):
                    image_urls.append(full_url)
        
        # 링크에서 이미지 파일 추출
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            if self._is_image_url(full_url):
                image_urls.append(full_url)
        
        return image_urls
    
    def _extract_page_urls(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """추가 탐색할 페이지 URL을 추출합니다."""
        page_urls = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # 같은 도메인 내의 URL만 추가
            if self._is_same_domain(full_url) and not self._is_image_url(full_url):
                page_urls.append(full_url)
        
        return page_urls
    
    def _is_image_url(self, url: str) -> bool:
        """URL이 이미지 파일인지 확인합니다."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # 확장자 확인
        for ext in self.image_extensions:
            if path.endswith(ext):
                return True
        
        # URL 패턴 확인 (cosmos CDN 특화)
        if 'cosmos.so' in url and any(pattern in url for pattern in ['image', 'img', 'photo', 'pic']):
            return True
            
        return False
    
    def _is_same_domain(self, url: str) -> bool:
        """같은 도메인인지 확인합니다."""
        parsed = urlparse(url)
        base_parsed = urlparse(self.base_url)
        return parsed.netloc == base_parsed.netloc
    
    def categorize_images(self, image_urls: List[str]) -> Dict[str, List[str]]:
        """
        이미지 URL을 카테고리별로 분류합니다.
        
        Args:
            image_urls: 이미지 URL 리스트
            
        Returns:
            Dict[str, List[str]]: 카테고리별 이미지 URL 딕셔너리
        """
        logger.info("이미지 카테고리 분류 시작...")
        
        categorized = defaultdict(list)
        
        for url in image_urls:
            category = self._extract_category_from_url(url)
            categorized[category].append(url)
        
        # 카테고리별 통계 출력
        for category, urls in categorized.items():
            logger.info(f"카테고리 '{category}': {len(urls)}개 이미지")
        
        return dict(categorized)
    
    def _extract_category_from_url(self, url: str) -> str:
        """
        URL에서 카테고리를 추출합니다.
        
        Args:
            url: 이미지 URL
            
        Returns:
            str: 추출된 카테고리명
        """
        parsed = urlparse(url)
        path_parts = [part for part in parsed.path.split('/') if part]
        
        # URL 경로에서 카테고리 추출
        category_keywords = {
            'book': 'book_layout',
            'art': 'artwork',
            'photo': 'photography',
            'magazine': 'magazine_layout',
            'portfolio': 'portfolio',
            'design': 'design',
            'minimal': 'minimal_design',
            'abstract': 'abstract_art',
            'texture': 'texture',
            'pattern': 'pattern',
            'layout': 'layout',
            'creative': 'creative',
            'black': 'black_white',
            'white': 'monochrome',
            'color': 'colorful',
            'vintage': 'vintage',
            'modern': 'modern',
            'classic': 'classic'
        }
        
        # 경로에서 키워드 찾기
        for part in path_parts:
            part_lower = part.lower()
            for keyword, category in category_keywords.items():
                if keyword in part_lower:
                    return category
        
        # 파일명에서 카테고리 추출
        filename = path_parts[-1] if path_parts else ""
        filename_lower = filename.lower()
        
        for keyword, category in category_keywords.items():
            if keyword in filename_lower:
                return category
        
        # 기본 카테고리
        return 'general'

class CosmosImageDataset(Dataset):
    """cosmos CDN 이미지 데이터셋 클래스"""
    
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

class AutoScrapingPipeline:
    """자동 스크래핑 이미지 분류 파이프라인 메인 클래스"""
    
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
        self.scraper = CosmosScraper()
        
        # 데이터 저장 변수
        self.x_train_data = None
        self.y_train_data = None
        self.label_encoder = None
        
    def scrape_and_categorize(self, max_pages: int = 10) -> Tuple[List[str], List[str]]:
        """
        cosmos CDN에서 이미지를 스크래핑하고 카테고리별로 분류합니다.
        
        Args:
            max_pages: 최대 탐색 페이지 수
            
        Returns:
            Tuple[List[str], List[str]]: (이미지 URL 리스트, 라벨 리스트)
        """
        logger.info("이미지 스크래핑 및 카테고리 분류 시작...")
        
        # 이미지 URL 스크래핑
        image_urls = self.scraper.scrape_image_urls(max_pages=max_pages)
        
        if not image_urls:
            logger.warning("스크래핑된 이미지가 없습니다. 더미 데이터를 사용합니다.")
            return self._create_dummy_data()
        
        # 카테고리별 분류
        categorized_images = self.scraper.categorize_images(image_urls)
        
        # 데이터 구성
        all_urls = []
        all_labels = []
        
        for category, urls in categorized_images.items():
            all_urls.extend(urls)
            all_labels.extend([category] * len(urls))
        
        logger.info(f"총 {len(all_urls)}개의 이미지 데이터 준비 완료")
        logger.info(f"카테고리별 분포: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
        
        return all_urls, all_labels
    
    def _create_dummy_data(self) -> Tuple[List[str], List[str]]:
        """더미 데이터 생성 (스크래핑 실패 시 사용)"""
        logger.info("더미 데이터 생성 중...")
        
        # 실제 작동하는 이미지 URL 사용 (Unsplash 등)
        dummy_data = {
            "book_layout": [
                "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400",  # 책 이미지
                "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # 책 이미지
                "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=400",   # 책 이미지
                "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=400", # 책 이미지
                "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400", # 책 이미지
            ],
            "photography": [
                "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400", # 자연 사진
                "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400", # 자연 사진
                "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=400", # 자연 사진
                "https://images.unsplash.com/photo-1447752875215-b2761acb3c5d?w=400", # 자연 사진
                "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400", # 자연 사진
            ],
            "design": [
                "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400",   # 디자인 이미지
                "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400",   # 디자인 이미지
                "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400",   # 디자인 이미지
                "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400",   # 디자인 이미지
                "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400",   # 디자인 이미지
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
    logger.info("자동 스크래핑 이미지 분류 파이프라인 시작")
    
    # 파이프라인 초기화
    pipeline = AutoScrapingPipeline()
    
    try:
        # 1. cosmos CDN에서 이미지 스크래핑 및 카테고리 분류
        image_urls, labels = pipeline.scrape_and_categorize(max_pages=5)
        
        # 2. CSV 파일 생성
        pipeline.create_csv_files(image_urls, labels)
        
        # 3. CSV 데이터 로딩
        x_train_df, y_train_df = pipeline.load_csv_data()
        
        # 4. 데이터 전처리
        train_loader, test_loader, num_classes = pipeline.preprocess_data(x_train_df, y_train_df)
        
        # 5. 모델 학습
        model, train_losses, train_accuracies, test_losses, test_accuracies = pipeline.train_model(
            train_loader, test_loader, num_classes, epochs=10
        )
        
        # 6. 모델 테스트
        model_path = pipeline.model_dir / "model.pt"
        pipeline.test_model(str(model_path))
        
        logger.info("파이프라인 실행 완료!")
        
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    import io  # io 모듈을 여기서 임포트
    main()
