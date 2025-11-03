#!/usr/bin/env python3
"""
개선된 이미지 스크래핑 파이프라인
- 실제 이미지 검색 엔진 연동
- 이미지 품질 검증
- 다중 소스 지원
"""

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
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import time
import random
from urllib.parse import quote, urljoin
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedImageScraper:
    """개선된 이미지 스크래핑 클래스"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # 실제 작동하는 이미지 소스들
        self.image_sources = {
            'unsplash': self._scrape_unsplash,
            'pixabay': self._scrape_pixabay,
            'pexels': self._scrape_pexels,
            'flickr': self._scrape_flickr,
            'stockvault': self._scrape_stockvault
        }
        
        # 카테고리별 키워드 매핑 (더 구체적)
        self.category_keywords = {
            'nature': ['landscape', 'forest', 'mountain', 'ocean', 'sunset', 'sunrise', 'tree', 'flower', 'sky', 'cloud'],
            'animals': ['cat', 'dog', 'bird', 'wildlife', 'elephant', 'lion', 'tiger', 'bear', 'deer', 'rabbit'],
            'food': ['food', 'meal', 'restaurant', 'cooking', 'fruit', 'vegetable', 'bread', 'cake', 'pizza', 'salad'],
            'architecture': ['building', 'house', 'city', 'bridge', 'tower', 'castle', 'church', 'modern', 'classical'],
            'technology': ['computer', 'phone', 'robot', 'digital', 'tech', 'innovation', 'ai', 'software', 'hardware'],
            'art': ['painting', 'art', 'gallery', 'museum', 'sculpture', 'design', 'creative', 'colorful', 'abstract'],
            'people': ['portrait', 'person', 'face', 'smile', 'happy', 'family', 'children', 'business', 'professional'],
            'objects': ['object', 'tool', 'furniture', 'car', 'book', 'clock', 'camera', 'watch', 'bag', 'shoes'],
            'abstract': ['abstract', 'pattern', 'geometric', 'color', 'shape', 'texture', 'minimal', 'modern'],
            'korean_culture': ['korea', 'seoul', 'korean', 'hanbok', 'kimchi', 'k-pop', 'traditional', 'korean food'],
            'general': ['general', 'mixed', 'various', 'diverse', 'random', 'collection']
        }

    def _scrape_unsplash(self, query: str, count: int = 5) -> List[str]:
        """Unsplash에서 이미지 스크래핑"""
        urls = []
        try:
            # Unsplash의 검색 API 사용 (공개 API)
            search_url = f"https://unsplash.com/napi/search/photos?query={quote(query)}&per_page={count}&page=1"
            response = requests.get(search_url, headers=self.headers, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                for photo in data.get('results', []):
                    # 고품질 이미지 URL 추출
                    img_url = photo['urls']['regular']  # 1080px 이미지
                    urls.append(img_url)
                    
            logger.info(f"Unsplash에서 '{query}'에 대해 {len(urls)}개 이미지 발견")
        except Exception as e:
            logger.warning(f"Unsplash 스크래핑 실패: {e}")
            
        return urls

    def _scrape_pixabay(self, query: str, count: int = 5) -> List[str]:
        """Pixabay에서 이미지 스크래핑"""
        urls = []
        try:
            # Pixabay 공개 API 사용 (무료)
            api_key = "your_pixabay_api_key"  # 실제 사용시 API 키 필요
            search_url = f"https://pixabay.com/api/?key={api_key}&q={quote(query)}&image_type=photo&per_page={count}"
            
            # API 키가 없는 경우 웹 스크래핑 시도
            if api_key == "your_pixabay_api_key":
                search_url = f"https://pixabay.com/images/search/{quote(query)}/"
                response = requests.get(search_url, headers=self.headers, timeout=self.timeout)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Pixabay의 이미지 컨테이너 찾기
                    for img in soup.find_all('img', {'src': True}):
                        src = img.get('src')
                        if src and 'pixabay.com' in src and 'static' not in src:
                            urls.append(src)
                            if len(urls) >= count:
                                break
                                
            logger.info(f"Pixabay에서 '{query}'에 대해 {len(urls)}개 이미지 발견")
        except Exception as e:
            logger.warning(f"Pixabay 스크래핑 실패: {e}")
            
        return urls

    def _scrape_pexels(self, query: str, count: int = 5) -> List[str]:
        """Pexels에서 이미지 스크래핑"""
        urls = []
        try:
            search_url = f"https://www.pexels.com/search/{quote(query)}/"
            response = requests.get(search_url, headers=self.headers, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Pexels의 이미지 요소 찾기
                for img in soup.find_all('img', {'src': True}):
                    src = img.get('src')
                    if src and 'images.pexels.com' in src:
                        urls.append(src)
                        if len(urls) >= count:
                            break
                            
            logger.info(f"Pexels에서 '{query}'에 대해 {len(urls)}개 이미지 발견")
        except Exception as e:
            logger.warning(f"Pexels 스크래핑 실패: {e}")
            
        return urls

    def _scrape_flickr(self, query: str, count: int = 5) -> List[str]:
        """Flickr에서 이미지 스크래핑"""
        urls = []
        try:
            search_url = f"https://www.flickr.com/search/?text={quote(query)}"
            response = requests.get(search_url, headers=self.headers, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Flickr의 이미지 요소 찾기
                for img in soup.find_all('img', {'src': True}):
                    src = img.get('src')
                    if src and 'staticflickr.com' in src:
                        urls.append(src)
                        if len(urls) >= count:
                            break
                            
            logger.info(f"Flickr에서 '{query}'에 대해 {len(urls)}개 이미지 발견")
        except Exception as e:
            logger.warning(f"Flickr 스크래핑 실패: {e}")
            
        return urls

    def _scrape_stockvault(self, query: str, count: int = 5) -> List[str]:
        """StockVault에서 이미지 스크래핑"""
        urls = []
        try:
            search_url = f"https://www.stockvault.net/search/{quote(query)}"
            response = requests.get(search_url, headers=self.headers, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # StockVault의 이미지 요소 찾기
                for img in soup.find_all('img', {'src': True}):
                    src = img.get('src')
                    if src and 'stockvault.net' in src:
                        urls.append(src)
                        if len(urls) >= count:
                            break
                            
            logger.info(f"StockVault에서 '{query}'에 대해 {len(urls)}개 이미지 발견")
        except Exception as e:
            logger.warning(f"StockVault 스크래핑 실패: {e}")
            
        return urls

    def _validate_image_url(self, url: str) -> bool:
        """이미지 URL 유효성 검증"""
        try:
            response = requests.head(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                return content_type.startswith('image/')
        except:
            pass
        return False

    def _get_category_keywords(self, category: str) -> List[str]:
        """카테고리에 맞는 키워드 반환"""
        return self.category_keywords.get(category, ['general'])

    def scrape_images_for_category(self, category: str, count_per_source: int = 3) -> List[str]:
        """카테고리별로 이미지 스크래핑"""
        logger.info(f"카테고리 '{category}'에 대한 이미지 스크래핑 시작...")
        
        all_urls = []
        keywords = self._get_category_keywords(category)
        
        # 각 소스에서 이미지 수집
        for source_name, scrape_func in self.image_sources.items():
            try:
                # 키워드 중 랜덤 선택
                keyword = random.choice(keywords)
                urls = scrape_func(keyword, count_per_source)
                
                # URL 유효성 검증
                valid_urls = [url for url in urls if self._validate_image_url(url)]
                all_urls.extend(valid_urls)
                
                logger.info(f"{source_name}에서 {len(valid_urls)}개 유효한 이미지 수집")
                
                # 요청 간격 조절 (과도한 요청 방지)
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"{source_name} 스크래핑 중 오류: {e}")
        
        # 중복 제거 및 품질 필터링
        unique_urls = list(set(all_urls))
        
        # 더미 이미지가 아닌지 확인
        filtered_urls = []
        for url in unique_urls:
            if not any(dummy in url.lower() for dummy in ['default-avatars', 'placeholder', 'dummy']):
                filtered_urls.append(url)
        
        logger.info(f"카테고리 '{category}': 총 {len(filtered_urls)}개 고품질 이미지 수집")
        return filtered_urls[:count_per_source * 2]  # 최대 6개로 제한

    def run_improved_scraping(self, categories: List[str]) -> Tuple[List[str], List[str]]:
        """개선된 스크래핑 실행"""
        logger.info("개선된 이미지 스크래핑 시작...")
        
        all_image_urls = []
        all_labels = []
        
        for category in categories:
            try:
                urls = self.scrape_images_for_category(category)
                
                if urls:
                    all_image_urls.extend(urls)
                    all_labels.extend([category] * len(urls))
                    logger.info(f"카테고리 '{category}': {len(urls)}개 이미지 수집 완료")
                else:
                    logger.warning(f"카테고리 '{category}': 이미지 수집 실패")
                    
            except Exception as e:
                logger.error(f"카테고리 '{category}' 처리 중 오류: {e}")
        
        logger.info(f"총 {len(all_image_urls)}개의 고품질 이미지 데이터 준비 완료")
        return all_image_urls, all_labels

# 기존 ImageClassificationPipeline 클래스 재사용
class ImageClassificationPipeline:
    """이미지 분류 파이프라인"""
    
    def __init__(self, image_urls: List[str], labels: List[str], 
                 output_dir: str = "./dataset", model_save_path: str = "./models/model.pt",
                 metrics_save_path: str = "./results/metrics.png"):
        self.image_urls = image_urls
        self.labels = labels
        self.output_dir = output_dir
        self.model_save_path = model_save_path
        self.metrics_save_path = metrics_save_path
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
        
        # 라벨 인코더
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
        # 이미지 변환
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def create_csv_files(self):
        """CSV 파일 생성"""
        logger.info("CSV 파일 생성 중...")
        
        # x_train.csv 생성
        x_train_df = pd.DataFrame({
            'image_url': self.image_urls,
            'category': self.labels
        })
        x_train_path = os.path.join(self.output_dir, 'x_train.csv')
        x_train_df.to_csv(x_train_path, index=False)
        logger.info(f"x_train.csv 저장 완료: {x_train_path}")
        
        # y_train.csv 생성
        y_train_df = pd.DataFrame({
            'label': self.labels
        })
        y_train_path = os.path.join(self.output_dir, 'y_train.csv')
        y_train_df.to_csv(y_train_path, index=False)
        logger.info(f"y_train.csv 저장 완료: {y_train_path}")

    def run(self):
        """파이프라인 실행"""
        logger.info("이미지 분류 파이프라인 시작...")
        
        # CSV 파일 생성
        self.create_csv_files()
        
        # 데이터 로딩
        logger.info("CSV 파일 로딩 중...")
        x_train_df = pd.read_csv(os.path.join(self.output_dir, 'x_train.csv'))
        logger.info(f"CSV 파일 로딩 완료: {len(x_train_df)}개 샘플")
        
        # 데이터 전처리
        logger.info("데이터 전처리 시작...")
        
        # 클래스 정보
        unique_labels = np.unique(self.labels)
        num_classes = len(unique_labels)
        logger.info(f"클래스 수: {num_classes}")
        logger.info(f"클래스 목록: {unique_labels}")
        
        # 데이터셋 생성
        dataset = CosmosImageDataset(x_train_df['image_url'].tolist(), 
                                   self.encoded_labels, self.transform)
        
        # 훈련/테스트 분할
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        logger.info(f"훈련 데이터: {len(train_dataset)}개")
        logger.info(f"테스트 데이터: {len(test_dataset)}개")
        
        # 모델 학습
        logger.info("모델 학습 시작...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"사용 디바이스: {device}")
        
        model = SimpleCNN(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 학습 루프
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        
        for epoch in range(5):
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
                
                if batch_idx % 1 == 0:
                    logger.info(f'Epoch {epoch+1}/5, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
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
            
            logger.info(f'Epoch {epoch+1}/5:')
            logger.info(f'  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'  Test Loss: {test_loss_avg:.4f}, Test Acc: {test_acc:.2f}%')
        
        # 모델 저장
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': num_classes,
            'label_encoder': self.label_encoder
        }, self.model_save_path)
        logger.info(f"모델 저장 완료: {self.model_save_path}")
        
        # 시각화
        logger.info("학습 결과 시각화 중...")
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.metrics_save_path)
        logger.info(f"시각화 결과 저장 완료: {self.metrics_save_path}")
        
        logger.info("파이프라인 실행 완료!")

class CosmosImageDataset(Dataset):
    """이미지 데이터셋 클래스"""
    
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
            # 이미지 다운로드
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # PIL 이미지로 변환
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            logger.warning(f"이미지 로딩 실패 (URL: {image_url}): {e}")
            # 실패시 더미 이미지 생성
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

def main():
    """메인 함수"""
    logger.info("개선된 이미지 스크래핑 파이프라인 시작")
    
    # 카테고리 정의
    categories = ['nature', 'animals', 'food', 'architecture', 'technology', 
                 'art', 'people', 'objects', 'abstract', 'korean_culture']
    
    # 개선된 스크래핑 실행
    scraper = ImprovedImageScraper()
    image_urls, labels = scraper.run_improved_scraping(categories)
    
    if not image_urls:
        logger.error("이미지 스크래핑 실패!")
        return
    
    # 이미지 분류 파이프라인 실행
    pipeline = ImageClassificationPipeline(
        image_urls=image_urls,
        labels=labels,
        output_dir="./dataset",
        model_save_path="./models/improved_model.pt",
        metrics_save_path="./results/improved_metrics.png"
    )
    
    pipeline.run()
    
    logger.info("개선된 파이프라인 실행 완료!")

if __name__ == "__main__":
    main()
