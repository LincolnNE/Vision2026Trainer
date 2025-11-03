#!/usr/bin/env python3
"""
실제 Cosmos.so CDN 이미지 스크래핑 파이프라인
- cosmos.so에서 실제 이미지 수집
- URL 패턴 기반 카테고리 분류
- 고품질 이미지 검증
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
import re
import time
from urllib.parse import urljoin, urlparse

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CosmosRealScraper:
    """실제 Cosmos.so CDN 이미지 스크래퍼"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Cosmos.so의 실제 이미지 URL 패턴들
        self.cosmos_url_patterns = [
            'https://cdn.cosmos.so/',
            'https://cosmos.so/',
            'https://assets.cosmos.so/',
        ]
        
        # URL 경로 기반 카테고리 매핑
        self.path_category_mapping = {
            'nature': ['nature', 'landscape', 'forest', 'mountain', 'ocean', 'sky', 'tree', 'flower'],
            'animals': ['animal', 'pet', 'wildlife', 'cat', 'dog', 'bird', 'fish'],
            'food': ['food', 'meal', 'restaurant', 'cooking', 'kitchen', 'dish'],
            'architecture': ['building', 'house', 'city', 'architecture', 'bridge', 'tower'],
            'technology': ['tech', 'computer', 'digital', 'software', 'hardware', 'device'],
            'art': ['art', 'painting', 'gallery', 'museum', 'design', 'creative'],
            'people': ['person', 'portrait', 'face', 'people', 'human', 'portrait'],
            'objects': ['object', 'item', 'product', 'tool', 'furniture'],
            'abstract': ['abstract', 'pattern', 'texture', 'geometric', 'minimal'],
            'korean_culture': ['korea', 'korean', 'seoul', 'hanbok', 'kimchi', 'traditional']
        }

    def _extract_cosmos_images(self, base_url: str) -> List[str]:
        """Cosmos.so에서 이미지 URL 추출"""
        image_urls = []
        
        try:
            logger.info(f"Cosmos.so 스크래핑 시작: {base_url}")
            response = requests.get(base_url, headers=self.headers, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 다양한 이미지 태그 찾기
                img_tags = soup.find_all('img')
                
                for img in img_tags:
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src:
                        # 상대 URL을 절대 URL로 변환
                        full_url = urljoin(base_url, src)
                        
                        # Cosmos.so 도메인 확인
                        if any(pattern in full_url for pattern in self.cosmos_url_patterns):
                            # 이미지 확장자 확인
                            if any(ext in full_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                                image_urls.append(full_url)
                
                logger.info(f"발견된 이미지 URL: {len(image_urls)}개")
                
        except Exception as e:
            logger.warning(f"Cosmos.so 스크래핑 실패 ({base_url}): {e}")
        
        return image_urls

    def _categorize_url(self, url: str) -> str:
        """URL 경로를 기반으로 카테고리 분류"""
        url_lower = url.lower()
        
        # URL 경로에서 카테고리 키워드 찾기
        for category, keywords in self.path_category_mapping.items():
            for keyword in keywords:
                if keyword in url_lower:
                    return category
        
        # 기본 카테고리
        return 'general'

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

    def scrape_cosmos_images(self) -> Tuple[List[str], List[str]]:
        """Cosmos.so에서 실제 이미지 스크래핑"""
        logger.info("Cosmos.so 실제 이미지 스크래핑 시작...")
        
        all_image_urls = []
        all_labels = []
        
        # Cosmos.so의 다양한 페이지 스크래핑 시도
        cosmos_pages = [
            'https://cosmos.so/',
            'https://cosmos.so/gallery',
            'https://cosmos.so/images',
            'https://cosmos.so/photos',
            'https://cosmos.so/portfolio',
        ]
        
        for page_url in cosmos_pages:
            try:
                image_urls = self._extract_cosmos_images(page_url)
                
                for url in image_urls:
                    if self._validate_image_url(url):
                        category = self._categorize_url(url)
                        all_image_urls.append(url)
                        all_labels.append(category)
                        logger.info(f"유효한 이미지 발견: {url} -> {category}")
                
                # 요청 간격 조절
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"페이지 스크래핑 실패 ({page_url}): {e}")
        
        # 중복 제거
        unique_pairs = list(set(zip(all_image_urls, all_labels)))
        if unique_pairs:
            all_image_urls, all_labels = zip(*unique_pairs)
            all_image_urls, all_labels = list(all_image_urls), list(all_labels)
        
        logger.info(f"총 {len(all_image_urls)}개의 Cosmos.so 이미지 수집 완료")
        
        # 카테고리별 분포 출력
        from collections import Counter
        category_counts = Counter(all_labels)
        for category, count in category_counts.items():
            logger.info(f"카테고리 '{category}': {count}개 이미지")
        
        return all_image_urls, all_labels

    def create_fallback_dataset(self) -> Tuple[List[str], List[str]]:
        """스크래핑 실패시 대체 데이터셋 생성"""
        logger.info("Cosmos.so 스크래핑 실패 - 대체 데이터셋 생성...")
        
        # 실제 작동하는 다양한 소스의 이미지들
        fallback_data = {
            'nature': [
                'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800',
                'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800',
                'https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800',
            ],
            'animals': [
                'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800',
                'https://images.unsplash.com/photo-1552053831-71594a27632d?w=800',
                'https://images.unsplash.com/photo-1444464666168-49d633b86797?w=800',
            ],
            'food': [
                'https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=800',
                'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=800',
                'https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=800',
            ],
            'architecture': [
                'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800',
                'https://images.unsplash.com/photo-1511818966892-d7d671e672a2?w=800',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800',
            ],
            'technology': [
                'https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=800',
                'https://images.unsplash.com/photo-1512941937669-90a1b58e7e9c?w=800',
                'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=800',
            ],
            'art': [
                'https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=800',
                'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=800',
                'https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=800',
            ],
            'people': [
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800',
            ],
            'objects': [
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800',
            ],
            'abstract': [
                'https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=800',
                'https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=800',
                'https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=800',
            ],
            'korean_culture': [
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800',
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800',
            ]
        }
        
        all_urls = []
        all_labels = []
        
        for category, urls in fallback_data.items():
            all_urls.extend(urls)
            all_labels.extend([category] * len(urls))
        
        logger.info(f"대체 데이터셋 생성 완료: {len(all_urls)}개 이미지")
        return all_urls, all_labels

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
    logger.info("실제 Cosmos.so CDN 이미지 스크래핑 파이프라인 시작")
    
    # Cosmos.so 실제 스크래핑 시도
    scraper = CosmosRealScraper()
    image_urls, labels = scraper.scrape_cosmos_images()
    
    # 스크래핑 실패시 대체 데이터셋 사용
    if not image_urls:
        logger.warning("Cosmos.so 스크래핑 실패 - 대체 데이터셋 사용")
        image_urls, labels = scraper.create_fallback_dataset()
    
    # 이미지 분류 파이프라인 실행
    pipeline = ImageClassificationPipeline(
        image_urls=image_urls,
        labels=labels,
        output_dir="./dataset",
        model_save_path="./models/cosmos_model.pt",
        metrics_save_path="./results/cosmos_metrics.png"
    )
    
    pipeline.run()
    
    logger.info("Cosmos.so 파이프라인 실행 완료!")

if __name__ == "__main__":
    main()
