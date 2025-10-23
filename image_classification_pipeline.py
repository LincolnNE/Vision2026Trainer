#!/usr/bin/env python3
"""
Cosmos CDN 이미지 분류 모델 학습 파이프라인

이 스크립트는 cosmos CDN의 이미지 데이터를 기반으로 
이미지 분류 모델을 학습하는 완전한 파이프라인을 제공합니다.

주요 기능:
- cosmos CDN 이미지 URL 데이터 로딩
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
from typing import List, Tuple, Dict
import logging
from pathlib import Path
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CosmosImageDataset(Dataset):
    """cosmos CDN 이미지 데이터셋 클래스"""
    
    def __init__(self, image_paths: List[str], labels: List[str], transform=None, base_url: str = ""):
        """
        Args:
            image_paths: 이미지 파일 경로 리스트 (cosmos_cdn/...)
            labels: 해당 이미지의 라벨 리스트
            transform: 이미지 변환 함수
            base_url: 기본 URL (선택사항)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.base_url = base_url
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """인덱스에 해당하는 이미지와 라벨을 반환"""
        try:
            image_path = self.image_paths[idx]
            
            # 로컬 파일인지 URL인지 확인
            if image_path.startswith('http'):
                # URL인 경우 다운로드
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            else:
                # 로컬 파일 경로인 경우
                full_path = os.path.join(self.base_url, image_path) if self.base_url else image_path
                
                if os.path.exists(full_path):
                    image = Image.open(full_path).convert('RGB')
                else:
                    # 파일이 없는 경우 더미 이미지 생성
                    logger.warning(f"이미지 파일을 찾을 수 없습니다: {full_path}")
                    image = self._create_dummy_image()
            
            # 변환 적용
            if self.transform:
                image = self.transform(image)
                
            return image, self.labels[idx]
            
        except Exception as e:
            logger.warning(f"이미지 로딩 실패 (경로: {self.image_paths[idx]}): {e}")
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

class ImageClassificationPipeline:
    """이미지 분류 파이프라인 메인 클래스"""
    
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
        
        # 데이터 저장 변수
        self.x_train_data = None
        self.y_train_data = None
        self.label_encoder = None
        
    def load_cosmos_data(self) -> Tuple[List[str], List[str]]:
        """
        cosmos CDN 이미지 데이터를 로딩합니다.
        
        Returns:
            Tuple[List[str], List[str]]: (이미지 파일명 리스트, 라벨 리스트)
        """
        logger.info("cosmos CDN 데이터 로딩 시작...")
        
        # 실제 cosmos CDN 이미지 데이터 구조
        cosmos_data = {
            "artbook_layout": [
                "cosmos_cdn/book_spread_001.webp",
                "cosmos_cdn/book_spread_002.webp",
                "cosmos_cdn/book_spread_003.webp",
                "cosmos_cdn/book_spread_004.webp",
                "cosmos_cdn/book_spread_005.webp",
                "cosmos_cdn/book_spread_006.webp",
                "cosmos_cdn/book_spread_007.webp",
                "cosmos_cdn/book_spread_008.webp",
            ],
            "photobook_minimal_blackwhite": [
                "cosmos_cdn/photo_minimal_001.webp",
                "cosmos_cdn/photo_minimal_002.webp",
                "cosmos_cdn/photo_minimal_003.webp",
                "cosmos_cdn/photo_minimal_004.webp",
                "cosmos_cdn/photo_minimal_005.webp",
                "cosmos_cdn/photo_minimal_006.webp",
                "cosmos_cdn/photo_minimal_007.webp",
                "cosmos_cdn/photo_minimal_008.webp",
            ],
            "magazine_layout": [
                "cosmos_cdn/magazine_001.webp",
                "cosmos_cdn/magazine_002.webp",
                "cosmos_cdn/magazine_003.webp",
                "cosmos_cdn/magazine_004.webp",
                "cosmos_cdn/magazine_005.webp",
                "cosmos_cdn/magazine_006.webp",
                "cosmos_cdn/magazine_007.webp",
                "cosmos_cdn/magazine_008.webp",
            ],
            "portfolio_creative": [
                "cosmos_cdn/portfolio_001.webp",
                "cosmos_cdn/portfolio_002.webp",
                "cosmos_cdn/portfolio_003.webp",
                "cosmos_cdn/portfolio_004.webp",
                "cosmos_cdn/portfolio_005.webp",
                "cosmos_cdn/portfolio_006.webp",
                "cosmos_cdn/portfolio_007.webp",
                "cosmos_cdn/portfolio_008.webp",
            ]
        }
        
        # 데이터 구성
        image_paths = []
        labels = []
        
        for category, paths in cosmos_data.items():
            image_paths.extend(paths)
            labels.extend([category] * len(paths))
        
        logger.info(f"총 {len(image_paths)}개의 이미지 데이터 로딩 완료")
        logger.info(f"카테고리별 분포: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return image_paths, labels
    
    def create_csv_files(self, image_paths: List[str], labels: List[str]):
        """
        x_train과 y_train 데이터를 CSV 파일로 저장합니다.
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            labels: 라벨 리스트
        """
        logger.info("CSV 파일 생성 중...")
        
        # x_train 데이터프레임 생성
        x_train_df = pd.DataFrame({
            'image_url': image_paths,
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
        
        # 모델 로딩
        checkpoint = torch.load(model_path, map_location='cpu')
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
    logger.info("이미지 분류 파이프라인 시작")
    
    # 파이프라인 초기화
    pipeline = ImageClassificationPipeline()
    
    try:
        # 1. cosmos 데이터 로딩
        image_paths, labels = pipeline.load_cosmos_data()
        
        # 2. CSV 파일 생성
        pipeline.create_csv_files(image_paths, labels)
        
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
