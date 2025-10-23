#!/usr/bin/env python3
"""
Automated Cosmos CDN Image Classification Model Training Pipeline

This script provides a fully automated pipeline that automatically scrapes images
from the cosmos.so CDN and trains an image classification model.

Key Features:
- Automated cosmos.so CDN scraping
- Automatic category labeling based on URL paths
- Image download and preprocessing
- CNN model training and evaluation
- Result visualization and model saving
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

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CosmosScraper:
    """cosmos.so CDN image scraper class"""
    
    def __init__(self, base_url: str = "https://cdn.cosmos.so", timeout: int = 10):
        """
        Args:
            base_url: cosmos CDN base URL
            timeout: request timeout (seconds)
        """
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Supported image extensions
        self.image_extensions = {'.webp', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        
    def scrape_image_urls(self, max_pages: int = 10) -> List[str]:
        """
        Scrapes image URLs from cosmos CDN.
        
        Args:
            max_pages: maximum number of pages to explore
            
        Returns:
            List[str]: list of discovered image URLs
        """
        logger.info(f"Starting cosmos CDN scraping: {self.base_url}")
        
        image_urls = []
        visited_urls = set()
        
        try:
            # Start from main page
            urls_to_visit = [self.base_url]
            
            for page_num in range(max_pages):
                if not urls_to_visit:
                    break
                    
                current_url = urls_to_visit.pop(0)
                if current_url in visited_urls:
                    continue
                    
                visited_urls.add(current_url)
                logger.info(f"Scraping page: {current_url}")
                
                try:
                    response = self.session.get(current_url, timeout=self.timeout)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find image links
                    page_image_urls = self._extract_image_urls(soup, current_url)
                    image_urls.extend(page_image_urls)
                    
                    # Find additional page links
                    new_urls = self._extract_page_urls(soup, current_url)
                    for url in new_urls:
                        if url not in visited_urls and url not in urls_to_visit:
                            urls_to_visit.append(url)
                            
                except Exception as e:
                    logger.warning(f"Page scraping failed ({current_url}): {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error occurred during scraping: {e}")
            
        # Remove duplicates
        image_urls = list(set(image_urls))
        logger.info(f"Total {len(image_urls)} image URLs found")
        
        return image_urls
    
    def _extract_image_urls(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs from HTML."""
        image_urls = []
        
        # Extract image URLs from img tags
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                full_url = urljoin(base_url, src)
                if self._is_image_url(full_url):
                    image_urls.append(full_url)
        
        # Extract image files from links
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            if self._is_image_url(full_url):
                image_urls.append(full_url)
        
        return image_urls
    
    def _extract_page_urls(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract page URLs for additional exploration."""
        page_urls = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Only add URLs within the same domain
            if self._is_same_domain(full_url) and not self._is_image_url(full_url):
                page_urls.append(full_url)
        
        return page_urls
    
    def _is_image_url(self, url: str) -> bool:
        """Check if URL is an image file."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check extension
        for ext in self.image_extensions:
            if path.endswith(ext):
                return True
        
        # Check URL pattern (cosmos CDN specific)
        if 'cosmos.so' in url and any(pattern in url for pattern in ['image', 'img', 'photo', 'pic']):
            return True
            
        return False
    
    def _is_same_domain(self, url: str) -> bool:
        """Check if it's the same domain."""
        parsed = urlparse(url)
        base_parsed = urlparse(self.base_url)
        return parsed.netloc == base_parsed.netloc
    
    def categorize_images(self, image_urls: List[str]) -> Dict[str, List[str]]:
        """
        Categorize image URLs by category.
        
        Args:
            image_urls: list of image URLs
            
        Returns:
            Dict[str, List[str]]: dictionary of image URLs by category
        """
        logger.info("Starting image category classification...")
        
        categorized = defaultdict(list)
        
        for url in image_urls:
            category = self._extract_category_from_url(url)
            categorized[category].append(url)
        
        # Print statistics by category
        for category, urls in categorized.items():
            logger.info(f"Category '{category}': {len(urls)} images")
        
        return dict(categorized)
    
    def _extract_category_from_url(self, url: str) -> str:
        """
        Extract category from URL.
        
        Args:
            url: image URL
            
        Returns:
            str: extracted category name
        """
        parsed = urlparse(url)
        path_parts = [part for part in parsed.path.split('/') if part]
        
        # Extract category from URL path
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
        
        # Find keywords in path
        for part in path_parts:
            part_lower = part.lower()
            for keyword, category in category_keywords.items():
                if keyword in part_lower:
                    return category
        
        # Extract category from filename
        filename = path_parts[-1] if path_parts else ""
        filename_lower = filename.lower()
        
        for keyword, category in category_keywords.items():
            if keyword in filename_lower:
                return category
        
        # Default category
        return 'general'

class CosmosImageDataset(Dataset):
    """cosmos CDN image dataset class"""
    
    def __init__(self, image_urls: List[str], labels: List[str], transform=None):
        """
        Args:
            image_urls: list of image URLs
            labels: list of labels for corresponding images
            transform: image transformation function
        """
        self.image_urls = image_urls
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_urls)
    
    def __getitem__(self, idx):
        """Return image and label corresponding to index"""
        try:
            # Download image
            response = requests.get(self.image_urls[idx], timeout=10)
            response.raise_for_status()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            # Apply transformation
            if self.transform:
                image = self.transform(image)
                
            return image, self.labels[idx]
            
        except Exception as e:
            logger.warning(f"Image loading failed (URL: {self.image_urls[idx]}): {e}")
            # Return dummy image if failed
            image = self._create_dummy_image()
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
    
    def _create_dummy_image(self):
        """Create dummy image (used when actual image is not available)"""
        # Create dummy image with random pattern
        dummy_image = Image.new('RGB', (224, 224), color='white')
        return dummy_image

class SimpleCNN(nn.Module):
    """Simple CNN image classification model"""
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: number of classes to classify
        """
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """Forward pass"""
        # First convolutional block
        x = self.pool(self.relu(self.conv1(x)))  # 224x224 -> 112x112
        
        # Second convolutional block
        x = self.pool(self.relu(self.conv2(x)))   # 112x112 -> 56x56
        
        # Third convolutional block
        x = self.pool(self.relu(self.conv3(x)))   # 56x56 -> 28x28
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class AutoScrapingPipeline:
    """Main class for automated scraping image classification pipeline"""
    
    def __init__(self, data_dir: str = "./dataset", model_dir: str = "./models", results_dir: str = "./results"):
        """
        Args:
            data_dir: data storage directory
            model_dir: model storage directory
            results_dir: results storage directory
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize scraper
        self.scraper = CosmosScraper()
        
        # Data storage variables
        self.x_train_data = None
        self.y_train_data = None
        self.label_encoder = None
        
    def scrape_and_categorize(self, max_pages: int = 10) -> Tuple[List[str], List[str]]:
        """
        Scrapes images from cosmos CDN and categorizes them.
        
        Args:
            max_pages: maximum number of pages to explore
            
        Returns:
            Tuple[List[str], List[str]]: (image URL list, label list)
        """
        logger.info("Starting image scraping and category classification...")
        
        # Scrape image URLs
        image_urls = self.scraper.scrape_image_urls(max_pages=max_pages)
        
        if not image_urls:
            logger.warning("No scraped images found. Using dummy data.")
            return self._create_dummy_data()
        
        # Categorize by category
        categorized_images = self.scraper.categorize_images(image_urls)
        
        # Compose data
        all_urls = []
        all_labels = []
        
        for category, urls in categorized_images.items():
            all_urls.extend(urls)
            all_labels.extend([category] * len(urls))
        
        logger.info(f"Total {len(all_urls)} image data prepared")
        logger.info(f"Distribution by category: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
        
        return all_urls, all_labels
    
    def _create_dummy_data(self) -> Tuple[List[str], List[str]]:
        """Create dummy data (used when scraping fails)"""
        logger.info("Creating dummy data...")
        
        # Use actual working image URLs (Unsplash etc.)
        dummy_data = {
            "book_layout": [
                "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400",  # book image
                "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # book image
                "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=400",   # book image
                "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=400", # book image
                "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400", # book image
            ],
            "photography": [
                "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400", # nature photo
                "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400", # nature photo
                "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=400", # nature photo
                "https://images.unsplash.com/photo-1447752875215-b2761acb3c5d?w=400", # nature photo
                "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400", # nature photo
            ],
            "design": [
                "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400",   # design image
                "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400",   # design image
                "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400",   # design image
                "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400",   # design image
                "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400",   # design image
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
        Save x_train and y_train data as CSV files.
        
        Args:
            image_urls: list of image URLs
            labels: list of labels
        """
        logger.info("Creating CSV files...")
        
        # Create x_train dataframe
        x_train_df = pd.DataFrame({
            'image_url': image_urls,
            'category': labels
        })
        
        # Create y_train dataframe
        y_train_df = pd.DataFrame({
            'label': labels
        })
        
        # Save CSV files
        x_train_path = self.data_dir / "x_train.csv"
        y_train_path = self.data_dir / "y_train.csv"
        
        x_train_df.to_csv(x_train_path, index=False)
        y_train_df.to_csv(y_train_path, index=False)
        
        logger.info(f"x_train.csv saved: {x_train_path}")
        logger.info(f"y_train.csv saved: {y_train_path}")
        
        # Store data
        self.x_train_data = x_train_df
        self.y_train_data = y_train_df
    
    def load_csv_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load saved CSV files.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (x_train, y_train) dataframes
        """
        logger.info("Loading CSV files...")
        
        x_train_path = self.data_dir / "x_train.csv"
        y_train_path = self.data_dir / "y_train.csv"
        
        if not x_train_path.exists() or not y_train_path.exists():
            raise FileNotFoundError("CSV files do not exist. Please run create_csv_files() first.")
        
        x_train_df = pd.read_csv(x_train_path)
        y_train_df = pd.read_csv(y_train_path)
        
        logger.info(f"CSV files loaded: {len(x_train_df)} samples")
        
        return x_train_df, y_train_df
    
    def preprocess_data(self, x_train_df: pd.DataFrame, y_train_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, int]:
        """
        Data preprocessing and dataloader creation
        
        Args:
            x_train_df: x_train dataframe
            y_train_df: y_train dataframe
            
        Returns:
            Tuple[DataLoader, DataLoader, int]: (train_loader, test_loader, num_classes)
        """
        logger.info("Starting data preprocessing...")
        
        # Label encoding
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(y_train_df['label'].values)
        num_classes = len(self.label_encoder.classes_)
        
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Class list: {self.label_encoder.classes_}")
        
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        dataset = CosmosImageDataset(
            x_train_df['image_url'].tolist(),
            encoded_labels.tolist(),
            transform=transform
        )
        
        # train/test split (8:2)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        logger.info(f"Training data: {len(train_dataset)} samples")
        logger.info(f"Test data: {len(test_dataset)} samples")
        
        return train_loader, test_loader, num_classes
    
    def train_model(self, train_loader: DataLoader, test_loader: DataLoader, num_classes: int, epochs: int = 10):
        """
        Model training
        
        Args:
            train_loader: training dataloader
            test_loader: test dataloader
            num_classes: number of classes
            epochs: number of training epochs
        """
        logger.info("Starting model training...")
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create model
        model = SimpleCNN(num_classes).to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training records
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Training mode
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
            
            # Calculate training accuracy
            train_accuracy = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Test evaluation
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
            
            # Save records
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(avg_test_loss)
            test_accuracies.append(test_accuracy)
            
            logger.info(f'Epoch {epoch+1}/{epochs}:')
            logger.info(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            logger.info(f'  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        
        # Save model
        model_path = self.model_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder': self.label_encoder,
            'num_classes': num_classes
        }, model_path)
        
        logger.info(f"Model saved: {model_path}")
        
        # Visualize training results
        self.plot_training_results(train_losses, train_accuracies, test_losses, test_accuracies)
        
        return model, train_losses, train_accuracies, test_losses, test_accuracies
    
    def plot_training_results(self, train_losses: List[float], train_accuracies: List[float], 
                            test_losses: List[float], test_accuracies: List[float]):
        """
        Visualize training results
        
        Args:
            train_losses: list of training losses
            train_accuracies: list of training accuracies
            test_losses: list of test losses
            test_accuracies: list of test accuracies
        """
        logger.info("Visualizing training results...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss graph
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(test_losses, label='Test Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy graph
        ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(test_accuracies, label='Test Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save graph
        metrics_path = self.results_dir / "metrics.png"
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Visualization results saved: {metrics_path}")
    
    def test_model(self, model_path: str, test_image_url: str = None):
        """
        Model testing and prediction
        
        Args:
            model_path: model file path
            test_image_url: test image URL (optional)
        """
        logger.info("Starting model testing...")
        
        # Load model (set weights_only=False to include LabelEncoder)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model = SimpleCNN(checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        label_encoder = checkpoint['label_encoder']
        
        # Test image transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if test_image_url:
            try:
                # Download and predict test image
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
                
                logger.info(f"Test image prediction results:")
                logger.info(f"  Predicted class: {predicted_label}")
                logger.info(f"  Confidence: {confidence:.4f}")
                
                return predicted_label, confidence
                
            except Exception as e:
                logger.error(f"Test image processing failed: {e}")
                return None, None
        else:
            logger.info("No test image URL provided.")
            return None, None

def main():
    """Main execution function"""
    logger.info("Starting automated scraping image classification pipeline")
    
    # Initialize pipeline
    pipeline = AutoScrapingPipeline()
    
    try:
        # 1. Scrape images from cosmos CDN and categorize
        image_urls, labels = pipeline.scrape_and_categorize(max_pages=5)
        
        # 2. Create CSV files
        pipeline.create_csv_files(image_urls, labels)
        
        # 3. Load CSV data
        x_train_df, y_train_df = pipeline.load_csv_data()
        
        # 4. Preprocess data
        train_loader, test_loader, num_classes = pipeline.preprocess_data(x_train_df, y_train_df)
        
        # 5. Train model
        model, train_losses, train_accuracies, test_losses, test_accuracies = pipeline.train_model(
            train_loader, test_loader, num_classes, epochs=10
        )
        
        # 6. Test model
        model_path = pipeline.model_dir / "model.pt"
        pipeline.test_model(str(model_path))
        
        logger.info("Pipeline execution completed!")
        
    except Exception as e:
        logger.error(f"Error occurred during pipeline execution: {e}")
        raise

if __name__ == "__main__":
    import io  # Import io module here
    main()
