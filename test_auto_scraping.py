#!/usr/bin/env python3
"""
Automated Scraping Image Classification Pipeline Test Script

This script tests each component of the automated scraping pipeline individually.
"""

import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from auto_scraping_pipeline import AutoScrapingPipeline, CosmosScraper, CosmosImageDataset, SimpleCNN
import torch
import pandas as pd

def test_scraper_initialization():
    """스크래퍼 초기화 테스트"""
    print("=== 스크래퍼 초기화 테스트 ===")
    try:
        scraper = CosmosScraper()
        print("✓ 스크래퍼 초기화 성공")
        print(f"  - 기본 URL: {scraper.base_url}")
        print(f"  - 타임아웃: {scraper.timeout}초")
        print(f"  - 지원 확장자: {scraper.image_extensions}")
        return True
    except Exception as e:
        print(f"✗ 스크래퍼 초기화 실패: {e}")
        return False

def test_url_categorization():
    """URL 카테고리 분류 테스트"""
    print("\n=== URL 카테고리 분류 테스트 ===")
    try:
        scraper = CosmosScraper()
        
        # 테스트 URL들
        test_urls = [
            "https://cdn.cosmos.so/book/layout/book1.webp",
            "https://cdn.cosmos.so/photo/minimal/photo1.webp",
            "https://cdn.cosmos.so/design/creative/design1.webp",
            "https://cdn.cosmos.so/art/abstract/art1.webp",
            "https://cdn.cosmos.so/texture/pattern/texture1.webp"
        ]
        
        categorized = scraper.categorize_images(test_urls)
        
        print(f"✓ URL 카테고리 분류 성공")
        print(f"  - 입력 URL 수: {len(test_urls)}")
        print(f"  - 분류된 카테고리 수: {len(categorized)}")
        
        for category, urls in categorized.items():
            print(f"  - {category}: {len(urls)}개")
        
        return True
    except Exception as e:
        print(f"✗ URL 카테고리 분류 실패: {e}")
        return False

def test_pipeline_initialization():
    """파이프라인 초기화 테스트"""
    print("\n=== 파이프라인 초기화 테스트 ===")
    try:
        pipeline = AutoScrapingPipeline()
        print("✓ 파이프라인 초기화 성공")
        print(f"  - 데이터 디렉토리: {pipeline.data_dir}")
        print(f"  - 모델 디렉토리: {pipeline.model_dir}")
        print(f"  - 결과 디렉토리: {pipeline.results_dir}")
        return True
    except Exception as e:
        print(f"✗ 파이프라인 초기화 실패: {e}")
        return False

def test_dummy_data_creation():
    """더미 데이터 생성 테스트"""
    print("\n=== 더미 데이터 생성 테스트 ===")
    try:
        pipeline = AutoScrapingPipeline()
        image_urls, labels = pipeline._create_dummy_data()
        
        print(f"✓ 더미 데이터 생성 성공")
        print(f"  - 총 이미지 수: {len(image_urls)}")
        print(f"  - 총 라벨 수: {len(labels)}")
        print(f"  - 고유 카테고리: {set(labels)}")
        
        # 데이터 타입 확인
        assert isinstance(image_urls, list), "image_urls는 리스트여야 합니다"
        assert isinstance(labels, list), "labels는 리스트여야 합니다"
        assert len(image_urls) == len(labels), "이미지와 라벨의 개수가 일치해야 합니다"
        
        return True
    except Exception as e:
        print(f"✗ 더미 데이터 생성 실패: {e}")
        return False

def test_csv_creation():
    """CSV 파일 생성 테스트"""
    print("\n=== CSV 파일 생성 테스트 ===")
    try:
        pipeline = AutoScrapingPipeline()
        image_urls, labels = pipeline._create_dummy_data()
        pipeline.create_csv_files(image_urls, labels)
        
        # CSV 파일 존재 확인
        x_train_path = pipeline.data_dir / "x_train.csv"
        y_train_path = pipeline.data_dir / "y_train.csv"
        
        assert x_train_path.exists(), "x_train.csv 파일이 생성되지 않았습니다"
        assert y_train_path.exists(), "y_train.csv 파일이 생성되지 않았습니다"
        
        # CSV 내용 확인
        x_train_df = pd.read_csv(x_train_path)
        y_train_df = pd.read_csv(y_train_path)
        
        print(f"✓ CSV 파일 생성 성공")
        print(f"  - x_train.csv: {len(x_train_df)}행, 컬럼: {list(x_train_df.columns)}")
        print(f"  - y_train.csv: {len(y_train_df)}행, 컬럼: {list(y_train_df.columns)}")
        
        return True
    except Exception as e:
        print(f"✗ CSV 파일 생성 실패: {e}")
        return False

def test_model_creation():
    """모델 생성 테스트"""
    print("\n=== 모델 생성 테스트 ===")
    try:
        num_classes = 3  # 더미 데이터의 클래스 수
        model = SimpleCNN(num_classes)
        
        # 모델 구조 확인
        print(f"✓ 모델 생성 성공")
        print(f"  - 클래스 수: {num_classes}")
        print(f"  - 모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")
        
        # 더미 입력으로 순전파 테스트
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        
        assert output.shape == (1, num_classes), f"출력 형태가 예상과 다릅니다: {output.shape}"
        print(f"  - 출력 형태: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 모델 생성 실패: {e}")
        return False

def test_dataset_creation():
    """데이터셋 생성 테스트"""
    print("\n=== 데이터셋 생성 테스트 ===")
    try:
        # 더미 데이터로 테스트
        dummy_urls = ["https://cdn.cosmos.so/test1.webp", "https://cdn.cosmos.so/test2.webp"]
        dummy_labels = [0, 1]
        
        # 변환 없이 데이터셋 생성
        dataset = CosmosImageDataset(dummy_urls, dummy_labels, transform=None)
        
        print(f"✓ 데이터셋 생성 성공")
        print(f"  - 데이터셋 크기: {len(dataset)}")
        
        return True
    except Exception as e:
        print(f"✗ 데이터셋 생성 실패: {e}")
        return False

def test_image_url_detection():
    """이미지 URL 감지 테스트"""
    print("\n=== 이미지 URL 감지 테스트 ===")
    try:
        scraper = CosmosScraper()
        
        # 테스트 URL들
        test_urls = [
            "https://cdn.cosmos.so/image.webp",  # 이미지
            "https://cdn.cosmos.so/image.png",   # 이미지
            "https://cdn.cosmos.so/image.jpg",   # 이미지
            "https://cdn.cosmos.so/page.html",   # 이미지 아님
            "https://cdn.cosmos.so/data.json",   # 이미지 아님
        ]
        
        expected_results = [True, True, True, False, False]
        
        print(f"✓ 이미지 URL 감지 테스트 시작")
        
        for url, expected in zip(test_urls, expected_results):
            result = scraper._is_image_url(url)
            status = "✓" if result == expected else "✗"
            print(f"  {status} {url}: {result} (예상: {expected})")
            
            if result != expected:
                return False
        
        return True
    except Exception as e:
        print(f"✗ 이미지 URL 감지 테스트 실패: {e}")
        return False

def run_all_tests():
    """모든 테스트 실행"""
    print("자동 스크래핑 이미지 분류 파이프라인 테스트 시작\n")
    
    tests = [
        test_scraper_initialization,
        test_url_categorization,
        test_pipeline_initialization,
        test_dummy_data_creation,
        test_csv_creation,
        test_model_creation,
        test_dataset_creation,
        test_image_url_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== 테스트 결과 ===")
    print(f"통과: {passed}/{total}")
    
    if passed == total:
        print("✓ 모든 테스트 통과! 자동 스크래핑 파이프라인이 정상적으로 작동합니다.")
        return True
    else:
        print("✗ 일부 테스트 실패. 코드를 확인해주세요.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
