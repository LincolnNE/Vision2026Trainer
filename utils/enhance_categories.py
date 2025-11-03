#!/usr/bin/env python3
"""
이미지 URL을 분석하여 적절한 카테고리를 자동으로 추가하는 스크립트
"""

import pandas as pd
import requests
from PIL import Image
import io
import re
from collections import Counter
import time

def analyze_image_content(image_url):
    """이미지 URL을 분석하여 적절한 카테고리 추천"""
    
    # URL 패턴 분석
    url_lower = image_url.lower()
    
    # 기본 카테고리 매핑
    category_mapping = {
        # 아키텍처 관련
        'architecture': ['architecture', 'building', 'design'],
        'building': ['architecture', 'building', 'urban'],
        'house': ['architecture', 'home', 'building'],
        'interior': ['interior', 'design', 'architecture'],
        
        # 아트 관련
        'art': ['art', 'creative', 'design'],
        'painting': ['art', 'painting', 'creative'],
        'drawing': ['art', 'drawing', 'creative'],
        'illustration': ['art', 'illustration', 'design'],
        
        # 사람 관련
        'people': ['people', 'portrait', 'human'],
        'portrait': ['people', 'portrait', 'human'],
        'face': ['people', 'portrait', 'human'],
        'person': ['people', 'portrait', 'human'],
        
        # 디자인 관련
        'design': ['design', 'creative', 'art'],
        'graphic': ['design', 'graphic', 'art'],
        'logo': ['design', 'logo', 'brand'],
        
        # 자연 관련
        'nature': ['nature', 'outdoor', 'landscape'],
        'landscape': ['nature', 'landscape', 'outdoor'],
        'tree': ['nature', 'tree', 'outdoor'],
        'flower': ['nature', 'flower', 'plant'],
        
        # 기술 관련
        'tech': ['technology', 'digital', 'modern'],
        'digital': ['technology', 'digital', 'modern'],
        'computer': ['technology', 'digital', 'modern'],
        
        # 음식 관련
        'food': ['food', 'cooking', 'restaurant'],
        'cooking': ['food', 'cooking', 'kitchen'],
        'restaurant': ['food', 'restaurant', 'dining'],
        
        # 패션 관련
        'fashion': ['fashion', 'style', 'clothing'],
        'clothing': ['fashion', 'clothing', 'style'],
        'style': ['fashion', 'style', 'design'],
        
        # 추상/일반
        'abstract': ['abstract', 'art', 'creative'],
        'pattern': ['pattern', 'design', 'abstract'],
        'texture': ['texture', 'pattern', 'abstract']
    }
    
    # URL에서 키워드 추출
    detected_categories = []
    
    for keyword, categories in category_mapping.items():
        if keyword in url_lower:
            detected_categories.extend(categories)
    
    # 기본 카테고리 (키워드가 없을 때)
    if not detected_categories:
        detected_categories = ['general', 'design', 'creative']
    
    # 중복 제거하고 상위 3개 선택
    unique_categories = list(dict.fromkeys(detected_categories))[:3]
    
    return unique_categories

def enhance_csv_with_categories(input_file, output_file):
    """CSV 파일의 카테고리를 향상시킴"""
    
    print(f"📖 CSV 파일 읽는 중: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"📊 총 {len(df)}개 이미지 발견")
    
    enhanced_data = []
    
    for index, row in df.iterrows():
        image_url = row['image_url']
        original_category = row['category']
        
        print(f"🔍 이미지 {index+1}/{len(df)} 분석 중...")
        print(f"   URL: {image_url[:60]}...")
        
        # 이미지 분석하여 카테고리 추천
        recommended_categories = analyze_image_content(image_url)
        
        # 기존 카테고리와 추천 카테고리 결합
        all_categories = [original_category] + recommended_categories
        unique_categories = list(dict.fromkeys(all_categories))[:3]  # 중복 제거하고 최대 3개
        
        # 콤마로 구분된 문자열로 변환
        enhanced_category = ', '.join(unique_categories)
        
        enhanced_data.append({
            'image_url': image_url,
            'category': enhanced_category
        })
        
        print(f"   ✅ 카테고리: {enhanced_category}")
        print()
        
        # 요청 간격 조절 (서버 부하 방지)
        time.sleep(0.1)
    
    # 새로운 DataFrame 생성
    enhanced_df = pd.DataFrame(enhanced_data)
    
    # CSV 파일 저장
    enhanced_df.to_csv(output_file, index=False)
    
    print(f"💾 향상된 CSV 파일 저장 완료: {output_file}")
    
    # 통계 출력
    all_categories = []
    for category_string in enhanced_df['category']:
        categories = [cat.strip() for cat in category_string.split(',')]
        all_categories.extend(categories)
    
    category_counts = Counter(all_categories)
    
    print(f"\n📈 카테고리 통계:")
    for category, count in category_counts.most_common():
        print(f"   {category}: {count}개")
    
    return enhanced_df

def main():
    """메인 함수"""
    input_file = "dataset/x_train.csv"
    output_file = "dataset/x_train_enhanced.csv"
    
    try:
        enhanced_df = enhance_csv_with_categories(input_file, output_file)
        print(f"\n🎉 카테고리 향상 완료!")
        print(f"   입력 파일: {input_file}")
        print(f"   출력 파일: {output_file}")
        print(f"   총 {len(enhanced_df)}개 이미지 처리됨")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
