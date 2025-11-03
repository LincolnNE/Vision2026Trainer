#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import logging
import pandas as pd
import requests
from PIL import Image
import io
import threading
import time
from datetime import datetime
import glob
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import functools
from dotenv import load_dotenv
import google.generativeai as genai

# PyQt6 imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                               QPushButton, QTextEdit, QListWidget, QListWidgetItem,
                               QCheckBox, QProgressBar, QSplitter, QFrame, QScrollArea,
                               QDialog, QDialogButtonBox, QMessageBox, QFileDialog,
                               QGroupBox, QComboBox, QSpinBox, QSlider)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPalette, QColor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cosmos_gui.log'),
        logging.StreamHandler()
    ]
)

class GeminiAPIClient:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        # Use the correct endpoint for Gemini API
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash"
        self.max_workers = min(4, multiprocessing.cpu_count())
        self.session = requests.Session()
        
        # Try to set CPU priority (may require admin privileges)
        try:
            os.nice(-5)
        except PermissionError:
            logging.warning("CPU 우선순위 설정 권한 없음")
        
        logging.info(f"🔧 하드웨어 가속 활성화: {self.max_workers}개 워커")

    def analyze_image(self, image_url):
        """Analyze a single image using Gemini API"""
        if not self.api_key:
            return "API 키가 설정되지 않음"
        
        try:
            # Configure Gemini API with the API key
            genai.configure(api_key=self.api_key)
            
            # Download image
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = self.session.get(image_url, timeout=10, headers=headers)
            response.raise_for_status()
            
            # Create PIL Image from downloaded content
            image = Image.open(io.BytesIO(response.content))
            
            # Initialize the model - use gemini-flash-latest which supports vision
            model = genai.GenerativeModel('gemini-flash-latest')
            
            # Create prompt
            prompt = """이미지의 카테고리를 추천해주세요.

다음 형식으로 답변해주세요:
영문주요카테고리1, 영문주요카테고리2, 영문주요카테고리3 | 한국어주요카테고리1, 한국어주요카테고리2, 한국어주요카테고리3

예시:
nature, landscape, mountain | 자연, 풍경, 산
animal, cat, pet | 동물, 고양이, 반려동물
food, dessert, cake | 음식, 디저트, 케이크"""
            
            # Generate content using Gemini API
            result = model.generate_content([prompt, image])
            
            if result and hasattr(result, 'text'):
                logging.info(f"Gemini 분석 결과: {result.text}")
                return result.text.strip()
            else:
                return "분석 결과를 찾을 수 없음"
                
        except requests.exceptions.RequestException as e:
            logging.error(f"네트워크 오류: {e}")
            return f"네트워크 오류: {str(e)}"
        except Exception as e:
            logging.error(f"분석 오류: {e}")
            return f"분석 오류: {str(e)}"

    def batch_analyze_images(self, image_urls, progress_callback=None):
        """Analyze multiple images in parallel"""
        results = []
        
        def _analyze_single_image(index, url):
            # Sequential delay to distribute API load
            time.sleep(index * 2.0)
            result = self.analyze_image(url)
            return index, result
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(_analyze_single_image, i, url) 
                      for i, url in enumerate(image_urls)]
            
            # Collect results as they complete
            for i, future in enumerate(futures):
                try:
                    index, result = future.result()
                    results.append((index, result))
                    
                    if progress_callback:
                        progress_callback(i + 1, len(image_urls))
                        
                except Exception as e:
                    logging.error(f"이미지 {i} 분석 실패: {e}")
                    results.append((i, f"분석 실패: {str(e)}"))
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

class ImageAnalysisWorker(QThread):
    progress_updated = pyqtSignal(int, int)  # current, total
    analysis_completed = pyqtSignal(list)   # results
    error_occurred = pyqtSignal(str)        # error message
    
    def __init__(self, image_urls, gemini_client):
        super().__init__()
        self.image_urls = image_urls
        self.gemini_client = gemini_client
    
    def run(self):
        try:
            def progress_callback(current, total):
                self.progress_updated.emit(current, total)
            
            results = self.gemini_client.batch_analyze_images(
                self.image_urls, progress_callback
            )
            self.analysis_completed.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))

class BulkAnalysisWorker(QThread):
    progress_updated = pyqtSignal(int, int)  # current, total
    analysis_completed = pyqtSignal(list)   # results as [(url, category), ...]
    error_occurred = pyqtSignal(str)        # error message
    
    def __init__(self, image_urls, gemini_client):
        super().__init__()
        self.image_urls = image_urls
        self.gemini_client = gemini_client
    
    def run(self):
        try:
            results = []
            
            def progress_callback(current, total):
                self.progress_updated.emit(current, total)
            
            # Analyze images
            categories = self.gemini_client.batch_analyze_images(
                self.image_urls, progress_callback
            )
            
            # Combine URLs with categories
            for url, category in zip(self.image_urls, categories):
                results.append((url, category))
            
            self.analysis_completed.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))

class CosmosGUIPyQt6(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_urls = []
        self.labels = []
        self.gemini_client = GeminiAPIClient()
        self.preview_width = 300
        self.preview_height = 200
        self.is_resizing = False
        
        self.init_ui()
        self.load_existing_data()
        
    def init_ui(self):
        self.setWindowTitle("Cosmos Image Classification GUI - PyQt6")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (image list and controls)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel (image preview)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([1000, 400])
        
    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title_label = QLabel("Cosmos Image Classification")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Image input section
        input_group = QGroupBox("이미지 링크 입력")
        input_layout = QVBoxLayout(input_group)
        
        # Single image input
        single_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("이미지 URL을 입력하세요...")
        single_layout.addWidget(self.url_input)
        
        add_button = QPushButton("추가")
        add_button.clicked.connect(self.add_single_image)
        single_layout.addWidget(add_button)
        
        input_layout.addLayout(single_layout)
        
        # Bulk add button
        bulk_button = QPushButton("일괄 추가")
        bulk_button.clicked.connect(self.bulk_add_images)
        input_layout.addWidget(bulk_button)
        
        layout.addWidget(input_group)
        
        # Image list section
        list_group = QGroupBox("이미지 목록")
        list_layout = QVBoxLayout(list_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        select_all_button = QPushButton("전체 선택")
        select_all_button.clicked.connect(self.select_all_images)
        control_layout.addWidget(select_all_button)
        
        delete_button = QPushButton("선택 삭제")
        delete_button.clicked.connect(self.delete_selected_images)
        control_layout.addWidget(delete_button)
        
        list_layout.addLayout(control_layout)
        
        # Image list
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.image_list.currentRowChanged.connect(self.load_image_preview)
        self.image_list.itemSelectionChanged.connect(self.on_selection_changed)
        list_layout.addWidget(self.image_list)
        
        layout.addWidget(list_group)
        
        # AI Analysis section
        ai_group = QGroupBox("AI 분석")
        ai_layout = QVBoxLayout(ai_group)
        
        # Gemini API status
        self.gemini_status_label = QLabel("Gemini API 상태: 확인 중...")
        ai_layout.addWidget(self.gemini_status_label)
        
        # Analysis mode toggle
        mode_layout = QHBoxLayout()
        self.analyze_selected_only = QCheckBox("선택 항목만 분석")
        self.analyze_selected_only.setChecked(False)
        mode_layout.addWidget(self.analyze_selected_only)
        mode_layout.addStretch()
        ai_layout.addLayout(mode_layout)
        
        # Performance mode
        perf_layout = QHBoxLayout()
        self.perf_checkbox = QCheckBox("고성능 모드 (80% CPU)")
        self.perf_checkbox.setChecked(True)
        self.perf_checkbox.toggled.connect(self.toggle_performance_mode)
        perf_layout.addWidget(self.perf_checkbox)
        
        perf_info_label = QLabel("하드웨어 가속")
        perf_info_label.setStyleSheet("color: #666; font-size: 10px;")
        perf_layout.addWidget(perf_info_label)
        perf_layout.addStretch()
        
        ai_layout.addLayout(perf_layout)
        
        # Analysis button
        analyze_button = QPushButton("AI 카테고리 분석 시작")
        analyze_button.clicked.connect(self.start_ai_analysis)
        ai_layout.addWidget(analyze_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        ai_layout.addWidget(self.progress_bar)
        
        layout.addWidget(ai_group)
        
        # CSV operations section
        csv_group = QGroupBox("CSV 작업")
        csv_layout = QHBoxLayout(csv_group)
        
        save_button = QPushButton("CSV 저장")
        save_button.clicked.connect(self.save_csv)
        csv_layout.addWidget(save_button)
        
        new_csv_button = QPushButton("CSV 새로 쓰기")
        new_csv_button.clicked.connect(self.create_new_csv)
        csv_layout.addWidget(new_csv_button)
        
        layout.addWidget(csv_group)
        
        # API Key management
        api_group = QGroupBox("API 키 관리")
        api_layout = QVBoxLayout(api_group)
        
        # API key status
        self.api_key_label = QLabel("API 키: 확인 중...")
        api_layout.addWidget(self.api_key_label)
        
        # API key buttons
        api_button_layout = QHBoxLayout()
        
        manage_key_button = QPushButton("API 키 관리")
        manage_key_button.clicked.connect(self.manage_api_key)
        api_button_layout.addWidget(manage_key_button)
        
        delete_key_button = QPushButton("API 키 삭제")
        delete_key_button.clicked.connect(self.delete_api_key)
        api_button_layout.addWidget(delete_key_button)
        
        test_button = QPushButton("연결 테스트")
        test_button.clicked.connect(self.test_gemini_connection)
        api_button_layout.addWidget(test_button)
        
        api_layout.addLayout(api_button_layout)
        
        layout.addWidget(api_group)
        
        layout.addStretch()
        
        # Update API status
        self.update_api_status()
        
        return panel
    
    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Preview title
        preview_label = QLabel("이미지 미리보기")
        preview_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(preview_label)
        
        # Image preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        self.preview_label.setMinimumSize(300, 200)
        self.preview_label.setText("이미지를 선택하세요")
        layout.addWidget(self.preview_label)
        
        # Resize handle
        resize_frame = QFrame()
        resize_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        resize_layout = QHBoxLayout(resize_frame)
        
        resize_label = QLabel("크기 조절")
        resize_label.setStyleSheet("color: #666; font-size: 10px;")
        resize_layout.addWidget(resize_label)
        
        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider.setRange(200, 600)
        self.width_slider.setValue(300)
        self.width_slider.valueChanged.connect(self.resize_preview)
        resize_layout.addWidget(self.width_slider)
        
        layout.addWidget(resize_frame)
        
        layout.addStretch()
        
        return panel
    
    def add_single_image(self):
        url = self.url_input.text().strip()
        if url:
            self.image_urls.append(url)
            self.labels.append("")
            self.update_image_list()
            self.url_input.clear()
            self.auto_save_data()
            QMessageBox.information(self, "성공", "이미지가 추가되었습니다!")
        else:
            QMessageBox.warning(self, "경고", "URL을 입력해주세요!")
    
    def bulk_add_images(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("일괄 이미지 추가")
        dialog.setModal(True)
        dialog.resize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Instructions
        instructions = QLabel("여러 이미지 URL을 한 줄에 하나씩 입력하세요:")
        layout.addWidget(instructions)
        
        # Text area
        text_area = QTextEdit()
        text_area.setPlaceholderText("https://example.com/image1.jpg\nhttps://example.com/image2.jpg\n...")
        layout.addWidget(text_area)
        
        # Progress bar for AI analysis
        progress_label = QLabel("AI 분석 진행률:")
        progress_label.setVisible(False)
        layout.addWidget(progress_label)
        
        progress_bar = QProgressBar()
        progress_bar.setVisible(False)
        layout.addWidget(progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ai_analyze_button = QPushButton("AI 분석 후 추가")
        ai_analyze_button.clicked.connect(lambda: self.bulk_add_with_ai_analysis(dialog, text_area, progress_label, progress_bar))
        button_layout.addWidget(ai_analyze_button)
        
        simple_add_button = QPushButton("단순 추가")
        simple_add_button.clicked.connect(lambda: self.bulk_add_simple(dialog, text_area))
        button_layout.addWidget(simple_add_button)
        
        cancel_button = QPushButton("취소")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def bulk_add_simple(self, dialog, text_area):
        """Simple bulk add without AI analysis"""
        urls = text_area.toPlainText().strip().split('\n')
        urls = [url.strip() for url in urls if url.strip()]
        
        if urls:
            self.image_urls.extend(urls)
            self.labels.extend([""] * len(urls))
            self.update_image_list()
            self.auto_save_data()
            QMessageBox.information(self, "성공", f"{len(urls)}개의 이미지가 추가되었습니다!")
            dialog.accept()
        else:
            QMessageBox.warning(self, "경고", "URL을 입력해주세요!")
    
    def bulk_add_with_ai_analysis(self, dialog, text_area, progress_label, progress_bar):
        """Bulk add with AI analysis"""
        urls = text_area.toPlainText().strip().split('\n')
        urls = [url.strip() for url in urls if url.strip()]
        
        if not urls:
            QMessageBox.warning(self, "경고", "URL을 입력해주세요!")
            return
        
        if not self.gemini_client.api_key:
            QMessageBox.warning(self, "경고", "Gemini API 키가 설정되지 않았습니다!")
            return
        
        # Show progress bar
        progress_label.setVisible(True)
        progress_bar.setVisible(True)
        progress_bar.setMaximum(len(urls))
        progress_bar.setValue(0)
        
        # Disable buttons during analysis
        for button in dialog.findChildren(QPushButton):
            button.setEnabled(False)
        
        # Start AI analysis in background thread
        self.bulk_analysis_worker = BulkAnalysisWorker(urls, self.gemini_client)
        self.bulk_analysis_worker.progress_updated.connect(lambda current, total: progress_bar.setValue(current))
        self.bulk_analysis_worker.analysis_completed.connect(lambda results: self.bulk_analysis_finished(dialog, results))
        self.bulk_analysis_worker.error_occurred.connect(lambda error: self.bulk_analysis_error(dialog, error))
        self.bulk_analysis_worker.start()
    
    def bulk_analysis_finished(self, dialog, results):
        """Handle bulk analysis completion"""
        # Add results to lists
        for url, category in results:
            self.image_urls.append(url)
            self.labels.append(category)
        
        self.update_image_list()
        self.auto_save_data()
        
        QMessageBox.information(self, "완료", f"{len(results)}개의 이미지가 AI 분석과 함께 추가되었습니다!")
        dialog.accept()
    
    def bulk_analysis_error(self, dialog, error_msg):
        """Handle bulk analysis error"""
        QMessageBox.critical(self, "오류", f"AI 분석 중 오류가 발생했습니다:\n{error_msg}")
        
        # Re-enable buttons
        for button in dialog.findChildren(QPushButton):
            button.setEnabled(True)
        
        # Hide progress bar
        progress_label = dialog.findChild(QLabel, "progress_label")
        progress_bar = dialog.findChild(QProgressBar)
        if progress_label:
            progress_label.setVisible(False)
        if progress_bar:
            progress_bar.setVisible(False)
    
    def update_image_list(self):
        self.image_list.clear()
        for i, url in enumerate(self.image_urls):
            item_text = f"{i+1}. {url}"
            if i < len(self.labels) and self.labels[i]:
                item_text += f" - {self.labels[i]}"
            
            item = QListWidgetItem(item_text)
            self.image_list.addItem(item)
    
    def select_all_images(self):
        # Select all items
        self.image_list.selectAll()
    
    def on_selection_changed(self):
        """Handle selection changes for preview"""
        selected_items = self.image_list.selectedItems()
        if selected_items:
            # Show preview of the first selected item
            first_selected = selected_items[0]
            row = self.image_list.row(first_selected)
            self.load_image_preview(row)
    
    def delete_selected_images(self):
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "경고", "삭제할 이미지를 선택해주세요!")
            return
        
        reply = QMessageBox.question(self, "확인", 
                                   f"{len(selected_items)}개의 이미지를 삭제하시겠습니까?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # Get indices of selected items
            indices = []
            for item in selected_items:
                indices.append(self.image_list.row(item))
            
            # Sort indices in descending order to delete from end
            indices.sort(reverse=True)
            
            # Delete from lists
            for idx in indices:
                if 0 <= idx < len(self.image_urls):
                    del self.image_urls[idx]
                if 0 <= idx < len(self.labels):
                    del self.labels[idx]
            
            self.update_image_list()
            self.auto_save_data()
            QMessageBox.information(self, "성공", f"{len(selected_items)}개의 이미지가 삭제되었습니다!")
    
    def start_ai_analysis(self):
        if not self.image_urls:
            QMessageBox.warning(self, "경고", "분석할 이미지가 없습니다!")
            return
        
        if not self.gemini_client.api_key:
            QMessageBox.warning(self, "경고", "Gemini API 키가 설정되지 않았습니다!")
            return
        
        # Check if "selected only" mode is enabled
        if self.analyze_selected_only.isChecked():
            # Get selected items
            selected_items = self.image_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "경고", "분석할 이미지를 선택하세요!")
                return
            
            # Get indices of selected items
            selected_indices = []
            selected_urls = []
            for item in selected_items:
                row = self.image_list.row(item)
                if 0 <= row < len(self.image_urls):
                    selected_indices.append(row)
                    selected_urls.append(self.image_urls[row])
            
            if not selected_urls:
                QMessageBox.warning(self, "경고", "유효한 선택 항목이 없습니다!")
                return
            
            # Use selected URLs for analysis
            urls_to_analyze = selected_urls
            indices_to_update = selected_indices
        else:
            # Analyze all images
            urls_to_analyze = self.image_urls
            indices_to_update = list(range(len(self.image_urls)))
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(urls_to_analyze))
        self.progress_bar.setValue(0)
        
        # Store indices to update after analysis
        self.current_analysis_indices = indices_to_update
        
        # Start analysis in background thread
        self.analysis_worker = ImageAnalysisWorker(urls_to_analyze, self.gemini_client)
        self.analysis_worker.progress_updated.connect(self.update_progress)
        self.analysis_worker.analysis_completed.connect(self.analysis_finished)
        self.analysis_worker.error_occurred.connect(self.analysis_error)
        self.analysis_worker.start()
    
    def update_progress(self, current, total):
        self.progress_bar.setValue(current)
        self.gemini_status_label.setText(f"분석 진행 중... ({current}/{total})")
    
    def analysis_finished(self, results):
        self.progress_bar.setVisible(False)
        
        # Update labels with results
        # If indices were stored, only update those specific indices
        if hasattr(self, 'current_analysis_indices') and self.current_analysis_indices:
            # Selected mode - only update specific indices
            for i, idx in enumerate(self.current_analysis_indices):
                if i < len(results) and idx < len(self.labels):
                    self.labels[idx] = results[i]
        else:
            # All images mode - update all
            for i, result in enumerate(results):
                if i < len(self.labels):
                    self.labels[i] = result
        
        # Clear the stored indices
        if hasattr(self, 'current_analysis_indices'):
            delattr(self, 'current_analysis_indices')
        
        self.update_image_list()
        self.auto_save_data()
        
        QMessageBox.information(self, "완료", "AI 분석이 완료되었습니다!")
        self.update_api_status()
    
    def analysis_error(self, error_msg):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "오류", f"분석 중 오류가 발생했습니다:\n{error_msg}")
        self.update_api_status()
    
    def toggle_performance_mode(self, checked):
        if checked:
            self.gemini_client.max_workers = 4
            logging.info("🔧 고성능 모드 활성화: 4개 워커")
        else:
            self.gemini_client.max_workers = 2
            logging.info("🔧 절전 모드 활성화: 2개 워커")
    
    def resize_preview(self, width):
        self.preview_width = width
        self.preview_height = int(width * 0.67)  # 3:2 aspect ratio
        
        # Update preview if image is loaded
        current_item = self.image_list.currentItem()
        if current_item:
            self.load_image_preview(self.image_list.currentRow())
    
    def load_image_preview(self, index):
        if 0 <= index < len(self.image_urls):
            url = self.image_urls[index]
            try:
                # Show loading message
                self.preview_label.setText("이미지 로딩 중...")
                self.preview_label.repaint()
                
                # Download image with headers to avoid blocking
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, timeout=15, headers=headers)
                response.raise_for_status()
                
                # Load image
                image = Image.open(io.BytesIO(response.content))
                
                # Convert to RGB if necessary
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                
                # Resize image maintaining aspect ratio
                image.thumbnail((self.preview_width, self.preview_height), Image.Resampling.LANCZOS)
                
                # Convert to QPixmap
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='PNG')
                image_bytes.seek(0)
                
                pixmap = QPixmap()
                if pixmap.loadFromData(image_bytes.getvalue()):
                    # Scale pixmap to fit the label
                    scaled_pixmap = pixmap.scaled(
                        self.preview_width, self.preview_height, 
                        Qt.AspectRatioMode.KeepAspectRatio, 
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.preview_label.setPixmap(scaled_pixmap)
                else:
                    self.preview_label.setText("이미지 형식 오류")
                
            except requests.exceptions.RequestException as e:
                self.preview_label.setText(f"네트워크 오류:\n{str(e)}")
            except Exception as e:
                self.preview_label.setText(f"이미지 로드 실패:\n{str(e)}")
        else:
            self.preview_label.setText("이미지를 선택하세요")
    
    def save_csv(self):
        if not self.image_urls:
            QMessageBox.warning(self, "경고", "저장할 데이터가 없습니다!")
            return
        
        try:
            # Create DataFrame
            df = pd.DataFrame({
                'image_url': self.image_urls,
                'category': self.labels
            })
            
            # Let user choose save directory
            save_dir = QFileDialog.getExistingDirectory(
                self, 
                "CSV 저장 폴더 선택", 
                "dataset",
                QFileDialog.Option.ShowDirsOnly
            )
            
            if not save_dir:
                return  # User cancelled
            
            # Create a new folder with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"cosmos_data_{timestamp}"
            full_save_dir = os.path.join(save_dir, folder_name)
            
            # Create the folder
            os.makedirs(full_save_dir, exist_ok=True)
            
            # Save all CSV files in the new folder
            df.to_csv(os.path.join(full_save_dir, "full_data.csv"), index=False, encoding='utf-8')
            df[['image_url']].to_csv(os.path.join(full_save_dir, "x_train.csv"), index=False, encoding='utf-8')
            df[['category']].to_csv(os.path.join(full_save_dir, "y_train.csv"), index=False, encoding='utf-8')
            
            # Also save timestamped version
            df.to_csv(os.path.join(full_save_dir, f"full_data_{timestamp}.csv"), index=False, encoding='utf-8')
            
            QMessageBox.information(self, "성공", 
                f"CSV 파일이 폴더로 저장되었습니다!\n"
                f"저장 위치: {full_save_dir}\n"
                f"포함된 파일:\n"
                f"- full_data.csv\n"
                f"- x_train.csv\n"
                f"- y_train.csv\n"
                f"- full_data_{timestamp}.csv")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"CSV 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def create_new_csv(self):
        reply = QMessageBox.question(self, "확인", 
                                   "모든 데이터를 삭제하고 새 CSV를 만들겠습니까?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.image_urls = []
            self.labels = []
            self.update_image_list()
            
            # Create empty CSV
            df = pd.DataFrame(columns=['image_url', 'category'])
            os.makedirs("dataset", exist_ok=True)
            df.to_csv("dataset/full_data.csv", index=False, encoding='utf-8')
            df[['image_url']].to_csv("dataset/x_train.csv", index=False, encoding='utf-8')
            df[['category']].to_csv("dataset/y_train.csv", index=False, encoding='utf-8')
            
            QMessageBox.information(self, "성공", "새 CSV 파일이 생성되었습니다!")
    
    def auto_save_data(self):
        """Auto-save data with timestamp"""
        if not self.image_urls:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create DataFrame
            df = pd.DataFrame({
                'image_url': self.image_urls,
                'category': self.labels
            })
            
            # Save with timestamp
            os.makedirs("dataset", exist_ok=True)
            df.to_csv(f"dataset/full_data_auto_{timestamp}.csv", index=False, encoding='utf-8')
            df[['image_url']].to_csv(f"dataset/x_train_auto_{timestamp}.csv", index=False, encoding='utf-8')
            df[['category']].to_csv(f"dataset/y_train_auto_{timestamp}.csv", index=False, encoding='utf-8')
            
            # Also update main files
            df.to_csv("dataset/full_data.csv", index=False, encoding='utf-8')
            df[['image_url']].to_csv("dataset/x_train.csv", index=False, encoding='utf-8')
            df[['category']].to_csv("dataset/y_train.csv", index=False, encoding='utf-8')
            
            logging.info(f"자동 저장 완료: {len(self.image_urls)}개 이미지")
            
        except Exception as e:
            logging.error(f"자동 저장 오류: {e}")
    
    def cleanup_old_data_files(self):
        """Delete old auto-save files, keeping only the latest"""
        try:
            # Find all auto-save files
            auto_files = glob.glob("dataset/*_auto_*.csv")
            scraped_files = glob.glob("dataset/scraped_images_*.csv")
            
            all_files = auto_files + scraped_files
            
            if len(all_files) > 1:
                # Sort by modification time
                all_files.sort(key=os.path.getmtime, reverse=True)
                
                # Keep the latest file, delete the rest
                for file_path in all_files[1:]:
                    try:
                        os.remove(file_path)
                        logging.info(f"오래된 파일 삭제: {file_path}")
                    except Exception as e:
                        logging.warning(f"파일 삭제 실패: {file_path} - {e}")
                        
        except Exception as e:
            logging.error(f"파일 정리 오류: {e}")
    
    def load_latest_data(self):
        """Load the most recent data file"""
        try:
            # Try to load from main files first
            if os.path.exists("dataset/full_data.csv"):
                df = pd.read_csv("dataset/full_data.csv")
                if not df.empty:
                    self.image_urls = df['image_url'].tolist()
                    self.labels = df['category'].fillna('').tolist()
                    logging.info(f"메인 데이터 로드: {len(self.image_urls)}개 이미지")
                    return
            
            # Try to load from latest auto-save file
            auto_files = glob.glob("dataset/full_data_auto_*.csv")
            if auto_files:
                latest_file = max(auto_files, key=os.path.getmtime)
                df = pd.read_csv(latest_file)
                if not df.empty:
                    self.image_urls = df['image_url'].tolist()
                    self.labels = df['category'].fillna('').tolist()
                    logging.info(f"자동 저장 데이터 로드: {len(self.image_urls)}개 이미지")
                    return
            
            # Try to load from separate files
            if os.path.exists("dataset/x_train.csv") and os.path.exists("dataset/y_train.csv"):
                x_df = pd.read_csv("dataset/x_train.csv")
                y_df = pd.read_csv("dataset/y_train.csv")
                
                if not x_df.empty and not y_df.empty:
                    self.image_urls = x_df['image_url'].tolist()
                    self.labels = y_df['category'].fillna('').tolist()
                    logging.info(f"분리된 데이터 로드: {len(self.image_urls)}개 이미지")
                    return
            
            logging.info("로드할 데이터가 없습니다.")
            
        except Exception as e:
            logging.error(f"데이터 로드 오류: {e}")
    
    def load_existing_data(self):
        """Load existing data and cleanup old files"""
        self.cleanup_old_data_files()
        self.load_latest_data()
        self.update_image_list()
    
    def manage_api_key(self):
        """Open API key management dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("API 키 관리")
        dialog.setModal(True)
        dialog.resize(500, 450)
        
        layout = QVBoxLayout(dialog)
        
        # Current API key status
        current_key = self.gemini_client.api_key
        if current_key:
            masked_key = self._mask_api_key(current_key)
            status_label = QLabel(f"현재 API 키: {masked_key}")
            status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            status_label = QLabel("API 키가 설정되지 않음")
            status_label.setStyleSheet("color: red; font-weight: bold;")
        
        layout.addWidget(status_label)
        
        # Instructions
        instructions = QLabel("Google AI Studio에서 API 키를 생성하고 입력하세요:")
        layout.addWidget(instructions)
        
        # API key input
        key_layout = QHBoxLayout()
        key_label = QLabel("API 키:")
        key_layout.addWidget(key_label)
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("API 키를 입력하세요...")
        if current_key:
            self.api_key_input.setText(current_key)
        key_layout.addWidget(self.api_key_input)
        
        layout.addLayout(key_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        test_button = QPushButton("API 키 테스트")
        test_button.clicked.connect(lambda: self.test_api_key(self.api_key_input.text()))
        button_layout.addWidget(test_button)
        
        save_button = QPushButton("저장 및 적용")
        save_button.clicked.connect(lambda: self.save_api_key(self.api_key_input.text()))
        button_layout.addWidget(save_button)
        
        cancel_button = QPushButton("취소")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def _mask_api_key(self, api_key):
        """Mask API key for display"""
        if len(api_key) <= 8:
            return "*" * len(api_key)
        return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
    
    def test_api_key(self, api_key):
        """Test API key validity"""
        if not api_key.strip():
            QMessageBox.warning(self, "경고", "API 키를 입력해주세요!")
            return
        
        # Temporarily set API key for testing
        original_key = self.gemini_client.api_key
        self.gemini_client.api_key = api_key.strip()
        
        # Test with a simple image
        test_url = "https://picsum.photos/150/150"
        result = self.gemini_client.analyze_image(test_url)
        
        # Restore original key
        self.gemini_client.api_key = original_key
        
        if "API 키가 설정되지 않음" in result or "API 오류" in result or "expired" in result or "INVALID" in result:
            QMessageBox.warning(self, "테스트 실패", 
                f"API 키가 유효하지 않거나 만료되었습니다.\n\n"
                f"오류 내용:\n{result}\n\n"
                f"Google AI Studio (https://aistudio.google.com)에서 새로운 API 키를 생성해주세요.")
        else:
            QMessageBox.information(self, "테스트 성공", f"API 키가 유효합니다!\n\n테스트 결과: {result}")
    
    def save_api_key(self, api_key):
        """Save API key to .env file"""
        if not api_key.strip():
            QMessageBox.warning(self, "경고", "API 키를 입력해주세요!")
            return
        
        try:
            # Save to .env file
            self._save_api_key_to_env(api_key.strip())
            
            # Update client
            self.gemini_client.api_key = api_key.strip()
            
            # Update UI
            self.update_api_status()
            
            QMessageBox.information(self, "성공", "API 키가 저장되었습니다!")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"API 키 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def _save_api_key_to_env(self, api_key):
        """Save API key to .env file"""
        env_content = f"GEMINI_API_KEY={api_key}\n"
        
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
    
    def delete_api_key(self):
        """Delete API key from .env file"""
        reply = QMessageBox.question(self, "확인", 
                                   "API 키를 삭제하시겠습니까?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Remove from .env file
                self._remove_api_key_from_env()
                
                # Update client
                self.gemini_client.api_key = None
                
                # Update UI
                self.update_api_status()
                
                QMessageBox.information(self, "성공", "API 키가 삭제되었습니다!")
                
            except Exception as e:
                QMessageBox.critical(self, "오류", f"API 키 삭제 중 오류가 발생했습니다:\n{str(e)}")
    
    def _remove_api_key_from_env(self):
        """Remove API key from .env file"""
        if os.path.exists('.env'):
            with open('.env', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Filter out GEMINI_API_KEY line
            filtered_lines = [line for line in lines if not line.startswith('GEMINI_API_KEY=')]
            
            with open('.env', 'w', encoding='utf-8') as f:
                f.writelines(filtered_lines)
    
    def test_gemini_connection(self):
        """Test Gemini API connection"""
        if not self.gemini_client.api_key:
            QMessageBox.warning(self, "경고", "API 키가 설정되지 않았습니다!")
            return
        
        # Test with a simple image
        test_url = "https://picsum.photos/150/150"
        result = self.gemini_client.analyze_image(test_url)
        
        if "API 키가 설정되지 않음" in result or "API 오류" in result:
            QMessageBox.warning(self, "연결 실패", f"Gemini API 연결 실패:\n{result}")
        else:
            QMessageBox.information(self, "연결 성공", f"Gemini API 연결 성공!\n테스트 결과: {result}")
        
        self.update_api_status()
    
    def update_api_status(self):
        """Update API status display"""
        if self.gemini_client.api_key:
            masked_key = self._mask_api_key(self.gemini_client.api_key)
            self.api_key_label.setText(f"API 키: {masked_key}")
            self.gemini_status_label.setText("Gemini API 상태: 연결됨")
        else:
            self.api_key_label.setText("API 키: 없음")
            self.gemini_status_label.setText("Gemini API 상태: 연결 실패")
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        # Check for Shift+Option+Arrow keys (macOS style)
        modifiers = event.modifiers()
        
        # Check if Shift and Option are pressed
        if (modifiers & Qt.KeyboardModifier.ShiftModifier and 
            modifiers & Qt.KeyboardModifier.AltModifier):
            
            if event.key() == Qt.Key.Key_Up:
                self.extend_selection_up()
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Down:
                self.extend_selection_down()
                event.accept()
                return
        
        # Call parent keyPressEvent for other keys
        super().keyPressEvent(event)
    
    def extend_selection_up(self):
        """Extend selection upward"""
        current_row = self.image_list.currentRow()
        if current_row > 0:
            # Select the item above
            self.image_list.setCurrentRow(current_row - 1)
            # Extend selection to include the new item
            self.image_list.setItemSelected(self.image_list.item(current_row - 1), True)
    
    def extend_selection_down(self):
        """Extend selection downward"""
        current_row = self.image_list.currentRow()
        if current_row < self.image_list.count() - 1:
            # Select the item below
            self.image_list.setCurrentRow(current_row + 1)
            # Extend selection to include the new item
            self.image_list.setItemSelected(self.image_list.item(current_row + 1), True)
    
    def closeEvent(self, event):
        """Handle application close"""
        self.auto_save_data()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set dark theme
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = CosmosGUIPyQt6()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
