#!/usr/bin/env python3
"""
간단한 이미지 분류 GUI - macOS 호환성 문제 해결
- matplotlib 제거
- 기본 tkinter만 사용
- 다중 선택 기능 포함
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import pandas as pd
import requests
import logging
from typing import List, Tuple, Dict, Optional
import time
import base64
from PIL import Image, ImageTk
import io

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleImageGUI:
    """간단한 이미지 분류 GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("이미지 분류 관리자 - 다중 선택 지원")
        self.root.geometry("1400x800")
        
        # 데이터 저장
        self.image_urls = []
        self.labels = []
        
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
        
        # 상단 패널 (제어)
        self.setup_control_panel(main_frame)
        
        # 중간 패널 (이미지 관리)
        self.setup_image_panel(main_frame)
        
        # 하단 패널 (로그)
        self.setup_logging_panel(main_frame)

    def setup_control_panel(self, parent):
        """제어 패널 구성"""
        control_frame = ttk.LabelFrame(parent, text="제어", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 수동 이미지 링크 입력
        manual_frame = ttk.Frame(control_frame)
        manual_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(manual_frame, text="이미지 링크:").pack(side=tk.LEFT)
        self.manual_url_var = tk.StringVar()
        self.manual_url_entry = ttk.Entry(manual_frame, textvariable=self.manual_url_var, width=50)
        self.manual_url_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        ttk.Button(manual_frame, text="추가", command=self.add_manual_image).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 데이터셋 관리
        data_frame = ttk.Frame(control_frame)
        data_frame.pack(fill=tk.X)
        
        ttk.Button(data_frame, text="CSV 저장", command=self.save_csv).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(data_frame, text="CSV 로드", command=self.load_csv).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(data_frame, text="전체 선택", command=self.select_all_images).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(data_frame, text="선택 삭제", command=self.delete_selected_images).pack(side=tk.LEFT)

    def setup_image_panel(self, parent):
        """이미지 관리 패널 구성"""
        image_frame = ttk.LabelFrame(parent, text="이미지 관리", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 좌측: 이미지 리스트
        left_frame = ttk.Frame(image_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 이미지 리스트박스
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(list_frame, text="이미지 목록:").pack(anchor=tk.W)
        
        # 다중 선택 도움말
        help_text = "💡 다중 선택: Shift+Option+↑↓ 또는 Shift+Ctrl+↑↓로 범위 선택"
        help_label = ttk.Label(list_frame, text=help_text, font=("Arial", 8), foreground='gray')
        help_label.pack(anchor=tk.W, pady=(0, 5))
        
        # 리스트박스와 스크롤바
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.image_listbox = tk.Listbox(list_container, selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.image_listbox.yview)
        self.image_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # 키보드 단축키 바인딩
        self.image_listbox.bind('<Command-a>', self.select_all_images)
        self.image_listbox.bind('<Control-a>', self.select_all_images)
        
        # Shift+Option+위아래 화살표로 다중 선택 지원
        self.image_listbox.bind('<Shift-Option-Up>', self.extend_selection_up)
        self.image_listbox.bind('<Shift-Option-Down>', self.extend_selection_down)
        self.image_listbox.bind('<Shift-Control-Up>', self.extend_selection_up)  # Windows/Linux 지원
        self.image_listbox.bind('<Shift-Control-Down>', self.extend_selection_down)  # Windows/Linux 지원
        
        self.image_listbox.focus_set()
        
        # 카테고리 관리
        category_frame = ttk.Frame(left_frame)
        category_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(category_frame, text="카테고리:").pack(side=tk.LEFT)
        
        self.category_var = tk.StringVar()
        self.category_entry = ttk.Entry(category_frame, textvariable=self.category_var, width=20)
        self.category_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        ttk.Button(category_frame, text="카테고리 변경", command=self.change_category).pack(side=tk.RIGHT)
        
        # 우측: 이미지 미리보기
        right_frame = ttk.Frame(image_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_frame, text="이미지 미리보기:").pack(anchor=tk.W)
        
        self.image_label = ttk.Label(right_frame, text="이미지를 선택하세요", 
                                   background='white', relief=tk.SUNKEN)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    def setup_logging_panel(self, parent):
        """로깅 패널 구성"""
        log_frame = ttk.LabelFrame(parent, text="로그", padding=5)
        log_frame.pack(fill=tk.X)
        
        self.log_text = tk.Text(log_frame, height=6, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_logging(self):
        """로깅 설정"""
        pass

    def log_message(self, message: str):
        """로그 메시지 추가"""
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def add_manual_image(self):
        """수동으로 이미지 링크 추가"""
        url = self.manual_url_var.get().strip()
        if not url:
            messagebox.showwarning("경고", "이미지 링크를 입력하세요.")
            return
        
        if not url.startswith(('http://', 'https://')):
            messagebox.showerror("오류", "올바른 URL 형식이 아닙니다.")
            return
        
        category = "manual, general, creative"
        
        self.image_urls.append(url)
        self.labels.append(category)
        
        self.update_image_list()
        self.manual_url_var.set("")
        
        self.log_message(f"이미지 추가: {url}")
        messagebox.showinfo("성공", "이미지가 추가되었습니다.")

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
        
        # 이미지 로드 및 표시
        def load_image():
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content))
                image.thumbnail((300, 300), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
            except Exception as e:
                self.image_label.config(image="", text=f"이미지 로드 실패:\n{str(e)}")
        
        threading.Thread(target=load_image, daemon=True).start()

    def select_all_images(self, event=None):
        """모든 이미지 선택"""
        if not self.image_urls:
            return
        
        self.image_listbox.selection_clear(0, tk.END)
        for i in range(len(self.image_urls)):
            self.image_listbox.selection_set(i)
        
        self.image_listbox.activate(0)
        self.image_listbox.see(0)
        
        self.on_image_select(None)
        
        self.log_message(f"모든 이미지 선택됨 ({len(self.image_urls)}개)")
        return "break"

    def extend_selection_up(self, event=None):
        """Shift+Option+위 화살표로 선택 범위 확장 (위쪽)"""
        if not self.image_urls:
            return "break"
        
        current_selection = self.image_listbox.curselection()
        if not current_selection:
            active_index = self.image_listbox.index(tk.ACTIVE)
            if active_index >= 0:
                self.image_listbox.selection_set(active_index)
                self.image_listbox.activate(active_index)
                self.on_image_select(None)
            return "break"
        
        first_selected = min(current_selection)
        
        if first_selected > 0:
            new_index = first_selected - 1
            
            self.image_listbox.selection_set(new_index)
            self.image_listbox.activate(new_index)
            self.image_listbox.see(new_index)
            
            current_selection = self.image_listbox.curselection()
            self.log_message(f"다중 선택: {len(current_selection)}개 항목 선택됨")
            
            self.on_image_select(None)
        
        return "break"

    def extend_selection_down(self, event=None):
        """Shift+Option+아래 화살표로 선택 범위 확장 (아래쪽)"""
        if not self.image_urls:
            return "break"
        
        current_selection = self.image_listbox.curselection()
        if not current_selection:
            active_index = self.image_listbox.index(tk.ACTIVE)
            if active_index >= 0:
                self.image_listbox.selection_set(active_index)
                self.image_listbox.activate(active_index)
                self.on_image_select(None)
            return "break"
        
        last_selected = max(current_selection)
        
        if last_selected < len(self.image_urls) - 1:
            new_index = last_selected + 1
            
            self.image_listbox.selection_set(new_index)
            self.image_listbox.activate(new_index)
            self.image_listbox.see(new_index)
            
            current_selection = self.image_listbox.curselection()
            self.log_message(f"다중 선택: {len(current_selection)}개 항목 선택됨")
            
            self.on_image_select(None)
        
        return "break"

    def change_category(self):
        """카테고리 변경 (다중 선택 지원)"""
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("경고", "이미지를 선택하세요.")
            return
        
        new_category_input = self.category_var.get().strip()
        
        if not new_category_input:
            messagebox.showwarning("경고", "새 카테고리를 입력하세요.")
            return
        
        categories = [cat.strip() for cat in new_category_input.split(',') if cat.strip()]
        
        if not categories:
            messagebox.showwarning("경고", "유효한 카테고리를 입력하세요.")
            return
        
        changed_count = 0
        for index in selection:
            self.labels[index] = new_category_input
            
            filename = self.image_urls[index].split('/')[-1].split('?')[0]
            self.image_listbox.delete(index)
            
            if len(categories) <= 3:
                display_categories = ', '.join(categories)
            else:
                display_categories = ', '.join(categories[:3]) + f" (+{len(categories)-3})"
            
            self.image_listbox.insert(index, f"{index+1:2d}. [{display_categories}] {filename}")
            changed_count += 1
        
        for index in selection:
            self.image_listbox.selection_set(index)
        
        if len(selection) == 1:
            self.log_message(f"이미지 {selection[0]+1}의 카테고리를 '{new_category_input}'로 변경 ({len(categories)}개 카테고리)")
        else:
            self.log_message(f"{changed_count}개 이미지의 카테고리를 '{new_category_input}'로 변경 ({len(categories)}개 카테고리)")

    def delete_selected_images(self):
        """선택된 이미지들 삭제 (다중 선택 지원)"""
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("경고", "삭제할 이미지를 선택하세요.")
            return
        
        if len(selection) == 1:
            confirm_msg = f"선택된 이미지 1개를 삭제하시겠습니까?"
        else:
            confirm_msg = f"선택된 이미지 {len(selection)}개를 삭제하시겠습니까?"
        
        if not messagebox.askyesno("확인", confirm_msg):
            return
        
        deleted_count = 0
        for index in reversed(sorted(selection)):
            if 0 <= index < len(self.image_urls):
                deleted_url = self.image_urls[index]
                deleted_label = self.labels[index]
                
                del self.image_urls[index]
                del self.labels[index]
                
                deleted_count += 1
                self.log_message(f"이미지 삭제: {deleted_url.split('/')[-1].split('?')[0]} ({deleted_label})")
        
        self.update_image_list()
        
        if deleted_count == 1:
            messagebox.showinfo("완료", "이미지 1개가 삭제되었습니다.")
        else:
            messagebox.showinfo("완료", f"이미지 {deleted_count}개가 삭제되었습니다.")
        
        self.log_message(f"✅ {deleted_count}개 이미지 삭제 완료")

    def save_csv(self):
        """CSV 파일 저장"""
        if not self.image_urls:
            messagebox.showwarning("경고", "저장할 데이터가 없습니다.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                df = pd.DataFrame({
                    'image_url': self.image_urls,
                    'category': self.labels
                })
                df.to_csv(filename, index=False)
                
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
                
                if 'image_url' in df.columns and 'category' in df.columns:
                    self.image_urls = df['image_url'].tolist()
                    self.labels = df['category'].tolist()
                
                self.update_image_list()
                self.log_message(f"CSV 파일 로드 완료: {filename}")
                
            except Exception as e:
                messagebox.showerror("오류", f"파일 로드 실패: {str(e)}")

    def load_existing_data(self):
        """기존 데이터 로드"""
        try:
            if os.path.exists('./dataset/full_data.csv'):
                df = pd.read_csv('./dataset/full_data.csv')
                if 'image_url' in df.columns and 'category' in df.columns:
                    self.image_urls = df['image_url'].tolist()
                    self.labels = df['category'].tolist()
                    self.update_image_list()
                    self.log_message(f"기존 데이터 로드: {len(self.image_urls)}개 이미지")
        except Exception as e:
            self.log_message(f"기존 데이터 로드 실패: {str(e)}")

def main():
    """메인 함수"""
    root = tk.Tk()
    app = SimpleImageGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()






