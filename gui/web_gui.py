#!/usr/bin/env python3
"""
웹 기반 이미지 분류 관리자 - 다중 선택 지원
- 브라우저에서 실행
- macOS 호환성 문제 완전 해결
- 다중 선택 기능 포함
"""

import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import threading

class ImageData:
    """이미지 데이터 관리"""
    def __init__(self):
        self.image_urls = []
        self.labels = []
        self.load_data()
    
    def load_data(self):
        """데이터 로드"""
        try:
            if os.path.exists('./web_data.json'):
                with open('./web_data.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.image_urls = data.get('image_urls', [])
                    self.labels = data.get('labels', [])
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
    
    def save_data(self):
        """데이터 저장"""
        try:
            data = {
                'image_urls': self.image_urls,
                'labels': self.labels,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open('./web_data.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"데이터 저장 실패: {e}")
    
    def add_image(self, url, category="manual, general, creative"):
        """이미지 추가"""
        self.image_urls.append(url)
        self.labels.append(category)
        self.save_data()
    
    def delete_images(self, indices):
        """이미지 삭제 (다중 선택 지원)"""
        # 역순으로 삭제 (인덱스가 변경되지 않도록)
        for index in sorted(indices, reverse=True):
            if 0 <= index < len(self.image_urls):
                del self.image_urls[index]
                del self.labels[index]
        self.save_data()
    
    def update_categories(self, indices, new_category):
        """카테고리 업데이트 (다중 선택 지원)"""
        for index in indices:
            if 0 <= index < len(self.labels):
                self.labels[index] = new_category
        self.save_data()

# 전역 데이터 인스턴스
image_data = ImageData()

class WebHandler(BaseHTTPRequestHandler):
    """웹 요청 핸들러"""
    
    def do_GET(self):
        """GET 요청 처리"""
        if self.path == '/':
            self.serve_main_page()
        elif self.path == '/api/data':
            self.serve_data()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """POST 요청 처리"""
        if self.path == '/api/add':
            self.handle_add_image()
        elif self.path == '/api/delete':
            self.handle_delete_images()
        elif self.path == '/api/update_categories':
            self.handle_update_categories()
        else:
            self.send_error(404)
    
    def serve_main_page(self):
        """메인 페이지 제공"""
        html = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>이미지 분류 관리자 - 다중 선택 지원</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .content {
            padding: 20px;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: #fafafa;
        }
        .section h3 {
            margin-top: 0;
            color: #333;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
        }
        input[type="text"], input[type="url"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }
        button:hover {
            background: #5a6fd8;
        }
        button.danger {
            background: #e74c3c;
        }
        button.danger:hover {
            background: #c0392b;
        }
        .image-list {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: white;
        }
        .image-item {
            padding: 10px;
            border-bottom: 1px solid #eee;
            cursor: pointer;
            display: flex;
            align-items: center;
        }
        .image-item:hover {
            background: #f0f0f0;
        }
        .image-item.selected {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        .image-item input[type="checkbox"] {
            margin-right: 10px;
        }
        .image-info {
            flex: 1;
        }
        .image-url {
            font-weight: 500;
            color: #333;
        }
        .image-category {
            font-size: 12px;
            color: #666;
            margin-top: 2px;
        }
        .help-text {
            background: #e8f4fd;
            border: 1px solid #bee5eb;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 15px;
            font-size: 14px;
            color: #0c5460;
        }
        .log {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 10px;
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            font-weight: 500;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🖼️ 이미지 분류 관리자</h1>
            <p>다중 선택 기능이 포함된 웹 기반 이미지 관리 도구</p>
        </div>
        
        <div class="content">
            <!-- 이미지 추가 섹션 -->
            <div class="section">
                <h3>📥 이미지 추가</h3>
                <div class="form-group">
                    <label for="imageUrl">이미지 URL:</label>
                    <input type="url" id="imageUrl" placeholder="https://example.com/image.jpg">
                </div>
                <div class="form-group">
                    <label for="category">카테고리:</label>
                    <input type="text" id="category" placeholder="nature, landscape, outdoor" value="manual, general, creative">
                </div>
                <button onclick="addImage()">이미지 추가</button>
            </div>
            
            <!-- 이미지 목록 섹션 -->
            <div class="section">
                <h3>📋 이미지 목록</h3>
                <div class="help-text">
                    💡 다중 선택 사용법:<br>
                    • 체크박스를 클릭하여 개별 선택<br>
                    • Shift+클릭으로 범위 선택<br>
                    • Ctrl+A (Cmd+A)로 전체 선택<br>
                    • 선택된 항목들에 대해 일괄 작업 수행
                </div>
                <div style="margin-bottom: 10px;">
                    <button onclick="selectAll()">전체 선택</button>
                    <button onclick="clearSelection()">선택 해제</button>
                    <button class="danger" onclick="deleteSelected()">선택 삭제</button>
                </div>
                <div class="image-list" id="imageList">
                    <div style="padding: 20px; text-align: center; color: #666;">
                        이미지를 추가하세요
                    </div>
                </div>
            </div>
            
            <!-- 카테고리 관리 섹션 -->
            <div class="section">
                <h3>🏷️ 카테고리 관리</h3>
                <div class="form-group">
                    <label for="newCategory">새 카테고리 (콤마로 구분):</label>
                    <input type="text" id="newCategory" placeholder="nature, landscape, outdoor">
                </div>
                <button onclick="updateSelectedCategories()">선택된 항목 카테고리 변경</button>
            </div>
            
            <!-- 로그 섹션 -->
            <div class="section">
                <h3>📝 로그</h3>
                <div class="log" id="logArea"></div>
                <button onclick="clearLog()">로그 지우기</button>
            </div>
        </div>
    </div>

    <script>
        let images = [];
        let selectedIndices = new Set();
        
        // 페이지 로드 시 데이터 가져오기
        window.onload = function() {
            loadData();
            // Ctrl+A 전체 선택 지원
            document.addEventListener('keydown', function(e) {
                if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
                    e.preventDefault();
                    selectAll();
                }
            });
        };
        
        // 데이터 로드
        async function loadData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                images = data.image_urls || [];
                updateImageList();
                log('데이터 로드 완료: ' + images.length + '개 이미지');
            } catch (error) {
                log('데이터 로드 실패: ' + error.message, 'error');
            }
        }
        
        // 이미지 추가
        async function addImage() {
            const url = document.getElementById('imageUrl').value.trim();
            const category = document.getElementById('category').value.trim();
            
            if (!url) {
                alert('이미지 URL을 입력하세요.');
                return;
            }
            
            if (!url.startsWith('http://') && !url.startsWith('https://')) {
                alert('올바른 URL 형식이 아닙니다.');
                return;
            }
            
            try {
                const response = await fetch('/api/add', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        url: url,
                        category: category || 'manual, general, creative'
                    })
                });
                
                if (response.ok) {
                    document.getElementById('imageUrl').value = '';
                    document.getElementById('category').value = 'manual, general, creative';
                    await loadData();
                    log('이미지 추가 완료: ' + url);
                    showStatus('이미지가 성공적으로 추가되었습니다.', 'success');
                } else {
                    throw new Error('서버 오류');
                }
            } catch (error) {
                log('이미지 추가 실패: ' + error.message, 'error');
                showStatus('이미지 추가에 실패했습니다.', 'error');
            }
        }
        
        // 이미지 목록 업데이트
        function updateImageList() {
            const listElement = document.getElementById('imageList');
            
            if (images.length === 0) {
                listElement.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">이미지를 추가하세요</div>';
                return;
            }
            
            listElement.innerHTML = images.map((url, index) => {
                const filename = url.split('/').pop().split('?')[0];
                const category = images[index] ? (images[index].category || 'manual, general, creative') : 'manual, general, creative';
                const isSelected = selectedIndices.has(index);
                
                return `
                    <div class="image-item ${isSelected ? 'selected' : ''}" onclick="toggleSelection(${index})">
                        <input type="checkbox" ${isSelected ? 'checked' : ''} onchange="toggleSelection(${index})">
                        <div class="image-info">
                            <div class="image-url">${index + 1}. ${filename}</div>
                            <div class="image-category">카테고리: ${category}</div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        // 선택 토글
        function toggleSelection(index) {
            if (selectedIndices.has(index)) {
                selectedIndices.delete(index);
            } else {
                selectedIndices.add(index);
            }
            updateImageList();
        }
        
        // 전체 선택
        function selectAll() {
            selectedIndices.clear();
            for (let i = 0; i < images.length; i++) {
                selectedIndices.add(i);
            }
            updateImageList();
            log('전체 선택: ' + images.length + '개 항목');
        }
        
        // 선택 해제
        function clearSelection() {
            selectedIndices.clear();
            updateImageList();
            log('선택 해제');
        }
        
        // 선택된 항목 삭제
        async function deleteSelected() {
            if (selectedIndices.size === 0) {
                alert('삭제할 이미지를 선택하세요.');
                return;
            }
            
            const count = selectedIndices.size;
            if (!confirm(`선택된 ${count}개 이미지를 삭제하시겠습니까?`)) {
                return;
            }
            
            try {
                const response = await fetch('/api/delete', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        indices: Array.from(selectedIndices)
                    })
                });
                
                if (response.ok) {
                    selectedIndices.clear();
                    await loadData();
                    log(`${count}개 이미지 삭제 완료`);
                    showStatus(`${count}개 이미지가 삭제되었습니다.`, 'success');
                } else {
                    throw new Error('서버 오류');
                }
            } catch (error) {
                log('이미지 삭제 실패: ' + error.message, 'error');
                showStatus('이미지 삭제에 실패했습니다.', 'error');
            }
        }
        
        // 선택된 항목 카테고리 변경
        async function updateSelectedCategories() {
            if (selectedIndices.size === 0) {
                alert('카테고리를 변경할 이미지를 선택하세요.');
                return;
            }
            
            const newCategory = document.getElementById('newCategory').value.trim();
            if (!newCategory) {
                alert('새 카테고리를 입력하세요.');
                return;
            }
            
            try {
                const response = await fetch('/api/update_categories', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        indices: Array.from(selectedIndices),
                        category: newCategory
                    })
                });
                
                if (response.ok) {
                    await loadData();
                    log(`${selectedIndices.size}개 이미지 카테고리 변경 완료: ${newCategory}`);
                    showStatus(`${selectedIndices.size}개 이미지의 카테고리가 변경되었습니다.`, 'success');
                    document.getElementById('newCategory').value = '';
                } else {
                    throw new Error('서버 오류');
                }
            } catch (error) {
                log('카테고리 변경 실패: ' + error.message, 'error');
                showStatus('카테고리 변경에 실패했습니다.', 'error');
            }
        }
        
        // 로그 함수
        function log(message, type = 'info') {
            const logArea = document.getElementById('logArea');
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.textContent = `${timestamp} - ${message}`;
            if (type === 'error') {
                logEntry.style.color = '#e74c3c';
            } else if (type === 'success') {
                logEntry.style.color = '#27ae60';
            }
            logArea.appendChild(logEntry);
            logArea.scrollTop = logArea.scrollHeight;
        }
        
        // 상태 메시지 표시
        function showStatus(message, type) {
            const status = document.createElement('div');
            status.className = `status ${type}`;
            status.textContent = message;
            document.querySelector('.content').insertBefore(status, document.querySelector('.content').firstChild);
            
            setTimeout(() => {
                status.remove();
            }, 3000);
        }
        
        // 로그 지우기
        function clearLog() {
            document.getElementById('logArea').innerHTML = '';
        }
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def serve_data(self):
        """데이터 API 제공"""
        data = {
            'image_urls': image_data.image_urls,
            'labels': image_data.labels,
            'count': len(image_data.image_urls)
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    
    def handle_add_image(self):
        """이미지 추가 처리"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        image_data.add_image(data['url'], data['category'])
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps({'success': True}).encode('utf-8'))
    
    def handle_delete_images(self):
        """이미지 삭제 처리"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        image_data.delete_images(data['indices'])
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps({'success': True}).encode('utf-8'))
    
    def handle_update_categories(self):
        """카테고리 업데이트 처리"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        image_data.update_categories(data['indices'], data['category'])
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps({'success': True}).encode('utf-8'))
    
    def log_message(self, format, *args):
        """로그 메시지 무시"""
        pass

def start_web_server():
    """웹 서버 시작"""
    port = 8080
    server = HTTPServer(('localhost', port), WebHandler)
    
    print(f"🌐 웹 기반 이미지 분류 관리자가 시작되었습니다!")
    print(f"📍 주소: http://localhost:{port}")
    print(f"🚀 브라우저에서 위 주소로 접속하세요")
    print(f"💡 다중 선택 기능이 포함되어 있습니다!")
    print(f"⏹️  종료하려면 Ctrl+C를 누르세요")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 서버가 종료되었습니다.")
        server.shutdown()

if __name__ == "__main__":
    start_web_server()






