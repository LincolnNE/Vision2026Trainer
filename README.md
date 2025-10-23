# Automated Cosmos CDN Image Classification Model Training Pipeline

This project provides a fully automated pipeline that automatically scrapes images from the cosmos.so CDN and trains an image classification model.

## Key Features

- **Automated Web Scraping**: Automatic collection of image URLs from cosmos.so CDN
- **Intelligent Category Classification**: Automatic labeling based on URL paths
- **Image Download and Preprocessing**: 224x224 resize, RGB normalization
- **CNN Model Training**: PyTorch-based image classification model
- **Full Automation**: One-click execution from scraping to model training

## Data Format

### Automated Scraping Results:
- **x_train**: `https://cdn.cosmos.so/book/layout/book1.webp` (scraped image URL)
- **category**: `"book_layout"` (category automatically extracted from URL path)
- **y_train**: `"book_layout"` (automatically generated label)

### Supported Automatic Categories:
- `book_layout`: Book layout
- `photography`: Photography
- `design`: Design
- `artwork`: Artwork
- `minimal_design`: Minimal design
- `abstract_art`: Abstract art
- `texture`: Texture
- `pattern`: Pattern
- `layout`: Layout
- `creative`: Creative
- `black_white`: Black and white
- `monochrome`: Monochrome
- `colorful`: Colorful
- `vintage`: Vintage
- `modern`: Modern
- `classic`: Classic

## Project Structure

```
Vision2025Trainer/
├── dataset/
│   ├── x_train.csv      # Scraped image URLs and categories
│   └── y_train.csv      # Automatically generated labels
├── models/
│   └── model.pt         # Trained PyTorch model
├── results/
│   └── metrics.png      # Training results visualization graph
├── auto_scraping_pipeline.py    # Main automated scraping pipeline
├── image_classification_pipeline.py  # Original manual pipeline
├── test_auto_scraping.py        # Automated scraping test script
├── test_pipeline.py             # Original pipeline test script
├── requirements.txt              # Required packages list
└── README.md                    # This file
```

## Installation and Execution

### 1. Environment Setup

```bash
# Python 3.10 or higher required
python3 --version

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install required packages
pip install -r requirements.txt
```

### 2. Run Automated Scraping Pipeline

```bash
# Run fully automated pipeline
python3 auto_scraping_pipeline.py
```

### 3. Run Tests

```bash
# Test automated scraping pipeline
python3 test_auto_scraping.py

# Test original manual pipeline
python3 test_pipeline.py
```

## Usage

### Automated Scraping Pipeline

The pipeline automatically executes in the following steps:

1. **Web Scraping**: Automatic collection of image URLs from cosmos.so CDN
2. **Category Classification**: Automatic labeling based on URL paths
3. **CSV Generation**: Generate x_train.csv, y_train.csv files
4. **Data Preprocessing**: Image download, resize, normalization
5. **Model Training**: CNN model training (10 epochs)
6. **Result Saving**: Save model file and visualization graph

### Customization

#### Change Scraping Settings

```python
# In the main() function of auto_scraping_pipeline.py
pipeline = AutoScrapingPipeline()

# Adjust number of pages to scrape
image_urls, labels = pipeline.scrape_and_categorize(max_pages=10)  # Default: 5
```

#### Add Category Keywords

```python
# In the CosmosScraper class of auto_scraping_pipeline.py
category_keywords = {
    'book': 'book_layout',
    'art': 'artwork',
    'photo': 'photography',
    # Add new keywords
    'nature': 'nature_photography',
    'urban': 'urban_design',
    # ...
}
```

#### Change Scraping Target URL

```python
# In the CosmosScraper class of auto_scraping_pipeline.py
def __init__(self, base_url: str = "https://your-custom-cdn.com", timeout: int = 10):
    self.base_url = base_url
    # ...
```

### Using Local Files

If you have actual image files locally, you can use the `base_url` parameter of `CosmosImageDataset`:

```python
dataset = CosmosImageDataset(
    image_urls, 
    labels, 
    transform=transform,
    base_url="/path/to/your/images"  # Local image directory
)
```

## Detailed Scraping Features

### Supported Image Formats
- `.webp`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`

### Scraping Strategy
1. **HTML Parsing**: Structured data extraction using BeautifulSoup
2. **Link Following**: Navigate through page links to find images
3. **Duplicate Removal**: Automatic removal of identical image URLs
4. **Error Handling**: Robust handling of network errors and parsing errors

### Category Extraction Algorithm
1. **URL Path Analysis**: Extract keywords from `/book/layout/`, `/photo/minimal/` etc.
2. **Filename Analysis**: Extract categories from `book_spread_001.webp` etc.
3. **Keyword Matching**: Match with predefined keyword dictionary
4. **Default Category**: Assign 'general' category if no match is found

## Model Architecture

CNN model structure used:

```
Conv2D(3→32) → ReLU → MaxPool2D
Conv2D(32→64) → ReLU → MaxPool2D  
Conv2D(64→128) → ReLU → MaxPool2D
Flatten → Dense(128×28×28→512) → ReLU → Dropout(0.5)
Dense(512→num_classes) → Softmax
```

## Output Files

After execution completion, the following files are generated:

- `./dataset/x_train.csv`: Scraped image URLs and automatically generated categories
- `./dataset/y_train.csv`: Automatically generated label information  
- `./models/model.pt`: Trained PyTorch model
- `./results/metrics.png`: Training loss and accuracy graph

## Requirements

- Python 3.10 or higher
- PyTorch 2.0.0 or higher
- BeautifulSoup4 4.11.0 or higher
- Internet connection (for scraping and image download)
- Sufficient disk space (for image caching)

## Troubleshooting

### Common Errors

1. **Scraping Failure**: 
   - Check network connection
   - Check target site's robots.txt
   - Update User-Agent header

2. **Image Download Failure**: 
   - Validate URL
   - Adjust timeout settings

3. **Memory Insufficient**: 
   - Reduce batch size
   - Adjust image resolution

4. **Inaccurate Category Classification**: 
   - Update keyword dictionary
   - Analyze URL patterns

### Log Checking

Detailed logs are output during program execution, so check logs when problems occur.

### Performance Optimization

- Adjust scraping scope with `max_pages` parameter
- Adjust network timeout with `timeout` parameter
- Adjust memory usage with `batch_size` parameter

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Important Notes

- Comply with the terms of service of target sites when web scraping
- Be careful not to overload servers with excessive requests
- Be cautious when using copyrighted images