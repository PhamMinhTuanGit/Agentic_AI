# Hybrid PDF Embedding Pipeline với Semantic Chunking

Pipeline tự động để ingest, chunk và tạo hybrid embeddings cho tài liệu PDF với kỹ thuật kết hợp dense và sparse embeddings và semantic chunking thông minh.

## Tính năng chính

- **Semantic Chunking**: Chia tài liệu dựa trên ngữ nghĩa thay vì chỉ độ dài
- **Hybrid Embedding**: Kết hợp Dense Embeddings (từ API) và Sparse Embeddings (TF-IDF) với tỷ lệ alpha = 0.7
- **Intelligent Text Splitting**: Phân tích cấu trúc câu và độ tương đồng ngữ nghĩa
- **Metadata tracking**: Theo dõi thông tin về file gốc và chunk
- **FAISS indexing**: Lưu trữ hiệu quả để tìm kiếm nhanh

## Cấu trúc

```
Agentic/
├── documents/           # Thư mục chứa PDF cần xử lý
├── ingest/
│   ├── embedder.py     # Main pipeline code
│   └── requirements.txt # Dependencies
├── rag_backend/        # Output directory
└── test_pipeline.py    # Demo script
```

## Cài đặt

1. Cài đặt dependencies:
```bash
pip install -r ingest/requirements.txt
```

2. Đảm bảo API embedding server đang chạy tại `http://172.19.20.217:11435`

## Sử dụng

### Cách 1: Chạy trực tiếp
```bash
cd ingest
python3 embedder.py
```

### Cách 2: Sử dụng script
```bash
./run_embedding_pipeline.sh
```

### Cách 3: Test với demo
```bash
python3 test_pipeline.py
```

## Semantic Chunking với Cosine Similarity

Pipeline sử dụng semantic chunking với các tính năng:

### Phân tích cấu trúc:
- Tách câu dựa trên dấu câu và cấu trúc
- Sử dụng **TF-IDF vectorizer** để tạo embeddings
- Tính toán **cosine similarity** giữa các câu liền kề
- Tạo chunks dựa trên độ tương đồng ngữ nghĩa

### Cosine Similarity:
- **Vector hóa**: Chuyển câu thành TF-IDF vectors
- **Cosine distance**: Đo góc giữa hai vectors
- **Accuracy**: Chính xác hơn so với word intersection
- **Robustness**: Ít bị ảnh hưởng bởi độ dài câu

### Tham số chunking:
- `chunk_size`: Kích thước tối đa (mặc định: 800)
- `min_chunk_size`: Kích thước tối thiểu (mặc định: 200)
- `similarity_threshold`: Ngưỡng cosine similarity (mặc định: 0.5)

### Thuật toán:
1. Tách text thành câu
2. Fit TF-IDF vectorizer trên tất cả câu
3. Tính cosine similarity giữa các câu liền kề
4. Gộp các câu có similarity > threshold
5. Đảm bảo kích thước chunk trong khoảng cho phép

## Output Files

Pipeline tạo ra các file trong `rag_backend/`:

1. **hybrid_docs_index.faiss**: FAISS index cho tìm kiếm vector
2. **hybrid_docs_metadata.json**: Metadata và text chunks
3. **tfidf_vectorizer.pkl**: Pre-trained TF-IDF vectorizer

## Hybrid Embedding với Alpha = 0.7

Pipeline sử dụng công thức:
```
hybrid_embedding = α × dense_embedding + (1-α) × sparse_embedding
```

Với α = 0.7:
- 70% từ dense embeddings (semantic meaning)
- 30% từ sparse embeddings (keyword matching)

## Cấu hình

Có thể tuỳ chỉnh trong `HybridPDFEmbedder`:

- `chunk_size`: Kích thước chunk tối đa (mặc định: 800)
- `min_chunk_size`: Kích thước chunk tối thiểu (mặc định: 200)
- `similarity_threshold`: Ngưỡng tương đồng ngữ nghĩa (mặc định: 0.5)
- `alpha`: Tỷ lệ dense/sparse (mặc định: 0.7)
- `model`: Model embedding (mặc định: "nomic-embed-text")

## Thống kê

Pipeline cung cấp thống kê chi tiết:
- Số lượng tài liệu xử lý
- Tổng số chunks
- Kích thước trung bình của chunk
- Dimensions của embeddings
- Thời gian xử lý

## Troubleshooting

1. **API Connection Error**: Kiểm tra server embedding tại `http://172.19.20.217:11435`
2. **Memory Issues**: Giảm `chunk_size` hoặc xử lý ít file hơn
3. **No Text Extracted**: Kiểm tra định dạng PDF (có thể là scan image)
4. **TF-IDF Errors**: Đảm bảo có đủ text để fit vectorizer
5. **Low Similarity Scores**: Điều chỉnh `similarity_threshold` (thử 0.2-0.8)

## Test Cosine Similarity

```bash
# Test riêng semantic chunking
python3 test_semantic_chunking.py
```

Script sẽ test:
- Cosine similarity giữa các cặp câu
- Chunking với các threshold khác nhau
- So sánh kết quả chunking

## Ví dụ sử dụng

```python
from ingest.embedder import HybridPDFEmbedder

# Khởi tạo với semantic chunking
embedder = HybridPDFEmbedder(
    folder_path="./documents",
    chunk_size=800,
    min_chunk_size=200,
    similarity_threshold=0.5,
    alpha=0.7
)

# Xử lý
texts, embeddings = embedder.process_documents()

# Lưu kết quả
embedder.save_to_faiss()

# Xem thống kê
stats = embedder.get_stats()
print(stats)
```