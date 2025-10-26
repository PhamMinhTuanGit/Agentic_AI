# ZebOS-XP HTML Documentation Parser

Parser Python để trích xuất thông tin từ tài liệu HTML của ZebOS-XP 1.4.

## Tính năng

Parser này có thể trích xuất:

✅ **Thông tin lệnh (Commands)**:
- Tên lệnh
- Mô tả chi tiết
- Cú pháp (syntax)
- Tham số và giải thích
- Chế độ cấu hình (Command Mode)
- Ví dụ sử dụng
- Chương/phần thuộc về

✅ **Thông tin chương (Chapters)**:
- Tiêu đề chương
- Số thứ tự chương
- Mô tả/giới thiệu
- Danh sách các lệnh trong chương

✅ **Chức năng bổ sung**:
- Tìm kiếm lệnh theo từ khóa
- Lọc lệnh theo chương
- Xuất dữ liệu ra JSON
- Thống kê tài liệu

## Cài đặt

### 1. Cài đặt thư viện cần thiết

```bash
pip install beautifulsoup4 lxml
```

### 2. Cấu trúc thư mục

```
Agentic/
├── parsers/
│   ├── __init__.py
│   ├── zebos_html_parser.py      # Parser chính
│   └── demo_zebos_parser.py      # Ví dụ sử dụng
├── ZebOS-XP_1.4_HTML/
│   ├── index.html
│   └── ZebOS-XP 1.4/
│       ├── AAA Commands.603.01.html
│       ├── AAA Commands.603.02.html
│       └── ... (nhiều file khác)
└── README_ZEBOS_PARSER.md        # File này
```

## Sử dụng

### Ví dụ cơ bản

```python
from parsers.zebos_html_parser import ZebOSHTMLParser

# Khởi tạo parser
docs_dir = "/path/to/ZebOS-XP_1.4_HTML"
parser = ZebOSHTMLParser(docs_dir)

# Parse tất cả lệnh và lưu ra JSON
commands = parser.parse_all_commands(output_file="all_commands.json")

# Parse tất cả chương
chapters = parser.parse_all_chapters(output_file="all_chapters.json")
```

### Ví dụ parse một file

```python
from pathlib import Path
from parsers.zebos_html_parser import ZebOSHTMLParser

parser = ZebOSHTMLParser("/path/to/ZebOS-XP_1.4_HTML")

# Parse một file lệnh cụ thể
file_path = Path("/path/to/ZebOS-XP_1.4_HTML/ZebOS-XP 1.4/AAA Commands.603.02.html")
cmd_info = parser.parse_command_file(file_path)

if cmd_info:
    print(f"Command: {cmd_info.name}")
    print(f"Description: {cmd_info.description}")
    print(f"Syntax: {cmd_info.syntax}")
    print(f"Parameters: {cmd_info.parameters}")
```

### Tìm kiếm lệnh

```python
# Parse tất cả lệnh
commands = parser.parse_all_commands()

# Tìm kiếm theo từ khóa
aaa_commands = parser.search_commands("aaa", commands)
print(f"Found {len(aaa_commands)} AAA commands")

for cmd in aaa_commands:
    print(f"  - {cmd.name}")
```

### Lọc lệnh theo chương

```python
# Lấy tất cả lệnh từ một chương cụ thể
auth_commands = parser.get_commands_by_chapter("Authentication", commands)

for cmd in auth_commands:
    print(f"  - {cmd.name}: {cmd.description}")
```

### Chạy demo

```bash
cd /path/to/Agentic
python parsers/demo_zebos_parser.py
```

## Cấu trúc dữ liệu

### CommandInfo

```python
@dataclass
class CommandInfo:
    name: str                          # Tên lệnh
    description: str                   # Mô tả
    syntax: List[str]                  # Cú pháp
    parameters: List[Dict[str, str]]   # Tham số
    mode: str                          # Chế độ cấu hình
    examples: List[str]                # Ví dụ
    file_path: str                     # Đường dẫn file
    chapter: Optional[str]             # Tên chương
```

### ChapterInfo

```python
@dataclass
class ChapterInfo:
    title: str                         # Tiêu đề chương
    chapter_number: Optional[str]      # Số thứ tự
    introduction: str                  # Giới thiệu
    commands: List[str]                # Danh sách lệnh
    file_path: str                     # Đường dẫn file
```

## Kết quả JSON

### Ví dụ CommandInfo JSON

```json
{
  "name": "aaa accounting default",
  "description": "Use this command to set the AAA methods for accounting.",
  "syntax": [
    "aaa accounting default ((group LINE) | local)",
    "no aaa accounting default ((group LINE) | local)"
  ],
  "parameters": [
    {
      "name": "group",
      "description": "Use a server group list for authentication"
    },
    {
      "name": "LINE",
      "description": "Specify a space-separated list of up to 8 configured RADIUS or TACACS+ server group names..."
    }
  ],
  "mode": "Configure mode",
  "examples": [
    "#configure terminal",
    "(config)#aaa accounting default group radius"
  ],
  "file_path": "ZebOS-XP 1.4/AAA Commands.603.02.html",
  "chapter": "System Management Command Reference"
}
```

## Ứng dụng

Parser này có thể được sử dụng để:

1. **Xây dựng RAG system**: Tích hợp vào hệ thống RAG để trả lời câu hỏi về ZebOS commands
2. **Tạo chatbot**: Xây dựng chatbot hỗ trợ cấu hình ZebOS
3. **Tạo documentation database**: Lưu trữ tài liệu dạng có cấu trúc
4. **Command suggestion**: Gợi ý lệnh dựa trên mô tả
5. **Training data**: Tạo dữ liệu huấn luyện cho LLM về networking

## Mở rộng

Có thể mở rộng parser để:

- Parse thêm loại file khác (Configuration files, API docs, etc.)
- Trích xuất thêm metadata (links, references, related commands)
- Xử lý đặc biệt cho các loại lệnh khác nhau
- Tích hợp với embedding models để tạo vector database
- Tạo knowledge graph từ các mối quan hệ giữa các lệnh

## Lưu ý

- Parser sử dụng BeautifulSoup nên cần cài đặt `beautifulsoup4`
- Đường dẫn đến thư mục ZebOS-XP_1.4_HTML cần được cấu hình đúng
- Một số file có thể có cấu trúc khác nhau, parser đã xử lý các trường hợp phổ biến

## Tác giả

Phát triển cho dự án Agentic AI - Network Configuration Assistant
