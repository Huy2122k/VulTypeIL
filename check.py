import re

pattern = r'datetime="(.*?)"'

# Tìm kiếm mẫu trong nội dung HTML
match = re.search(pattern, a.text)

if match:
    # Group 1 (.*?) chứa giá trị mà chúng ta muốn trích xuất
    datetime_value = match.group(1)
    print(f"Giá trị datetime (Regex): {datetime_value}")
else:
    print("Không tìm thấy thuộc tính 'datetime'.")