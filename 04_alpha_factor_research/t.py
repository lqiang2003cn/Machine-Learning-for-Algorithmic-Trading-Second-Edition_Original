from io import BytesIO

# 打开文件并读取其内容
with open('file.txt', 'rb') as file:
    # 创建BytesIO对象
    byte_stream = BytesIO(file.read())

    # 逐行读取字节数据
    while True:
        line = byte_stream.readline()

        if not line:
            break

        print(line)