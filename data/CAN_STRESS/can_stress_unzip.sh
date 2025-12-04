#!/bin/bash
# 批量解压当前目录下所有 zip 文件到对应的文件夹中

for file in *.zip; do
    echo "正在解压：$file"
    unzip -o "\home\xjc\data\CAN_Stress\sci_unzip\\$file" -d "${file%.zip}"
done

echo "全部解压完成！"