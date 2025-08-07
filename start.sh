#!/bin/bash

# 安装Playwright浏览器 - 仅安装Chromium以节省空间和时间
echo "安装Playwright浏览器..."
python -m playwright install chromium

# 安装可能缺少的系统依赖
echo "安装系统依赖..."
apt-get update && apt-get install -y libglib2.0-0 libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2

# 验证安装是否成功
echo "验证Playwright安装..."
python -c "
import asyncio
from playwright.async_api import async_playwright

async def verify_installation():
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            await browser.close()
            print('Playwright浏览器安装验证成功!')
    except Exception as e:
        print(f'Playwright浏览器安装验证失败: {e}')

asyncio.run(verify_installation())
"

# 启动应用
echo "启动应用..."
python app.py