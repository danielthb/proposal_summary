#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask API服务
为保险方案分析脚本提供API接口
"""

from flask import Flask, request, jsonify
from railway_insurance_pipeline import process_insurance_pdfs
import asyncio
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "healthy", "message": "保险方案分析服务正常运行"}), 200

@app.route('/process', methods=['POST'])
def process():
    """处理保险PDF文件"""
    try:
        data = request.json
        
        # 获取请求数据
        supabase_url = data.get('supabase_url')
        supabase_key = data.get('supabase_key')
        pdf_urls = data.get('pdf_urls', [])
        task_id = data.get('task_id')
        
        # 兼容旧的API接口
        scheme1_file_url = data.get('scheme1_file_url')
        scheme2_file_url = data.get('scheme2_file_url')
        
        # 如果使用旧接口，合并PDF URL
        if scheme1_file_url and scheme2_file_url and not pdf_urls:
            pdf_urls = [scheme1_file_url, scheme2_file_url]
        
        # 验证必要参数
        if not all([supabase_url, supabase_key, task_id]) or len(pdf_urls) < 2:
            logger.error(f"请求缺少必要参数: {data}")
            missing = []
            if not supabase_url:
                missing.append("supabase_url")
            if not supabase_key:
                missing.append("supabase_key")
            if not task_id:
                missing.append("task_id")
            if len(pdf_urls) < 2:
                missing.append("pdf_urls (至少需要2个)")
            return jsonify({"error": f"缺少必要参数: {', '.join(missing)}"}), 400
        
        logger.info(f"开始处理任务 {task_id}")
        logger.info(f"PDF文件: {pdf_urls}")
        
        # 异步处理PDF文件
        result = asyncio.run(process_insurance_pdfs(
            pdf_urls=pdf_urls,
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            task_id=task_id
        ))
        
        logger.info(f"任务 {task_id} 处理完成: {result}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}", exc_info=True)
        return jsonify({"error": f"处理请求时发生错误: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("正在启动保险方案分析API服务...")
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)