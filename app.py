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
        scheme1_file_url = data.get('scheme1_file_url')
        scheme2_file_url = data.get('scheme2_file_url')
        task_id = data.get('task_id')
        
        # 验证必要参数
        if not all([supabase_url, supabase_key, scheme1_file_url, scheme2_file_url, task_id]):
            logger.error(f"请求缺少必要参数: {data}")
            return jsonify({"error": "缺少必要参数"}), 400
        
        logger.info(f"开始处理任务 {task_id}")
        logger.info(f"方案1文件: {scheme1_file_url}")
        logger.info(f"方案2文件: {scheme2_file_url}")
        
        # 异步处理PDF文件
        result = asyncio.run(process_insurance_pdfs(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            scheme1_file_url=scheme1_file_url,
            scheme2_file_url=scheme2_file_url,
            task_id=task_id
        ))
        
        logger.info(f"任务 {task_id} 处理完成: {result}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}", exc_info=True)
        return jsonify({"error": f"处理请求时发生错误: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("正在启动保险方案分析API服务...")
    app.run(host='0.0.0.0', port=8000)