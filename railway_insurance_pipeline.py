#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Railway适用的保险方案处理流水线
====================================

基于correct_complete_pipeline.py修改，适配云环境：
1. PDF从Supabase URL下载而不是本地文件系统
2. 结果上传到Supabase而不是保存到本地文件系统
3. 异步处理支持

作者: MiniMax Agent
日期: 2025-08-07
"""

import pdfplumber
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Any, Optional, Union
import os
from datetime import datetime
import logging
import numpy_financial as npf
import subprocess
import sys
import time
import tempfile
import shutil
import asyncio
import httpx
import mimetypes
import uuid

# 安装Playwright依赖
def install_playwright_dependencies():
    """安装Playwright及其浏览器依赖"""
    try:
        print("开始安装Playwright依赖...")
        # 安装playwright
        subprocess.run(["pip", "install", "playwright"], check=True)
        # 安装chromium浏览器
        subprocess.run(["python", "-m", "playwright", "install", "chromium"], check=True)
        # 安装系统依赖
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "libglib2.0-0", "libnss3", "libatk1.0-0", "libatk-bridge2.0-0", "libcups2", "libdrm2", "libxkbcommon0", "libxcomposite1", "libxdamage1", "libxfixes3", "libxrandr2", "libgbm1", "libasound2"], check=True)
        print("Playwright依赖安装完成")
        return True
    except Exception as e:
        print(f"Playwright依赖安装失败: {e}")
        return False

# 尝试安装Playwright依赖
install_playwright_dependencies()

# Playwright用于截图
try:
    from playwright.sync_api import sync_playwright
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("警告: Playwright未安装，截图功能将不可用")

# Supabase客户端
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("警告: Supabase客户端未安装，云存储功能将不可用")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加正确的IRR计算器类
class CorrectedIRRCalculator:
    """修正的IRR计算器（完全按照原脚本）"""
    
    @staticmethod
    def calculate_irr(cash_flows: List[float], max_iterations: int = 1000, precision: float = 1e-6) -> float:
        """计算内部收益率IRR（完全按照原脚本）"""
        try:
            # 使用numpy_financial计算IRR
            irr = npf.irr(cash_flows)
            
            if np.isnan(irr) or np.isinf(irr):
                # 如果numpy_financial失败，使用自定义算法
                return CorrectedIRRCalculator._custom_irr(cash_flows, max_iterations, precision)
            
            return irr
            
        except Exception as e:
            logger.warning(f"IRR计算失败: {e}")
            return CorrectedIRRCalculator._custom_irr(cash_flows, max_iterations, precision)
    
    @staticmethod
    def _custom_irr(cash_flows: List[float], max_iterations: int = 1000, precision: float = 1e-6) -> float:
        """自定义IRR计算算法（牛顿法）（完全按照原脚本）"""
        try:
            # 检查现金流是否有效
            if len(cash_flows) < 2:
                return 0.0
                
            # 检查是否有正负现金流
            has_positive = any(cf > 0 for cf in cash_flows)
            has_negative = any(cf < 0 for cf in cash_flows)
            
            if not (has_positive and has_negative):
                return 0.0
            
            # 初始猜测
            rate = 0.1
            
            for iteration in range(max_iterations):
                # 计算NPV和NPV导数
                npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
                npv_derivative = sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows))
                
                if abs(npv) < precision:
                    return rate
                
                if abs(npv_derivative) < precision:
                    break
                
                # 牛顿法更新
                new_rate = rate - npv / npv_derivative
                
                # 防止rate变为负数或过大
                if new_rate < -0.99:
                    new_rate = -0.99
                elif new_rate > 10:
                    new_rate = 10
                
                # 检查收敛
                if abs(new_rate - rate) < precision:
                    return new_rate
                    
                rate = new_rate
            
            return rate
            
        except Exception as e:
            logger.warning(f"自定义IRR计算失败: {e}")
            return 0.0

    @staticmethod
    def build_cash_flows_for_surrender(annual_premium: float, payment_period: int, policy_year: int, surrender_value: float) -> List[float]:
        """构建一次性退保的现金流（完全按照原脚本）"""
        cash_flows = []
        
        # 第0年到第(payment_period-1)年：负现金流（保费）
        for year in range(payment_period):
            cash_flows.append(-annual_premium)
        
        # 第payment_period年到第(policy_year-1)年：0现金流
        for year in range(payment_period, policy_year):
            cash_flows.append(0.0)
        
        # 第policy_year年：正现金流（退保价值）
        cash_flows.append(surrender_value)
        
        return cash_flows

    @staticmethod
    def build_cash_flows_for_withdrawal(annual_premium: float, payment_period: int, policy_year: int, 
                                       withdrawal_start_year: int, annual_withdrawal: float, 
                                       final_surrender_value: float) -> List[float]:
        """构建现金提取方案的现金流（完全按照原脚本）"""
        cash_flows = []
        
        # 第0年到第(payment_period-1)年：负现金流（保费）
        for year in range(payment_period):
            cash_flows.append(-annual_premium)
        
        # 第payment_period年到第(withdrawal_start_year-1)年：0现金流
        for year in range(payment_period, withdrawal_start_year):
            cash_flows.append(0.0)
        
        # 第withdrawal_start_year年到第(policy_year-1)年：正现金流（年提取金额）
        for year in range(withdrawal_start_year, policy_year):
            cash_flows.append(annual_withdrawal)
        
        # 第policy_year年：正现金流（年提取金额 + 剩余退保价值）
        cash_flows.append(annual_withdrawal + final_surrender_value)
        
        return cash_flows

class FinalFilteredExtractor:
    """最终过滤版数据提取器 - 适配Railway云环境"""
    
    def __init__(self, pdf_file_paths=None):
        # Railway版本：使用传入的文件路径，不进行自动检测
        if pdf_file_paths and len(pdf_file_paths) >= 2:
            self.scheme1_path = pdf_file_paths[0]
            self.scheme2_path = pdf_file_paths[1]
        else:
            raise ValueError("Railway版本需要明确指定两个PDF文件路径")
        
        # 页面排除规则（完全按照原脚本）
        self.exclude_rules = [
            "说明摘要",      # 第2页排除
            "最高保单贷款",  # 第15页排除
            "悲观情景"       # 第26页排除
        ]
        
    def find_customer_info_page(self, pdf_path: str) -> int:
        """查找包含'保障摘要'的客户信息页面"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and any(keyword in text for keyword in ['保障摘要', '建议书摘要', '保障项目']):
                        return page_num - 1  # 返回页面索引
        except Exception as e:
            logger.error(f"查找客户信息页面时出错: {e}")
        return 0  # 默认返回第一页

    def extract_customer_info(self, pdf_path: str) -> Dict[str, Any]:
        """使用改进的通用定位规则精确提取客户基本信息"""
        customer_info = {
            'name': 'VIP 先生',
            'age': 0,
            'annual_premium': 50000,  # 默认基础保费（整数）
            'payment_period': 5,
            'total_premium': 250000,  # 默认总保费（整数）
            'currency': '美元',
            'coverage_period': '终身'
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # 找到包含客户信息的页面
                page_index = self.find_customer_info_page(pdf_path)
                page = pdf.pages[page_index]
                text = page.extract_text()
                tables = page.extract_tables()
                
                if not text:
                    return customer_info
                
                # 方法1: 从表格中提取客户姓名和年龄
                for table in tables:
                    if not table:
                        continue
                        
                    for row in table:
                        if not row:
                            continue
                        
                        row_text = " ".join([cell.strip() if cell else "" for cell in row])
                        
                        # 客户姓名和年龄（通常在同一行）
                        if "受保人姓名" in row_text and "年龄" in row_text:
                            # 提取姓名
                            import re
                            name_match = re.search(r'受保人姓名[：:]\s*([^年]+)', row_text)
                            if name_match:
                                customer_info['name'] = name_match.group(1).strip()
                            
                            # 提取年龄
                            age_match = re.search(r'年龄[：:]\s*(\d+)', row_text)
                            if age_match:
                                customer_info['age'] = int(age_match.group(1))
                
                # 方法2: 从文本行中提取保单货币
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    
                    # 查找币种信息
                    if "保单货币：美元" in line:
                        customer_info['currency'] = "美元"
                    elif "保单货币：港币" in line:
                        customer_info['currency'] = "港币"
                    elif "保单货币：人民币" in line:
                        customer_info['currency'] = "人民币"
                    elif "美元" in line and "保单货币" in line:
                        customer_info['currency'] = "美元"
                
                # 方法3: 从表格中精确提取基础年缴保费（不含征费）
                for i, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                        
                    # 查找表头包含"年缴保费"的表格
                    if len(table[0]) >= 4 and any("年缴保费" in str(cell) for cell in table[0]):
                        logger.info(f"找到保费表格 {i+1}")
                        # 在数据行中查找保费信息
                        for row_idx, row in enumerate(table[1:], 1):
                            if len(row) >= 4 and row[3]:  # 第4列是年缴保费列
                                premium_cell = str(row[3]).strip()
                                logger.info(f"  第{row_idx}行年缴保费列内容: '{premium_cell}'")
                                
                                # 通用保费提取逻辑 - 支持任何金额格式
                                import re
                                # 查找形如 "XX,XXX.XX" 的金额格式
                                premium_match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', premium_cell)
                                if premium_match:
                                    amount_str = premium_match.group().replace(',', '')
                                    try:
                                        extracted_value = float(amount_str)
                                        customer_info['annual_premium'] = round(extracted_value)  # 四舍五入到个位数
                                        logger.info(f"从表格通用提取基础年保费: {customer_info['annual_premium']} (原值: {amount_str})")
                                        break
                                    except:
                                        pass
                
                # 方法4: 从表格中提取缴费期和保障期
                for table in tables:
                    if not table:
                        continue
                        
                    for row in table:
                        if not row:
                            continue
                        
                        # 查找包含"年"的列，可能是缴费期
                        for cell in row:
                            if not cell or not isinstance(cell, str):
                                continue
                                
                            cell = cell.strip()
                            
                            # 匹配缴费期 (例如 "5年")
                            if re.search(r'(\d+)\s*年\s*交', cell) or re.search(r'缴费期[:：]?\s*(\d+)\s*年', cell):
                                match = re.search(r'(\d+)', cell)
                                if match:
                                    customer_info['payment_period'] = int(match.group(1))
                                    logger.info(f"提取到缴费期: {customer_info['payment_period']}年")
                            
                            # 匹配保障期 (例如 "终身" 或 "至80岁")
                            if "终身" in cell or "whole life" in cell.lower() or "life" in cell.lower():
                                customer_info['coverage_period'] = "终身"
                                logger.info("提取到保障期: 终身")
                            elif re.search(r'至\s*(\d+)\s*[岁歲]', cell):
                                match = re.search(r'至\s*(\d+)\s*[岁歲]', cell)
                                if match:
                                    customer_info['coverage_period'] = f"至{match.group(1)}岁"
                                    logger.info(f"提取到保障期: {customer_info['coverage_period']}")
                
                # 计算总保费
                customer_info['total_premium'] = customer_info['annual_premium'] * customer_info['payment_period']
                
        except Exception as e:
            logger.error(f"提取客户信息时出错: {e}")
            
        return customer_info

    def extract_table_data_by_pattern(self, pdf_path: str, header_patterns: List[str], page_range: List[int] = None) -> List[List[str]]:
        """根据表头模式提取表格数据"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_range is None:
                    pages = pdf.pages
                else:
                    start_page = max(0, page_range[0])
                    end_page = min(len(pdf.pages), page_range[1] if len(page_range) > 1 else start_page + 1)
                    pages = pdf.pages[start_page:end_page]
                
                for page in pages:
                    tables = page.extract_tables()
                    
                    for table in tables:
                        if not table or len(table) < 2:
                            continue
                        
                        # 检查表头是否匹配所有模式
                        header_row = [str(cell).strip() if cell else "" for cell in table[0]]
                        header_text = " ".join(header_row)
                        
                        if all(pattern in header_text for pattern in header_patterns):
                            return table
            
            return []
            
        except Exception as e:
            logger.error(f"按模式提取表格数据时出错: {e}")
            return []

    def find_surrender_value_table_page(self, pdf_path: str) -> int:
        """查找包含退保价值表的页面"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and any(keyword in text for keyword in ['退保价值', '现金价值', '退保金额', 'Surrender', 'Cash Value']):
                        return page_num - 1  # 返回页面索引
        except Exception as e:
            logger.error(f"查找退保价值表页面时出错: {e}")
        return 3  # 默认返回第4页

    def extract_surrender_values(self, pdf_path: str) -> Dict[int, float]:
        """提取退保价值表数据"""
        surrender_values = {}
        try:
            start_page = self.find_surrender_value_table_page(pdf_path)
            
            with pdfplumber.open(pdf_path) as pdf:
                # 搜索连续3页
                for page_idx in range(start_page, min(start_page + 3, len(pdf.pages))):
                    page = pdf.pages[page_idx]
                    tables = page.extract_tables()
                    
                    for table in tables:
                        if not table or len(table) < 2:
                            continue
                            
                        # 判断是否是退保价值表
                        header_row = [str(cell).strip() if cell else "" for cell in table[0]]
                        header_text = " ".join(header_row)
                        
                        if any(keyword in header_text for keyword in ['退保价值', '现金价值', '退保金额', 'Surrender', 'Cash Value']):
                            # 确定年份列和价值列的索引
                            year_col_idx = None
                            value_col_idx = None
                            
                            for i, cell in enumerate(header_row):
                                if '年' in cell or 'Year' in cell:
                                    year_col_idx = i
                                if any(keyword in cell for keyword in ['退保价值', '现金价值', '退保金额', 'Surrender', 'Cash Value']):
                                    value_col_idx = i
                            
                            if year_col_idx is not None and value_col_idx is not None:
                                # 提取数据行
                                for row in table[1:]:
                                    if len(row) <= max(year_col_idx, value_col_idx) or not row[year_col_idx] or not row[value_col_idx]:
                                        continue
                                    
                                    # 提取年份
                                    year_cell = str(row[year_col_idx]).strip()
                                    year_match = re.search(r'(\d+)', year_cell)
                                    if not year_match:
                                        continue
                                    year = int(year_match.group(1))
                                    
                                    # 提取价值
                                    value_cell = str(row[value_col_idx]).strip()
                                    value_match = re.search(r'([\d,]+(?:\.\d+)?)', value_cell.replace(',', ''))
                                    if not value_match:
                                        continue
                                    value = float(value_match.group(1).replace(',', ''))
                                    
                                    surrender_values[year] = value
                                    logger.info(f"提取到第{year}年退保价值: {value}")
                            
                            # 如果已找到退保价值表，直接返回
                            if surrender_values:
                                return surrender_values
            
            if not surrender_values:
                logger.warning(f"未找到退保价值表: {pdf_path}")
            
            return surrender_values
            
        except Exception as e:
            logger.error(f"提取退保价值表时出错: {e}")
            return surrender_values

    def extract_withdrawal_values(self, pdf_path: str) -> Dict[int, float]:
        """提取最优领取方案数据"""
        withdrawal_values = {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # 从第10页开始搜索
                start_page = 9  # 0-indexed
                end_page = min(start_page + 5, len(pdf.pages))
                
                for page_idx in range(start_page, end_page):
                    page = pdf.pages[page_idx]
                    text = page.extract_text()
                    tables = page.extract_tables()
                    
                    # 如果页面文本包含关键词，则可能包含领取方案
                    if text and any(keyword in text for keyword in ['领取方案', '年金', '定期提取', '累积领取']):
                        for table in tables:
                            if not table or len(table) < 2:
                                continue
                                
                            # 合并表头判断是否是领取方案表
                            header_row = [str(cell).strip() if cell else "" for cell in table[0]]
                            header_text = " ".join(header_row)
                            
                            if any(keyword in header_text for keyword in ['领取方案', '年金', '定期提取', '累积领取']):
                                # 确定年份列和领取金额列的索引
                                year_col_idx = None
                                value_col_idx = None
                                
                                for i, cell in enumerate(header_row):
                                    if '年' in cell or 'Year' in cell:
                                        year_col_idx = i
                                    if any(keyword in cell for keyword in ['领取', '提取', '年金', 'Income', 'Withdrawal']):
                                        value_col_idx = i
                                
                                if year_col_idx is not None and value_col_idx is not None:
                                    # 提取数据行
                                    for row in table[1:]:
                                        if len(row) <= max(year_col_idx, value_col_idx) or not row[year_col_idx] or not row[value_col_idx]:
                                            continue
                                        
                                        # 提取年份
                                        year_cell = str(row[year_col_idx]).strip()
                                        year_match = re.search(r'(\d+)', year_cell)
                                        if not year_match:
                                            continue
                                        year = int(year_match.group(1))
                                        
                                        # 提取领取金额
                                        value_cell = str(row[value_col_idx]).strip()
                                        value_match = re.search(r'([\d,]+(?:\.\d+)?)', value_cell.replace(',', ''))
                                        if not value_match:
                                            continue
                                        value = float(value_match.group(1).replace(',', ''))
                                        
                                        withdrawal_values[year] = value
                                        logger.info(f"提取到第{year}年领取金额: {value}")
                                
                                # 如果已找到领取方案表，直接返回
                                if withdrawal_values:
                                    return withdrawal_values
                
            if not withdrawal_values:
                logger.warning(f"未找到领取方案表: {pdf_path}")
            
            return withdrawal_values
            
        except Exception as e:
            logger.error(f"提取领取方案表时出错: {e}")
            return withdrawal_values

    def calculate_average_withdrawal(self, withdrawal_values: Dict[int, float]) -> float:
        """计算平均年提取金额"""
        if not withdrawal_values:
            return 0.0
        
        return sum(withdrawal_values.values()) / len(withdrawal_values)

    def extract_proposal_data(self, scheme1_path: str, scheme2_path: str) -> pd.DataFrame:
        """提取两份保险方案数据并创建对比表"""
        try:
            # 提取方案1的数据
            scheme1_customer = self.extract_customer_info(scheme1_path)
            scheme1_surrender = self.extract_surrender_values(scheme1_path)
            scheme1_withdrawal = self.extract_withdrawal_values(scheme1_path)
            
            # 提取方案2的数据
            scheme2_customer = self.extract_customer_info(scheme2_path)
            scheme2_surrender = self.extract_surrender_values(scheme2_path)
            scheme2_withdrawal = self.extract_withdrawal_values(scheme2_path)
            
            # 创建对比数据框
            data = {
                'scheme1_name': scheme1_customer['name'],
                'scheme1_age': scheme1_customer['age'],
                'scheme1_currency': scheme1_customer['currency'],
                'scheme1_annual_premium': scheme1_customer['annual_premium'],
                'scheme1_payment_period': scheme1_customer['payment_period'],
                'scheme1_total_premium': scheme1_customer['total_premium'],
                'scheme1_coverage_period': scheme1_customer['coverage_period'],
                'scheme1_surrender_values': scheme1_surrender,
                'scheme1_withdrawal_values': scheme1_withdrawal,
                'scheme1_avg_withdrawal': self.calculate_average_withdrawal(scheme1_withdrawal),
                
                'scheme2_name': scheme2_customer['name'],
                'scheme2_age': scheme2_customer['age'],
                'scheme2_currency': scheme2_customer['currency'],
                'scheme2_annual_premium': scheme2_customer['annual_premium'],
                'scheme2_payment_period': scheme2_customer['payment_period'],
                'scheme2_total_premium': scheme2_customer['total_premium'],
                'scheme2_coverage_period': scheme2_customer['coverage_period'],
                'scheme2_surrender_values': scheme2_surrender,
                'scheme2_withdrawal_values': scheme2_withdrawal,
                'scheme2_avg_withdrawal': self.calculate_average_withdrawal(scheme2_withdrawal),
            }
            
            # 创建DataFrame
            df = pd.DataFrame([data])
            
            # 处理列名中的特殊字符
            df.columns = [col.replace(' ', '_').lower() for col in df.columns]
            
            return df
            
        except Exception as e:
            logger.error(f"提取方案数据时出错: {e}")
            raise

class RailwayInsurancePipeline:
    """Railway适用的保险数据处理流水线"""
    
    def __init__(self, pdf_urls=None, supabase_url=None, supabase_key=None, task_id=None):
        """初始化处理流水线"""
        print("\n开始执行Railway版本保险数据处理流水线")
        print("============================================================")
        
        # Supabase配置
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase = None
        
        if SUPABASE_AVAILABLE and supabase_url and supabase_key:
            try:
                self.supabase = create_client(supabase_url, supabase_key)
                print("已连接到Supabase")
            except Exception as e:
                print(f"连接Supabase失败: {e}")
        
        # 任务ID用于标识当前处理任务
        self.task_id = task_id or str(uuid.uuid4())
        print(f"任务ID: {self.task_id}")
        
        # 创建临时目录存储处理过程中的文件
        self.temp_dir = tempfile.mkdtemp(prefix=f"railway_task_{self.task_id}_")
        print(f"临时目录: {self.temp_dir}")
        
        # PDF文件URL
        self.pdf_urls = pdf_urls or []
        if len(self.pdf_urls) < 2:
            raise ValueError("需要至少两个PDF文件URL")
        
        # 设置临时文件路径 - 使用安全的文件名
        def safe_filename(url):
            # 先提取基本文件名
            base_name = os.path.basename(url.split('?')[0])
            # 如果还包含非法字符，则生成随机文件名
            if len(base_name) > 50 or not base_name.endswith('.pdf'):
                return f"{uuid.uuid4().hex}.pdf"
            return base_name
            
        self.scheme1_pdf_path = os.path.join(self.temp_dir, f"scheme1_{safe_filename(self.pdf_urls[0])}")
        self.scheme2_pdf_path = os.path.join(self.temp_dir, f"scheme2_{safe_filename(self.pdf_urls[1])}")
        
        # 设置输出文件路径
        self.extracted_data_file = os.path.join(self.temp_dir, "计划书数据提取结果.xlsx")
        self.irr_results_file = os.path.join(self.temp_dir, "计划书数据提取结果_含IRR计算.xlsx")
        self.final_html_file = os.path.join(self.temp_dir, "report.html")
        self.final_screenshot_file = os.path.join(self.temp_dir, "screenshot.png")
        
        # HTML模板路径（先尝试从当前目录加载，如果不存在，使用内嵌模板）
        self.html_template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "两套方案对比HTML模板_含占位符_修正7.html")
        
    def _cleanup(self):
        """清理临时文件和目录"""
        try:
            shutil.rmtree(self.temp_dir)
            print(f"临时目录已清理: {self.temp_dir}")
        except Exception as e:
            logger.error(f"清理临时目录时出错: {e}")
    
    async def _download_pdf(self, url: str, local_path: str) -> bool:
        """从URL下载PDF文件到本地路径"""
        try:
            # 处理不同类型的URL：Supabase URL或普通HTTP URL
            if self.supabase and url.startswith(self.supabase_url):
                # 如果是Supabase URL，从存储中获取
                # 解析存储路径
                parsed_url = url.replace(f"{self.supabase_url}/storage/v1/object/", "")
                parts = parsed_url.split("/")
                if len(parts) < 2:
                    raise ValueError(f"无效的Supabase存储URL: {url}")
                
                bucket_name = parts[0]
                storage_path = "/".join(parts[1:])
                
                # 下载文件
                try:
                    res = self.supabase.storage.from_(bucket_name).download(storage_path)
                    with open(local_path, 'wb') as f:
                        f.write(res)
                    print(f"从Supabase下载PDF成功: {url} -> {local_path}")
                    return True
                except Exception as e:
                    logger.error(f"从Supabase下载PDF失败: {url}, 错误: {e}")
                    raise
            else:
                # 普通HTTP URL
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(url, follow_redirects=True)
                        response.raise_for_status()
                        
                        # 确保目标目录存在
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        
                        with open(local_path, 'wb') as f:
                            f.write(response.content)
                        
                        print(f"下载PDF成功: {url} -> {local_path}")
                        return True
                except Exception as e:
                    logger.error(f"下载PDF失败: {url}, 错误: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"下载PDF失败: {url}, 错误: {e}")
            raise
    
    async def _upload_file_to_supabase(self, local_path: str, storage_path: str, bucket_name: str = "results") -> Optional[str]:
        """上传文件到Supabase存储并返回公共URL"""
        if not self.supabase:
            logger.error("Supabase客户端未初始化，无法上传文件")
            return None
        
        try:
            # 检查bucket是否存在，不存在则创建
            buckets = self.supabase.storage.list_buckets()
            bucket_exists = any(bucket.name == bucket_name for bucket in buckets)
            
            if not bucket_exists:
                self.supabase.storage.create_bucket(bucket_name, options={'public': True})
                print(f"创建存储桶: {bucket_name}")
            
            # 上传文件
            with open(local_path, 'rb') as f:
                file_content = f.read()
                
            mimetype, _ = mimetypes.guess_type(local_path)
            if not mimetype:
                mimetype = 'application/octet-stream'
            
            # 处理文件名，避免中文或特殊字符问题
            filename = os.path.basename(storage_path)
            storage_path = storage_path.replace(' ', '_').replace(',', '_')
            
            # 上传文件
            self.supabase.storage.from_(bucket_name).upload(
                path=storage_path,
                file=file_content,
                file_options={"content-type": mimetype}
            )
            
            # 获取公共URL
            public_url = f"{self.supabase_url}/storage/v1/object/public/{bucket_name}/{storage_path}"
            print(f"文件已上传: {local_path} -> {public_url}")
            
            return public_url
            
        except Exception as e:
            logger.error(f"文件上传失败: {local_path} -> {storage_path}, 错误: {e}")
            raise
    
    async def step1_extract_pdf_data(self) -> pd.DataFrame:
        """步骤1: 下载PDF并提取数据"""
        print("\n=== 步骤1: PDF数据提取 ===")
        
        # 下载PDF文件
        print("下载PDF文件...")
        pdf_download_tasks = [
            self._download_pdf(self.pdf_urls[0], self.scheme1_pdf_path),
            self._download_pdf(self.pdf_urls[1], self.scheme2_pdf_path)
        ]
        
        results = await asyncio.gather(*pdf_download_tasks)
        if not all(results):
            print("错误: PDF下载失败")
            return pd.DataFrame()
        
        print(f"PDF下载成功: {self.scheme1_pdf_path}, {self.scheme2_pdf_path}")
        
        # 提取数据
        print("提取PDF数据...")
        extractor = FinalFilteredExtractor([self.scheme1_pdf_path, self.scheme2_pdf_path])
        df = extractor.extract_proposal_data(self.scheme1_pdf_path, self.scheme2_pdf_path)
        
        # 保存提取的数据
        df.to_excel(self.extracted_data_file, index=False)
        print(f"数据提取完成，已保存到: {self.extracted_data_file}")
        
        return df
    
    def step2_calculate_irr(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤2: 计算IRR"""
        print("\n=== 步骤2: IRR计算 ===")
        
        if df.empty:
            print("错误: 无数据可计算IRR")
            return df
        
        # 创建IRR计算器
        irr_calculator = CorrectedIRRCalculator()
        
        try:
            # 为每种方案计算IRR
            for scheme_idx, prefix in enumerate(['scheme1', 'scheme2']):
                annual_premium = df.iloc[0][f'{prefix}_annual_premium']
                payment_period = df.iloc[0][f'{prefix}_payment_period']
                surrender_values = df.iloc[0][f'{prefix}_surrender_values']
                withdrawal_values = df.iloc[0][f'{prefix}_withdrawal_values']
                
                # 打印基本信息
                print(f"\n方案{scheme_idx + 1}基本信息:")
                print(f"  年缴保费: {annual_premium}")
                print(f"  缴费期: {payment_period}年")
                print(f"  退保价值条目数: {len(surrender_values)}")
                print(f"  提取方案条目数: {len(withdrawal_values)}")
                
                # 计算10年、15年、20年退保的IRR
                for policy_year in [10, 15, 20]:
                    if policy_year in surrender_values:
                        surrender_value = surrender_values[policy_year]
                        cash_flows = irr_calculator.build_cash_flows_for_surrender(
                            annual_premium=annual_premium,
                            payment_period=payment_period,
                            policy_year=policy_year,
                            surrender_value=surrender_value
                        )
                        
                        irr = irr_calculator.calculate_irr(cash_flows) * 100  # 转换为百分比
                        df.loc[0, f'{prefix}_surrender_irr_{policy_year}y'] = irr
                        
                        print(f"  {policy_year}年退保IRR: {irr:.2f}%")
                
                # 计算最优领取方案的IRR
                if withdrawal_values:
                    # 找到领取开始年份
                    withdrawal_start_year = min(withdrawal_values.keys())
                    
                    # 处理不同的退保终止年份
                    for policy_year in [25, 30]:
                        # 仅计算大于领取开始年份的退保年份
                        if withdrawal_start_year >= policy_year:
                            continue
                            
                        # 找到最接近的退保价值年份
                        available_years = [y for y in surrender_values.keys() if y >= policy_year]
                        if not available_years:
                            continue
                            
                        closest_year = min(available_years, key=lambda y: abs(y - policy_year))
                        final_surrender_value = surrender_values[closest_year]
                        
                        # 计算从领取开始到退保前一年的平均年提取金额
                        withdrawal_years = [y for y in withdrawal_values.keys() 
                                          if y >= withdrawal_start_year and y < policy_year]
                        if not withdrawal_years:
                            continue
                            
                        annual_withdrawal = sum(withdrawal_values[y] for y in withdrawal_years) / len(withdrawal_years)
                        
                        # 构建现金流
                        cash_flows = irr_calculator.build_cash_flows_for_withdrawal(
                            annual_premium=annual_premium,
                            payment_period=payment_period,
                            policy_year=policy_year,
                            withdrawal_start_year=withdrawal_start_year,
                            annual_withdrawal=annual_withdrawal,
                            final_surrender_value=final_surrender_value
                        )
                        
                        irr = irr_calculator.calculate_irr(cash_flows) * 100  # 转换为百分比
                        df.loc[0, f'{prefix}_withdrawal_irr_{policy_year}y'] = irr
                        
                        print(f"  {policy_year}年领取+退保IRR: {irr:.2f}%")
            
            # 保存计算结果
            df.to_excel(self.irr_results_file, index=False)
            print(f"IRR计算完成，已保存到: {self.irr_results_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"计算IRR时出错: {e}")
            import traceback
            traceback.print_exc()
            return df
    
    def calculate_cumulative_withdrawal(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算累计领取金额"""
        try:
            for prefix in ['scheme1', 'scheme2']:
                withdrawal_values = df.iloc[0][f'{prefix}_withdrawal_values']
                if not withdrawal_values:
                    continue
                
                # 计算不同时间点的累计领取金额
                years = sorted(withdrawal_values.keys())
                cumulative = {}
                
                running_sum = 0
                for year in years:
                    running_sum += withdrawal_values[year]
                    cumulative[year] = running_sum
                
                df.loc[0, f'{prefix}_cumulative_withdrawal'] = cumulative
                
                # 计算20年和30年的累计领取金额
                for target_year in [20, 30]:
                    eligible_years = [y for y in years if y <= target_year]
                    if eligible_years:
                        cum_amount = sum(withdrawal_values[y] for y in eligible_years)
                        df.loc[0, f'{prefix}_cumulative_withdrawal_{target_year}y'] = cum_amount
                        print(f"  {prefix} {target_year}年累计领取: {cum_amount}")
            
            return df
            
        except Exception as e:
            logger.error(f"计算累计领取金额时出错: {e}")
            return df
    
    def step3_generate_html(self, irr_df: pd.DataFrame) -> str:
        """步骤3: 生成HTML报告"""
        print("\n=== 步骤3: 生成HTML报告 ===")
        
        if irr_df.empty:
            print("错误: 无数据可生成HTML报告")
            return ""
        
        try:
            # 检查HTML模板是否存在
            if os.path.exists(self.html_template_path):
                print(f"找到HTML模板文件: {self.html_template_path}")
                with open(self.html_template_path, 'r', encoding='utf-8') as f:
                    template_html = f.read()
            else:
                print("错误: HTML模板文件不存在")
                return ""
            
            # 提取基本信息
            scheme1_name = irr_df.iloc[0].get('scheme1_name', 'VIP先生')
            scheme1_age = irr_df.iloc[0].get('scheme1_age', 45)
            scheme1_currency = irr_df.iloc[0].get('scheme1_currency', '美元')
            scheme1_annual_premium = irr_df.iloc[0].get('scheme1_annual_premium', 50000)
            scheme1_payment_period = irr_df.iloc[0].get('scheme1_payment_period', 5)
            scheme1_total_premium = irr_df.iloc[0].get('scheme1_total_premium', 250000)
            
            scheme2_name = irr_df.iloc[0].get('scheme2_name', 'VIP先生')
            scheme2_age = irr_df.iloc[0].get('scheme2_age', 45)
            scheme2_currency = irr_df.iloc[0].get('scheme2_currency', '美元')
            scheme2_annual_premium = irr_df.iloc[0].get('scheme2_annual_premium', 50000)
            scheme2_payment_period = irr_df.iloc[0].get('scheme2_payment_period', 5)
            scheme2_total_premium = irr_df.iloc[0].get('scheme2_total_premium', 250000)
            
            # 提取IRR数据
            scheme1_surrender_irr_10y = irr_df.iloc[0].get('scheme1_surrender_irr_10y', 0)
            scheme1_surrender_irr_15y = irr_df.iloc[0].get('scheme1_surrender_irr_15y', 0)
            scheme1_surrender_irr_20y = irr_df.iloc[0].get('scheme1_surrender_irr_20y', 0)
            scheme1_withdrawal_irr_25y = irr_df.iloc[0].get('scheme1_withdrawal_irr_25y', 0)
            scheme1_withdrawal_irr_30y = irr_df.iloc[0].get('scheme1_withdrawal_irr_30y', 0)
            
            scheme2_surrender_irr_10y = irr_df.iloc[0].get('scheme2_surrender_irr_10y', 0)
            scheme2_surrender_irr_15y = irr_df.iloc[0].get('scheme2_surrender_irr_15y', 0)
            scheme2_surrender_irr_20y = irr_df.iloc[0].get('scheme2_surrender_irr_20y', 0)
            scheme2_withdrawal_irr_25y = irr_df.iloc[0].get('scheme2_withdrawal_irr_25y', 0)
            scheme2_withdrawal_irr_30y = irr_df.iloc[0].get('scheme2_withdrawal_irr_30y', 0)
            
            # 提取退保价值数据
            scheme1_surrender_values = irr_df.iloc[0].get('scheme1_surrender_values', {})
            scheme2_surrender_values = irr_df.iloc[0].get('scheme2_surrender_values', {})
            
            # 提取领取方案数据
            scheme1_withdrawal_values = irr_df.iloc[0].get('scheme1_withdrawal_values', {})
            scheme2_withdrawal_values = irr_df.iloc[0].get('scheme2_withdrawal_values', {})
            
            # 提取累计领取金额
            scheme1_cum_withdrawal_20y = irr_df.iloc[0].get('scheme1_cumulative_withdrawal_20y', 0)
            scheme1_cum_withdrawal_30y = irr_df.iloc[0].get('scheme1_cumulative_withdrawal_30y', 0)
            scheme2_cum_withdrawal_20y = irr_df.iloc[0].get('scheme2_cumulative_withdrawal_20y', 0)
            scheme2_cum_withdrawal_30y = irr_df.iloc[0].get('scheme2_cumulative_withdrawal_30y', 0)
            
            # 计算IRR差值
            irr_diff_10y = scheme1_surrender_irr_10y - scheme2_surrender_irr_10y
            irr_diff_15y = scheme1_surrender_irr_15y - scheme2_surrender_irr_15y
            irr_diff_20y = scheme1_surrender_irr_20y - scheme2_surrender_irr_20y
            irr_diff_25y = scheme1_withdrawal_irr_25y - scheme2_withdrawal_irr_25y
            irr_diff_30y = scheme1_withdrawal_irr_30y - scheme2_withdrawal_irr_30y
            
            # 生成IRR对比HTML
            irr_comparison_html = f"""
            <tr><th>10年退保</th><td>{scheme1_surrender_irr_10y:.2f}%</td><td>{scheme2_surrender_irr_10y:.2f}%</td><td>{irr_diff_10y:.2f}%</td></tr>
            <tr><th>15年退保</th><td>{scheme1_surrender_irr_15y:.2f}%</td><td>{scheme2_surrender_irr_15y:.2f}%</td><td>{irr_diff_15y:.2f}%</td></tr>
            <tr><th>20年退保</th><td>{scheme1_surrender_irr_20y:.2f}%</td><td>{scheme2_surrender_irr_20y:.2f}%</td><td>{irr_diff_20y:.2f}%</td></tr>
            <tr><th>25年领取+退保</th><td>{scheme1_withdrawal_irr_25y:.2f}%</td><td>{scheme2_withdrawal_irr_25y:.2f}%</td><td>{irr_diff_25y:.2f}%</td></tr>
            <tr><th>30年领取+退保</th><td>{scheme1_withdrawal_irr_30y:.2f}%</td><td>{scheme2_withdrawal_irr_30y:.2f}%</td><td>{irr_diff_30y:.2f}%</td></tr>
            """
            
            # 生成退保价值对比表格HTML
            surrender_years = sorted(set(list(scheme1_surrender_values.keys()) + list(scheme2_surrender_values.keys())))
            surrender_table_rows = []
            
            for year in surrender_years:
                scheme1_value = scheme1_surrender_values.get(year, 0)
                scheme2_value = scheme2_surrender_values.get(year, 0)
                value_diff = scheme1_value - scheme2_value
                
                # 计算回本率
                scheme1_return_rate = (scheme1_value / scheme1_total_premium * 100) if scheme1_total_premium else 0
                scheme2_return_rate = (scheme2_value / scheme2_total_premium * 100) if scheme2_total_premium else 0
                return_rate_diff = scheme1_return_rate - scheme2_return_rate
                
                row = f"""
                <tr>
                    <td>{year}</td>
                    <td>{scheme1_value:,.0f}</td>
                    <td>{scheme2_value:,.0f}</td>
                    <td>{value_diff:,.0f}</td>
                    <td>{scheme1_return_rate:.2f}%</td>
                    <td>{scheme2_return_rate:.2f}%</td>
                    <td>{return_rate_diff:.2f}%</td>
                </tr>
                """
                surrender_table_rows.append(row)
            
            surrender_comparison_html = "\n".join(surrender_table_rows)
            
            # 生成领取方案对比表格HTML
            withdrawal_years = sorted(set(list(scheme1_withdrawal_values.keys()) + list(scheme2_withdrawal_values.keys())))
            withdrawal_table_rows = []
            
            # 初始化累计数据
            scheme1_cum = 0
            scheme2_cum = 0
            
            for year in withdrawal_years:
                scheme1_value = scheme1_withdrawal_values.get(year, 0)
                scheme2_value = scheme2_withdrawal_values.get(year, 0)
                value_diff = scheme1_value - scheme2_value
                
                # 计算累计领取金额
                scheme1_cum += scheme1_value
                scheme2_cum += scheme2_value
                cum_diff = scheme1_cum - scheme2_cum
                
                row = f"""
                <tr>
                    <td>{year}</td>
                    <td>{scheme1_value:,.0f}</td>
                    <td>{scheme2_value:,.0f}</td>
                    <td>{value_diff:,.0f}</td>
                    <td>{scheme1_cum:,.0f}</td>
                    <td>{scheme2_cum:,.0f}</td>
                    <td>{cum_diff:,.0f}</td>
                </tr>
                """
                withdrawal_table_rows.append(row)
            
            withdrawal_comparison_html = "\n".join(withdrawal_table_rows)
            
            # 生成累计领取金额HTML
            cumulative_withdrawal_html = f"""
            <tr><th>20年累计领取</th><td>{scheme1_cum_withdrawal_20y:,.0f}</td><td>{scheme2_cum_withdrawal_20y:,.0f}</td><td>{scheme1_cum_withdrawal_20y - scheme2_cum_withdrawal_20y:,.0f}</td></tr>
            <tr><th>30年累计领取</th><td>{scheme1_cum_withdrawal_30y:,.0f}</td><td>{scheme2_cum_withdrawal_30y:,.0f}</td><td>{scheme1_cum_withdrawal_30y - scheme2_cum_withdrawal_30y:,.0f}</td></tr>
            """
            
            # 获取当前日期
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # 替换模板中的占位符
            html_content = template_html.replace("{{currentDate}}", current_date)
            
            # 基本信息替换
            html_content = html_content.replace("{{customerName}}", scheme1_name)
            html_content = html_content.replace("{{customerAge}}", str(scheme1_age))
            
            # 方案1信息替换
            html_content = html_content.replace("{{scheme1Currency}}", scheme1_currency)
            html_content = html_content.replace("{{scheme1AnnualPremium}}", f"{scheme1_annual_premium:,.0f}")
            html_content = html_content.replace("{{scheme1PaymentPeriod}}", str(scheme1_payment_period))
            html_content = html_content.replace("{{scheme1TotalPremium}}", f"{scheme1_total_premium:,.0f}")
            
            # 方案2信息替换
            html_content = html_content.replace("{{scheme2Currency}}", scheme2_currency)
            html_content = html_content.replace("{{scheme2AnnualPremium}}", f"{scheme2_annual_premium:,.0f}")
            html_content = html_content.replace("{{scheme2PaymentPeriod}}", str(scheme2_payment_period))
            html_content = html_content.replace("{{scheme2TotalPremium}}", f"{scheme2_total_premium:,.0f}")
            
            # 表格数据替换
            html_content = html_content.replace("{{irrComparisonRows}}", irr_comparison_html)
            html_content = html_content.replace("{{surrenderValueRows}}", surrender_comparison_html)
            html_content = html_content.replace("{{withdrawalPlanRows}}", withdrawal_comparison_html)
            html_content = html_content.replace("{{cumulativeWithdrawalRows}}", cumulative_withdrawal_html)
            
            # 方案3数据处理 - 可选特性
            # 这里我们处理报告中的方案3部分，将其设置为与方案1相同（或者隐藏）
            # 因为我们只有方案1和方案2的数据
            html_content = html_content.replace("{{scheme3Currency}}", scheme1_currency)
            html_content = html_content.replace("{{scheme3AnnualPremium}}", f"{scheme1_annual_premium:,.0f}")
            html_content = html_content.replace("{{scheme3PaymentPeriod}}", str(scheme1_payment_period))
            html_content = html_content.replace("{{scheme3TotalPremium}}", f"{scheme1_total_premium:,.0f}")
            
            # 保存HTML报告
            with open(self.final_html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTML报告生成完成，已保存到: {self.final_html_file}")
            
            return self.final_html_file
            
        except Exception as e:
            logger.error(f"生成HTML报告时出错: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    async def step4_generate_screenshot(self, html_file: str) -> str:
        """步骤4: 生成网页截图"""
        print("\n=== 步骤4: 生成网页截图 ===")
        
        if not html_file or not os.path.exists(html_file):
            print("错误: HTML文件不存在，无法生成截图")
            return ""
        
        if not PLAYWRIGHT_AVAILABLE:
            print("错误: Playwright未安装，无法生成截图")
            return ""
        
        try:
            # 使用Playwright异步API
            async with async_playwright() as p:
                # 启动浏览器
                browser = await p.chromium.launch()
                
                # 创建新页面
                page = await browser.new_page()
                
                # 设置视口大小
                await page.set_viewport_size({"width": 1280, "height": 1600})
                
                # 加载HTML文件
                file_url = f"file://{os.path.abspath(html_file)}"
                print(f"加载HTML文件: {file_url}")
                await page.goto(file_url)
                
                # 等待页面加载完成
                await page.wait_for_load_state("networkidle")
                
                # 生成截图
                await page.screenshot(path=self.final_screenshot_file, full_page=True)
                
                # 关闭浏览器
                await browser.close()
                
                print(f"截图生成完成，已保存到: {self.final_screenshot_file}")
                
                return self.final_screenshot_file
                
        except Exception as e:
            logger.error(f"生成截图时出错: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    async def run_complete_pipeline(self) -> Dict[str, str]:
        """运行完整的处理流水线"""
        try:
            # 步骤1: 提取PDF数据
            df = await self.step1_extract_pdf_data()
            if df.empty:
                raise Exception("PDF数据提取失败")
            
            # 步骤2: 计算IRR
            irr_df = self.step2_calculate_irr(df)
            
            # 计算累计领取金额
            irr_df = self.calculate_cumulative_withdrawal(irr_df)
            
            # 步骤3: 生成HTML报告
            html_file = self.step3_generate_html(irr_df)
            if not html_file:
                raise Exception("HTML报告生成失败")
            
            # 步骤4: 生成截图
            screenshot_file = await self.step4_generate_screenshot(html_file)
            
            # 步骤5: 上传结果到Supabase
            result_urls = {}
            
            if self.supabase:
                # 上传Excel数据
                excel_url = await self._upload_file_to_supabase(
                    self.irr_results_file,
                    f"{self.task_id}/data.xlsx"
                )
                if excel_url:
                    result_urls['excel_url'] = excel_url
                
                # 上传HTML报告
                html_url = await self._upload_file_to_supabase(
                    self.final_html_file,
                    f"{self.task_id}/report.html"
                )
                if html_url:
                    result_urls['html_url'] = html_url
                
                # 上传截图
                if screenshot_file and os.path.exists(screenshot_file):
                    screenshot_url = await self._upload_file_to_supabase(
                        self.final_screenshot_file,
                        f"{self.task_id}/screenshot.png"
                    )
                    if screenshot_url:
                        result_urls['screenshot_url'] = screenshot_url
            
            # 保存结果到数据库（如果Supabase客户端可用）
            if self.supabase:
                try:
                    self.supabase.table('tasks').update({
                        'status': 'completed',
                        'results': result_urls,
                        'completed_at': datetime.now().isoformat()
                    }).eq('id', self.task_id).execute()
                    print(f"任务结果已更新到数据库: {self.task_id}")
                except Exception as e:
                    logger.error(f"更新任务状态时出错: {e}")
            
            # 清理临时文件
            self._cleanup()
            
            return result_urls
            
        except Exception as e:
            print(f"流水线执行过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 更新任务状态为失败（如果Supabase客户端可用）
            if self.supabase:
                try:
                    self.supabase.table('tasks').update({
                        'status': 'failed',
                        'error_message': str(e),
                        'completed_at': datetime.now().isoformat()
                    }).eq('id', self.task_id).execute()
                    print(f"任务失败状态已更新到数据库: {self.task_id}")
                except Exception as db_err:
                    logger.error(f"更新任务失败状态时出错: {db_err}")
            
            # 清理临时文件
            self._cleanup()
            
            # 如果是直接API调用，返回错误信息
            return {"error": str(e)}

async def process_insurance_pdfs(pdf_urls, supabase_url=None, supabase_key=None, task_id=None) -> Dict[str, str]:
    """
    处理保险PDF文件并返回结果URLs
    
    参数:
    pdf_urls: PDF文件URL列表（至少2个）
    supabase_url: Supabase URL（可选）
    supabase_key: Supabase密钥（可选）
    task_id: 任务ID（可选，如果未提供将自动生成）
    
    返回:
    Dict[str, str]: 包含生成的文件URLs的字典
    """
    # 创建并运行处理流水线
    pipeline = RailwayInsurancePipeline(
        pdf_urls=pdf_urls,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        task_id=task_id
    )
    
    return await pipeline.run_complete_pipeline()

# 直接运行测试
if __name__ == "__main__":
    # 获取命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='处理保险PDF文件')
    parser.add_argument('--pdf1', type=str, help='第一个PDF文件路径或URL')
    parser.add_argument('--pdf2', type=str, help='第二个PDF文件路径或URL')
    parser.add_argument('--supabase_url', type=str, help='Supabase URL')
    parser.add_argument('--supabase_key', type=str, help='Supabase密钥')
    parser.add_argument('--task_id', type=str, help='任务ID')
    
    args = parser.parse_args()
    
    if not args.pdf1 or not args.pdf2:
        print("错误: 需要提供两个PDF文件路径或URL")
        sys.exit(1)
    
    # 运行处理流水线
    result = asyncio.run(process_insurance_pdfs(
        pdf_urls=[args.pdf1, args.pdf2],
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
        task_id=args.task_id
    ))
    
    print("\n=== 处理结果 ===")
    for key, value in result.items():
        print(f"{key}: {value}")