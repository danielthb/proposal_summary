#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Railway适用的保险方案处理流水线（完整修复版）
====================================

基于correct_complete_pipeline.py修改，适配云环境：
1. PDF从Supabase URL下载而不是本地文件系统
2. 结果上传到Supabase而不是保存到本地文件系统
3. 异步处理支持

修复以下问题：
1. 文件名过长问题 - 通过安全地解析URL路径
2. 数据库列名错误 - error_message而不是error
3. Playwright浏览器不存在问题 - 通过优雅地处理错误，跳过截图但不中断整个流程

作者: MiniMax Agent
日期: 2025-08-08
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
from urllib.parse import urlparse, unquote

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
                            if cell and isinstance(cell, str):
                                if "5" in cell and "终身" in " ".join(row):
                                    customer_info['payment_period'] = 5
                                if "终身" in cell:
                                    customer_info['coverage_period'] = "终身"
                
                # 设置默认值
                if customer_info['payment_period'] is None:
                    customer_info['payment_period'] = 5     # 默认5年
                if customer_info['coverage_period'] is None:
                    customer_info['coverage_period'] = "终身"  # 默认终身
                
                # 计算总保费（基础保费 × 缴费年数）
                if customer_info['annual_premium'] and customer_info['payment_period']:
                    total_premium = customer_info['annual_premium'] * customer_info['payment_period']
                    customer_info['total_premium'] = round(total_premium)  # 四舍五入到整数
                    logger.info(f"计算总保费: {customer_info['annual_premium']} × {customer_info['payment_period']} = {customer_info['total_premium']}")
                
                print(f"改进版提取客户信息: 姓名={customer_info['name']}, 年龄={customer_info['age']}, 年缴保费=${customer_info['annual_premium']:,}")
                logger.info(f"客户信息: {customer_info}")
                
        except Exception as e:
            logger.error(f"提取客户信息时出错: {e}")
            
        return customer_info
    
    def should_exclude_page(self, text: str) -> bool:
        """检查页面是否应该被排除（完全按照原脚本）"""
        for exclude_keyword in self.exclude_rules:
            if exclude_keyword in text:
                return True
        return False
    
    def find_table_pages(self, pdf_path: str) -> Dict[str, List[int]]:
        """查找包含特定表格的页面，并应用排除规则（完全按照原脚本）"""
        table_pages = {
            'surrender_value': [],  # 退保发还金额
            'withdrawal_surrender_value': [],  # 现金提取后之退保发还金额
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # 首先检查是否应该排除此页面
                    if self.should_exclude_page(text):
                        logger.info(f"第{page_num}页包含排除关键字，跳过")
                        continue
                    
                    # 然后检查是否包含目标表格
                    if "退保发还金额" in text and "现金提取后之退保发还金额" not in text:
                        table_pages['surrender_value'].append(page_num)
                        logger.info(f"第{page_num}页包含'退保发还金额'表格")
                    elif "现金提取后之退保发还金额" in text:
                        table_pages['withdrawal_surrender_value'].append(page_num)
                        logger.info(f"第{page_num}页包含'现金提取后之退保发还金额'表格")
                        
        except Exception as e:
            logger.error(f"查找表格页面时出错: {e}")
            
        return table_pages
    
    def parse_newline_separated_data(self, cell_content: str) -> List[str]:
        """解析用换行符分隔的5年数据（完全按照原脚本）"""
        if not cell_content or pd.isna(cell_content):
            return []
        
        parts = str(cell_content).split('\n')
        cleaned_parts = []
        for part in parts:
            part = part.strip()
            if part:
                part = part.replace(',', '')
                cleaned_parts.append(part)
        
        return cleaned_parts[:5]
    
    def find_column_indices(self, table: List[List], table_type: str) -> Dict[str, int]:
        """精确查找列索引（完全按照原脚本）"""
        column_indices = {}
        
        if not table or len(table) < 2:
            return column_indices
        
        # 显示表格结构用于调试
        logger.info(f"表格结构分析 ({table_type}):")
        for row_idx, row in enumerate(table[:4]):
            logger.info(f"  第{row_idx + 1}行: {row}")
        
        # 查找年龄列（第1列，固定）
        column_indices['age'] = 0
        
        # 查找保单年度终结列（在第1行中查找）
        if len(table) > 0:
            header_row1 = table[0]
            for col_idx, cell in enumerate(header_row1):
                if cell and '保单' in str(cell) and '年度' in str(cell):
                    column_indices['policy_year'] = col_idx
                    logger.info(f"找到保单年度终结列: 第{col_idx + 1}列")
                    break
            
            # 如果没找到，使用默认位置（通常是第2列）
            if 'policy_year' not in column_indices:
                column_indices['policy_year'] = 1
                logger.info(f"使用默认保单年度终结列: 第2列")
        
        # 查找总额列（在第2行中查找）
        if len(table) > 1:
            header_row2 = table[1]
            for col_idx, cell in enumerate(header_row2):
                if cell and '总额' in str(cell):
                    column_indices['total_amount'] = col_idx
                    logger.info(f"找到总额列: 第{col_idx + 1}列")
                    break
        
        # 查找现金提取金额列（在第1行中查找）
        if len(table) > 0:
            header_row1 = table[0]
            for col_idx, cell in enumerate(header_row1):
                if cell and '现金提取' in str(cell) and '金额' in str(cell):
                    column_indices['withdrawal_amount'] = col_idx
                    logger.info(f"找到现金提取金额列: 第{col_idx + 1}列")
                    break
        
        logger.info(f"最终列索引: {column_indices}")
        return column_indices
    
    def extract_table_data_filtered(self, pdf_path: str, page_numbers: List[int], table_type: str) -> List[Dict]:
        """从指定页面提取过滤后的表格数据（完全按照原脚本）"""
        all_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in page_numbers:
                    if page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]
                        tables = page.extract_tables()
                        
                        logger.info(f"第{page_num}页包含 {len(tables)} 个表格")
                        
                        for table_idx, table in enumerate(tables):
                            if not table or len(table) < 3:
                                continue
                            
                            # 查找列索引
                            column_indices = self.find_column_indices(table, table_type)
                            
                            if not column_indices:
                                logger.warning(f"第{page_num}页表格{table_idx + 1}无法找到有效列索引")
                                continue
                            
                            # 从第3行开始提取数据（跳过表头）
                            for row_idx in range(2, len(table)):
                                row = table[row_idx]
                                
                                if len(row) <= max(column_indices.values()):
                                    continue
                                
                                # 提取各列数据
                                age_cell = row[column_indices['age']] if 'age' in column_indices else None
                                policy_year_cell = row[column_indices['policy_year']] if 'policy_year' in column_indices else None
                                total_amount_cell = row[column_indices['total_amount']] if 'total_amount' in column_indices else None
                                withdrawal_cell = row[column_indices['withdrawal_amount']] if 'withdrawal_amount' in column_indices else None
                                
                                # 解析各列的5年数据
                                age_values = self.parse_newline_separated_data(age_cell)
                                policy_year_values = self.parse_newline_separated_data(policy_year_cell)
                                total_amount_values = self.parse_newline_separated_data(total_amount_cell)
                                withdrawal_values = self.parse_newline_separated_data(withdrawal_cell) if withdrawal_cell else []
                                
                                logger.info(f"第{row_idx + 1}行数据:")
                                logger.info(f"  年龄: {age_values}")
                                logger.info(f"  保单年度: {policy_year_values}")
                                logger.info(f"  总额: {total_amount_values}")
                                logger.info(f"  现金提取: {withdrawal_values}")
                                
                                # 验证数据一致性
                                if not age_values or not policy_year_values or not total_amount_values:
                                    logger.warning(f"第{row_idx + 1}行数据不完整，跳过")
                                    continue
                                
                                # 确保数据长度一致（都应该是5个）
                                min_length = min(len(age_values), len(policy_year_values), len(total_amount_values))
                                min_length = min(min_length, 5)
                                
                                # 生成数据记录
                                for year_offset in range(min_length):
                                    try:
                                        age = int(age_values[year_offset])
                                        policy_year = int(policy_year_values[year_offset])
                                        
                                        # 检查是否是每5年汇总行
                                        if policy_year % 5 == 0 or policy_year == 1:
                                            total_amount = float(total_amount_values[year_offset])
                                            
                                            data = {
                                                'age': age,
                                                'policy_year': policy_year,
                                                'total_amount': total_amount
                                            }
                                            
                                            # 如果有现金提取数据，添加
                                            if withdrawal_values and year_offset < len(withdrawal_values) and withdrawal_values[year_offset]:
                                                try:
                                                    data['withdrawal_amount'] = float(withdrawal_values[year_offset])
                                                except:
                                                    pass
                                            
                                            all_data.append(data)
                                    except Exception as e:
                                        logger.error(f"解析第{row_idx + 1}行数据时出错: {e}")
        
        except Exception as e:
            logger.error(f"提取表格数据时出错: {e}")
        
        return all_data
    
    def extract_surrender_values(self, pdf_path: str) -> Dict[int, float]:
        """提取退保发还金额表（按保单年度）"""
        print(f"\n处理PDF文件: {os.path.basename(pdf_path)} (退保发还金额)")
        surrender_values = {}
        
        try:
            # 查找包含退保发还金额表的页面
            table_pages = self.find_table_pages(pdf_path)
            
            if not table_pages['surrender_value']:
                logger.warning(f"未找到退保发还金额表: {pdf_path}")
                return surrender_values
            
            # 从这些页面提取表格数据
            data = self.extract_table_data_filtered(pdf_path, table_pages['surrender_value'], '退保发还金额')
            
            # 将数据转换为字典格式 {年份: 金额}
            for item in data:
                policy_year = item.get('policy_year')
                total_amount = item.get('total_amount')
                
                if policy_year is not None and total_amount is not None:
                    surrender_values[policy_year] = total_amount
            
            # 显示提取结果
            print(f"提取到 {len(surrender_values)} 条退保发还金额数据")
            
            return surrender_values
            
        except Exception as e:
            logger.error(f"提取退保发还金额表时出错: {e}")
            return surrender_values
    
    def extract_withdrawal_values(self, pdf_path: str) -> Dict[int, float]:
        """提取现金提取后之退保发还金额表（按保单年度）"""
        print(f"\n处理PDF文件: {os.path.basename(pdf_path)} (现金提取方案)")
        withdrawal_values = {}
        
        try:
            # 查找包含现金提取表的页面
            table_pages = self.find_table_pages(pdf_path)
            
            if table_pages['withdrawal_surrender_value']:
                # 提取表格数据
                data = self.extract_table_data_filtered(pdf_path, table_pages['withdrawal_surrender_value'], '现金提取后之退保发还金额')
                
                # 将数据转换为字典格式 {年份: 提取金额}
                for item in data:
                    policy_year = item.get('policy_year')
                    withdrawal_amount = item.get('withdrawal_amount')
                    
                    if policy_year is not None and withdrawal_amount is not None:
                        withdrawal_values[policy_year] = withdrawal_amount
                
                print(f"提取到 {len(withdrawal_values)} 条现金提取金额数据")
                return withdrawal_values
        
            # 尝试特殊处理：有些PDF不在表格中包含提取金额，而是在文本中说明
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # 查找包含年度提取描述的段落
                    lines = text.split('\n')
                    for line in lines:
                        if "现金提取" in line and "%" in line:
                            matches = re.findall(r'(\d+)%.*?(\d+(?:\.\d+)?)', line)
                            if matches:
                                for year_pct, amount in matches:
                                    try:
                                        year = int(year_pct)
                                        value = float(amount.replace(',', ''))
                                        withdrawal_values[year] = value
                                    except:
                                        pass
                                
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
        self.scheme1_pdf_path = os.path.join(self.temp_dir, f"scheme1_{self._safe_filename(self.pdf_urls[0])}")
        self.scheme2_pdf_path = os.path.join(self.temp_dir, f"scheme2_{self._safe_filename(self.pdf_urls[1])}")
        
        # 设置输出文件路径
        self.extracted_data_file = os.path.join(self.temp_dir, "计划书数据提取结果.xlsx")
        self.irr_results_file = os.path.join(self.temp_dir, "IRR计算结果.xlsx")
        self.final_html_file = os.path.join(self.temp_dir, "保险方案对比报告.html")
        self.final_screenshot_file = os.path.join(self.temp_dir, "保险方案对比报告.png")
    
    def _safe_filename(self, url: str) -> str:
        """从URL安全地提取文件名（修复文件名过长问题）"""
        try:
            # 将URL拆分，只保留基本部分（无查询参数）
            url_parts = url.split('?')[0]
            # 从URL路径中提取文件名
            base_name = os.path.basename(url_parts)
            # 如果文件名太长或不是PDF，使用随机名称
            if len(base_name) > 50 or not base_name.endswith('.pdf'):
                return f"{uuid.uuid4().hex}.pdf"
            return base_name
        except Exception as e:
            logger.error(f"处理URL文件名时出错: {url}, 错误: {e}")
            # 安全回退：生成随机文件名
            return f"{uuid.uuid4().hex}.pdf"
    
    def _cleanup(self):
        """清理临时文件"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"临时目录已清理: {self.temp_dir}")
        except Exception as e:
            logger.error(f"清理临时文件时出错: {e}")
    
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
                
                # 如果包含查询参数，去除
                if "?" in storage_path:
                    storage_path = storage_path.split("?")[0]
                
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
        
        if not os.path.exists(local_path):
            logger.error(f"要上传的文件不存在: {local_path}")
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
            return None
    
    async def step1_extract_pdf_data(self) -> pd.DataFrame:
        """步骤1: 下载PDF并提取数据"""
        print("\n=== 步骤1: PDF数据提取 ===")
        
        # 下载PDF文件
        print("下载PDF文件...")
        pdf_download_tasks = [
            self._download_pdf(self.pdf_urls[0], self.scheme1_pdf_path),
            self._download_pdf(self.pdf_urls[1], self.scheme2_pdf_path)
        ]
        
        try:
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
        except Exception as e:
            logger.error(f"PDF数据提取失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def step2_calculate_irr(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤2: 计算IRR"""
        print("\n=== 步骤2: IRR计算 ===")
        
        if df.empty:
            print("错误: 无数据可计算IRR")
            return df
        
        try:
            # 创建结果数据框
            results = []
            
            # 提取客户信息
            s1_annual_premium = df['scheme1_annual_premium'].iloc[0]
            s1_payment_period = df['scheme1_payment_period'].iloc[0]
            s1_surrender_values = df['scheme1_surrender_values'].iloc[0]
            s1_withdrawal_values = df['scheme1_withdrawal_values'].iloc[0]
            
            s2_annual_premium = df['scheme2_annual_premium'].iloc[0]
            s2_payment_period = df['scheme2_payment_period'].iloc[0]
            s2_surrender_values = df['scheme2_surrender_values'].iloc[0]
            s2_withdrawal_values = df['scheme2_withdrawal_values'].iloc[0]
            
            # 按年度计算IRR
            irr_years = sorted(set(list(s1_surrender_values.keys()) + list(s2_surrender_values.keys())))
            
            for policy_year in irr_years:
                if policy_year < max(s1_payment_period, s2_payment_period):
                    continue  # 跳过缴费期内的年份
                
                # 方案1的退保IRR
                s1_surrender_irr = 0
                if policy_year in s1_surrender_values:
                    surrender_value = s1_surrender_values[policy_year]
                    cash_flows = CorrectedIRRCalculator.build_cash_flows_for_surrender(
                        s1_annual_premium, s1_payment_period, policy_year, surrender_value
                    )
                    s1_surrender_irr = CorrectedIRRCalculator.calculate_irr(cash_flows)
                
                # 方案2的退保IRR
                s2_surrender_irr = 0
                if policy_year in s2_surrender_values:
                    surrender_value = s2_surrender_values[policy_year]
                    cash_flows = CorrectedIRRCalculator.build_cash_flows_for_surrender(
                        s2_annual_premium, s2_payment_period, policy_year, surrender_value
                    )
                    s2_surrender_irr = CorrectedIRRCalculator.calculate_irr(cash_flows)
                
                # 方案1的提取方案IRR
                s1_withdrawal_irr = 0
                s1_cumulative_withdrawal = 0
                if s1_withdrawal_values and policy_year in s1_surrender_values:
                    # 计算累计提取金额
                    start_year = min(s1_withdrawal_values.keys())
                    annual_withdrawal = sum(s1_withdrawal_values.values()) / len(s1_withdrawal_values)
                    withdrawal_years = policy_year - start_year
                    s1_cumulative_withdrawal = annual_withdrawal * withdrawal_years if withdrawal_years > 0 else 0
                    
                    # 计算最终退保价值
                    final_surrender_value = s1_surrender_values[policy_year]
                    
                    # 构建现金流并计算IRR
                    cash_flows = CorrectedIRRCalculator.build_cash_flows_for_withdrawal(
                        s1_annual_premium, s1_payment_period, policy_year, 
                        start_year, annual_withdrawal, final_surrender_value
                    )
                    s1_withdrawal_irr = CorrectedIRRCalculator.calculate_irr(cash_flows)
                
                # 方案2的提取方案IRR
                s2_withdrawal_irr = 0
                s2_cumulative_withdrawal = 0
                if s2_withdrawal_values and policy_year in s2_surrender_values:
                    # 计算累计提取金额
                    start_year = min(s2_withdrawal_values.keys())
                    annual_withdrawal = sum(s2_withdrawal_values.values()) / len(s2_withdrawal_values)
                    withdrawal_years = policy_year - start_year
                    s2_cumulative_withdrawal = annual_withdrawal * withdrawal_years if withdrawal_years > 0 else 0
                    
                    # 计算最终退保价值
                    final_surrender_value = s2_surrender_values[policy_year]
                    
                    # 构建现金流并计算IRR
                    cash_flows = CorrectedIRRCalculator.build_cash_flows_for_withdrawal(
                        s2_annual_premium, s2_payment_period, policy_year, 
                        start_year, annual_withdrawal, final_surrender_value
                    )
                    s2_withdrawal_irr = CorrectedIRRCalculator.calculate_irr(cash_flows)
                
                # 添加结果
                results.append({
                    'policy_year': policy_year,
                    'scheme1_surrender_value': s1_surrender_values.get(policy_year, 0),
                    'scheme1_surrender_irr': s1_surrender_irr,
                    'scheme1_cumulative_withdrawal': s1_cumulative_withdrawal,
                    'scheme1_withdrawal_irr': s1_withdrawal_irr,
                    'scheme2_surrender_value': s2_surrender_values.get(policy_year, 0),
                    'scheme2_surrender_irr': s2_surrender_irr,
                    'scheme2_cumulative_withdrawal': s2_cumulative_withdrawal,
                    'scheme2_withdrawal_irr': s2_withdrawal_irr
                })
            
            # 创建结果数据框
            irr_df = pd.DataFrame(results)
            
            # 显示结果
            print("IRR计算结果:")
            print(irr_df[['policy_year', 'scheme1_surrender_irr', 'scheme2_surrender_irr']])
            
            # 保存结果
            with pd.ExcelWriter(self.irr_results_file) as writer:
                df.to_excel(writer, sheet_name='原始数据', index=False)
                irr_df.to_excel(writer, sheet_name='IRR计算结果', index=False)
            
            print(f"IRR计算完成，结果已保存到: {self.irr_results_file}")
            
            return irr_df
        
        except Exception as e:
            logger.error(f"IRR计算失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def calculate_cumulative_withdrawal(self, irr_df: pd.DataFrame) -> pd.DataFrame:
        """计算累计领取金额，更新到IRR结果表"""
        if irr_df.empty:
            return irr_df
        
        # 计算每年累计领取金额（根据withdrawal_irr不为0的第一年开始累计）
        s1_withdrawal_start = False
        s2_withdrawal_start = False
        s1_annual_avg = 0
        s2_annual_avg = 0
        
        # 计算年平均提取金额（从首次不为0的withdrawal_irr开始）
        for idx, row in irr_df.iterrows():
            if not s1_withdrawal_start and row['scheme1_withdrawal_irr'] != 0:
                s1_withdrawal_start = True
                s1_annual_avg = row['scheme1_cumulative_withdrawal'] / (row['policy_year'] - irr_df.loc[0, 'policy_year'])
            
            if not s2_withdrawal_start and row['scheme2_withdrawal_irr'] != 0:
                s2_withdrawal_start = True
                s2_annual_avg = row['scheme2_cumulative_withdrawal'] / (row['policy_year'] - irr_df.loc[0, 'policy_year'])
        
        # 重新计算累计提取金额
        for idx, row in irr_df.iterrows():
            policy_year = row['policy_year']
            start_year = irr_df.loc[0, 'policy_year']
            
            # 方案1
            if s1_withdrawal_start and row['scheme1_withdrawal_irr'] != 0:
                years = policy_year - start_year
                irr_df.at[idx, 'scheme1_cumulative_withdrawal'] = s1_annual_avg * years
            
            # 方案2
            if s2_withdrawal_start and row['scheme2_withdrawal_irr'] != 0:
                years = policy_year - start_year
                irr_df.at[idx, 'scheme2_cumulative_withdrawal'] = s2_annual_avg * years
        
        return irr_df
    
    def step3_generate_html(self, irr_df: pd.DataFrame) -> str:
        """步骤3: 生成HTML可视化报告"""
        print("\n=== 步骤3: HTML报告生成 ===")
        
        if irr_df.empty:
            print("错误: 无数据可生成HTML报告")
            return ""
        
        try:
            # 读取原始数据
            raw_df = pd.read_excel(self.extracted_data_file)
            
            # 提取方案名称和信息
            scheme1_name = raw_df['scheme1_name'].iloc[0]
            scheme1_premium = raw_df['scheme1_annual_premium'].iloc[0]
            scheme1_payment = raw_df['scheme1_payment_period'].iloc[0]
            scheme1_currency = raw_df['scheme1_currency'].iloc[0]
            
            scheme2_name = raw_df['scheme2_name'].iloc[0]
            scheme2_premium = raw_df['scheme2_annual_premium'].iloc[0]
            scheme2_payment = raw_df['scheme2_payment_period'].iloc[0]
            scheme2_currency = raw_df['scheme2_currency'].iloc[0]
            
            # 准备图表数据
            years = irr_df['policy_year'].tolist()
            s1_surrender_values = irr_df['scheme1_surrender_value'].tolist()
            s2_surrender_values = irr_df['scheme2_surrender_value'].tolist()
            s1_surrender_irrs = irr_df['scheme1_surrender_irr'].tolist()
            s2_surrender_irrs = irr_df['scheme2_surrender_irr'].tolist()
            
            # 生成报告HTML
            html_content = f"""
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>保险方案对比分析报告</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body {{
                        font-family: 'Arial', 'Microsoft YaHei', sans-serif;
                        line-height: 1.6;
                        color: #333;
                        margin: 0;
                        padding: 0;
                        background-color: #f8f9fa;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: white;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    }}
                    h1, h2, h3, h4 {{
                        color: #2c3e50;
                    }}
                    h1 {{
                        text-align: center;
                        margin-bottom: 30px;
                        padding-bottom: 15px;
                        border-bottom: 2px solid #e9ecef;
                    }}
                    .summary-box {{
                        background-color: #f8f9fa;
                        border-left: 4px solid #0275d8;
                        padding: 15px;
                        margin-bottom: 20px;
                    }}
                    .chart-container {{
                        position: relative;
                        height: 400px;
                        margin-bottom: 30px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-bottom: 20px;
                    }}
                    th, td {{
                        padding: 12px 15px;
                        text-align: left;
                        border-bottom: 1px solid #e9ecef;
                    }}
                    th {{
                        background-color: #f1f3f5;
                        font-weight: bold;
                    }}
                    tr:hover {{
                        background-color: #f8f9fa;
                    }}
                    .highlight {{
                        background-color: #e9f7ef;
                        font-weight: bold;
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 30px;
                        padding-top: 15px;
                        color: #6c757d;
                        font-size: 0.9em;
                        border-top: 1px solid #e9ecef;
                    }}
                    .scheme1-color {{
                        color: #3498db;
                    }}
                    .scheme2-color {{
                        color: #e74c3c;
                    }}
                    .flex-container {{
                        display: flex;
                        justify-content: space-between;
                        flex-wrap: wrap;
                    }}
                    .info-box {{
                        flex: 1;
                        min-width: 300px;
                        margin: 10px;
                        padding: 15px;
                        background-color: #f8f9fa;
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>保险方案IRR对比分析报告</h1>
                    
                    <div class="summary-box">
                        <h3>方案摘要</h3>
                        <div class="flex-container">
                            <div class="info-box">
                                <h4 class="scheme1-color">方案1: {scheme1_name}</h4>
                                <p>年缴保费: {scheme1_currency} {scheme1_premium:,.0f}</p>
                                <p>缴费期: {scheme1_payment} 年</p>
                                <p>总保费: {scheme1_currency} {scheme1_premium * scheme1_payment:,.0f}</p>
                            </div>
                            <div class="info-box">
                                <h4 class="scheme2-color">方案2: {scheme2_name}</h4>
                                <p>年缴保费: {scheme2_currency} {scheme2_premium:,.0f}</p>
                                <p>缴费期: {scheme2_payment} 年</p>
                                <p>总保费: {scheme2_currency} {scheme2_premium * scheme2_payment:,.0f}</p>
                            </div>
                        </div>
                    </div>
                    
                    <h2>退保价值对比</h2>
                    <div class="chart-container">
                        <canvas id="surrenderValueChart"></canvas>
                    </div>
                    
                    <h2>IRR对比</h2>
                    <div class="chart-container">
                        <canvas id="irrChart"></canvas>
                    </div>
                    
                    <h2>详细数据对比</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>保单年度</th>
                                <th>方案1退保价值({scheme1_currency})</th>
                                <th>方案1 IRR(%)</th>
                                <th>方案2退保价值({scheme2_currency})</th>
                                <th>方案2 IRR(%)</th>
                                <th>差额({scheme1_currency})</th>
                                <th>IRR差异(%)</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            # 添加数据行
            for i, row in irr_df.iterrows():
                year = row['policy_year']
                s1_value = row['scheme1_surrender_value']
                s1_irr = row['scheme1_surrender_irr'] * 100  # 转换为百分比
                s2_value = row['scheme2_surrender_value']
                s2_irr = row['scheme2_surrender_irr'] * 100  # 转换为百分比
                value_diff = s1_value - s2_value
                irr_diff = s1_irr - s2_irr
                
                # 判断哪个方案更好
                row_class = ""
                if s1_irr > s2_irr:
                    row_class = "highlight"
                
                html_content += f"""
                    <tr class="{row_class}">
                        <td>{year}</td>
                        <td>{s1_value:,.2f}</td>
                        <td>{s1_irr:.2f}%</td>
                        <td>{s2_value:,.2f}</td>
                        <td>{s2_irr:.2f}%</td>
                        <td>{value_diff:,.2f}</td>
                        <td>{irr_diff:.2f}%</td>
                    </tr>
                """
            
            # 添加图表脚本和页脚
            html_content += f"""
                        </tbody>
                    </table>
                    
                    <div class="footer">
                        <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p>报告ID: {self.task_id}</p>
                        <p>本报告仅用于保险产品对比参考，不构成任何投资或购买建议。</p>
                    </div>
                </div>
                
                <script>
                    // 退保价值图表
                    const valueCtx = document.getElementById('surrenderValueChart').getContext('2d');
                    const valueChart = new Chart(valueCtx, {{
                        type: 'line',
                        data: {{
                            labels: {years},
                            datasets: [
                                {{
                                    label: '方案1 退保价值',
                                    data: {s1_surrender_values},
                                    backgroundColor: 'rgba(52, 152, 219, 0.2)',
                                    borderColor: 'rgba(52, 152, 219, 1)',
                                    borderWidth: 2,
                                    pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                                    tension: 0.4
                                }},
                                {{
                                    label: '方案2 退保价值',
                                    data: {s2_surrender_values},
                                    backgroundColor: 'rgba(231, 76, 60, 0.2)',
                                    borderColor: 'rgba(231, 76, 60, 1)',
                                    borderWidth: 2,
                                    pointBackgroundColor: 'rgba(231, 76, 60, 1)',
                                    tension: 0.4
                                }}
                            ]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{
                                    position: 'top',
                                }},
                                title: {{
                                    display: true,
                                    text: '保单年度退保价值对比'
                                }}
                            }},
                            scales: {{
                                y: {{
                                    beginAtZero: true,
                                    title: {{
                                        display: true,
                                        text: '退保价值 ({scheme1_currency})'
                                    }}
                                }},
                                x: {{
                                    title: {{
                                        display: true,
                                        text: '保单年度'
                                    }}
                                }}
                            }}
                        }}
                    }});
                    
                    // IRR图表
                    const irrCtx = document.getElementById('irrChart').getContext('2d');
                    const irrChart = new Chart(irrCtx, {{
                        type: 'line',
                        data: {{
                            labels: {years},
                            datasets: [
                                {{
                                    label: '方案1 IRR',
                                    data: {[x * 100 for x in s1_surrender_irrs]},
                                    backgroundColor: 'rgba(52, 152, 219, 0.2)',
                                    borderColor: 'rgba(52, 152, 219, 1)',
                                    borderWidth: 2,
                                    pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                                    tension: 0.4
                                }},
                                {{
                                    label: '方案2 IRR',
                                    data: {[x * 100 for x in s2_surrender_irrs]},
                                    backgroundColor: 'rgba(231, 76, 60, 0.2)',
                                    borderColor: 'rgba(231, 76, 60, 1)',
                                    borderWidth: 2,
                                    pointBackgroundColor: 'rgba(231, 76, 60, 1)',
                                    tension: 0.4
                                }}
                            ]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{
                                    position: 'top',
                                }},
                                title: {{
                                    display: true,
                                    text: '保单年度IRR对比'
                                }}
                            }},
                            scales: {{
                                y: {{
                                    beginAtZero: true,
                                    title: {{
                                        display: true,
                                        text: 'IRR (%)'
                                    }}
                                }},
                                x: {{
                                    title: {{
                                        display: true,
                                        text: '保单年度'
                                    }}
                                }}
                            }}
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            
            # 保存HTML文件
            with open(self.final_html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTML报告生成完成: {self.final_html_file}")
            
            return self.final_html_file
            
        except Exception as e:
            logger.error(f"生成HTML报告失败: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    async def step4_generate_screenshot(self, html_file: str) -> str:
        """步骤4: 生成截图（异步版本，增强的错误处理）"""
        print("\n=== 步骤4: 截图生成 ===")
        
        if not html_file or not os.path.exists(html_file):
            print(f"错误: HTML文件不存在 - {html_file}")
            return ""
        
        # 检查是否已安装Playwright
        if not PLAYWRIGHT_AVAILABLE:
            print("警告: Playwright未安装，无法生成截图")
            return ""
        
        try:
            # 检查Playwright浏览器是否已安装
            try:
                from playwright.async_api import async_playwright
                
                async with async_playwright() as playwright:
                    # 尝试启动浏览器，可能会失败如果二进制文件不存在
                    try:
                        browser = await playwright.chromium.launch(
                            headless=True,
                            args=[
                                '--no-sandbox',
                                '--disable-setuid-sandbox',
                                '--disable-dev-shm-usage',
                                '--disable-accelerated-2d-canvas',
                                '--no-first-run',
                                '--no-zygote',
                                '--single-process',
                                '--disable-gpu'
                            ]
                        )
                        
                        # 浏览器启动成功，继续截图流程
                        context = await browser.new_context(
                            viewport={'width': 1200, 'height': 1600},
                            device_scale_factor=2
                        )
                        page = await context.new_page()
                        
                        # 加载HTML文件
                        file_url = f"file://{os.path.abspath(html_file)}"
                        print(f"加载HTML文件: {file_url}")
                        await page.goto(file_url)
                        
                        # 等待页面完全加载
                        await page.wait_for_load_state("networkidle")
                        await page.wait_for_load_state("domcontentloaded")
                        await asyncio.sleep(3)  # 额外等待确保所有样式加载完成 (异步等待)
                        
                        # 滚动到顶部确保内容可见
                        await page.evaluate("window.scrollTo(0, 0)")
                        await asyncio.sleep(1)  # 异步等待
                        
                        # 尝试定位主要内容容器，使用多种选择器
                        container_selectors = [
                            '.container',  # 常见的容器类名
                            '.content',    # 内容容器
                            'main',        # HTML5语义标签
                            'body > div:first-child',  # body下第一个div
                            'body > div',  # body下的div
                            'body'         # 最后回退到body
                        ]
                        
                        container_found = False
                        for selector in container_selectors:
                            try:
                                # 检查元素是否存在且可见
                                element = page.locator(selector).first
                                count = await element.count()
                                if count > 0:
                                    # 检查元素是否有实际内容
                                    bounding_box = await element.bounding_box()
                                    if bounding_box and bounding_box['width'] > 100 and bounding_box['height'] > 100:
                                        print(f"找到主要容器: {selector}")
                                        print(f"容器尺寸: {bounding_box}")
                                        
                                        # 直接截取容器元素，不需要手动裁剪
                                        print(f"正在生成超高分辨率截图...")
                                        await element.screenshot(
                                            path=self.final_screenshot_file,
                                            type='png'
                                        )
                                        
                                        container_found = True
                                        break
                            except Exception as e:
                                print(f"尝试选择器 '{selector}' 时出错: {e}")
                                continue
                        
                        # 如果没有找到合适的容器，回退到全页面截图
                        if not container_found:
                            print("未找到合适的容器，使用全页面截图...")
                            await page.screenshot(
                                path=self.final_screenshot_file,
                                full_page=True,
                                type='png'
                            )
                        
                        await browser.close()
                        
                        print(f"超高质量截图生成完成: {self.final_screenshot_file}")
                        
                        # 验证截图文件
                        if os.path.exists(self.final_screenshot_file):
                            file_size = os.path.getsize(self.final_screenshot_file)
                            print(f"截图文件大小: {file_size:,} 字节")
                            return self.final_screenshot_file
                        else:
                            print("错误: 截图文件未生成")
                            return ""
                    
                    except Exception as browser_error:
                        # 浏览器启动失败
                        print(f"生成截图时出错: {browser_error}")
                        import traceback
                        traceback.print_exc()
                        print("警告: 截图生成失败，但其他步骤已完成")
                        return ""
            
            except ImportError as import_error:
                # Playwright导入失败
                print(f"无法导入Playwright模块: {import_error}")
                print("警告: 截图生成失败，但其他步骤已完成")
                return ""
            
        except Exception as e:
            # 其他未预期的错误
            print(f"生成截图时出错: {e}")
            import traceback
            traceback.print_exc()
            print("警告: 截图生成失败，但其他步骤已完成")
            return ""
    
    async def run_complete_pipeline(self) -> Dict[str, str]:
        """运行完整的端到端流水线，返回结果文件URL"""
        print("开始执行Railway版本保险数据处理流水线")
        print("=" * 60)
        
        start_time = datetime.now()
        result_urls = {}
        
        try:
            # 步骤1: 提取PDF数据
            extracted_df = await self.step1_extract_pdf_data()
            if extracted_df.empty:
                raise Exception("PDF数据提取失败")
            
            # 步骤2: 计算IRR
            irr_df = self.step2_calculate_irr(extracted_df)
            
            # 计算累计领取金额
            irr_df = self.calculate_cumulative_withdrawal(irr_df)
            
            # 步骤3: 生成HTML报告
            html_file = self.step3_generate_html(irr_df)
            if not html_file:
                raise Exception("HTML报告生成失败")
            
            # 步骤4: 生成截图 - 即使失败也继续流程
            try:
                screenshot_file = await self.step4_generate_screenshot(html_file)
            except Exception as screenshot_error:
                logger.error(f"截图生成失败，但继续执行其他步骤: {screenshot_error}")
                screenshot_file = ""
                print("警告: 截图生成失败，但其他步骤将继续")
            
            # 步骤5: 上传结果到Supabase
            if self.supabase:
                print("\n=== 步骤5: 上传结果 ===")
                
                # 上传Excel数据
                excel_url = await self._upload_file_to_supabase(
                    self.irr_results_file,
                    f"{self.task_id}/data.xlsx"
                )
                if excel_url:
                    result_urls['excel_url'] = excel_url
                
                # 上传HTML报告
                html_url = await self._upload_file_to_supabase(
                    html_file,
                    f"{self.task_id}/report.html"
                )
                if html_url:
                    result_urls['html_url'] = html_url
                
                # 上传截图（如果成功生成）
                if screenshot_file and os.path.exists(screenshot_file):
                    screenshot_url = await self._upload_file_to_supabase(
                        screenshot_file,
                        f"{self.task_id}/screenshot.png"
                    )
                    if screenshot_url:
                        result_urls['screenshot_url'] = screenshot_url
                
                # 更新任务状态
                try:
                    self.supabase.table('tasks').update({
                        'status': 'completed',
                        'results': result_urls,
                        'completed_at': datetime.now().isoformat()
                    }).eq('id', self.task_id).execute()
                    print(f"任务结果已更新到数据库: {self.task_id}")
                except Exception as db_err:
                    logger.error(f"更新任务状态时出错: {db_err}")
            
            # 清理临时文件
            self._cleanup()
            
            # 计算处理时间
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"\n流水线完成! 处理时间: {duration:.2f} 秒")
            
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
                        'error_message': str(e),  # 修复: 使用 error_message 而不是 error
                        'completed_at': datetime.now().isoformat()
                    }).eq('id', self.task_id).execute()
                    print(f"任务失败状态已更新到数据库: {self.task_id}")
                except Exception as db_err:
                    logger.error(f"更新任务失败状态时出错: {db_err}")
            
            # 清理临时文件
            self._cleanup()
            
            # 返回错误信息
            return {"error": str(e)}

async def process_insurance_pdfs(pdf_urls, supabase_url=None, supabase_key=None, task_id=None):
    """处理保险PDF文件的异步函数"""
    try:
        # 创建并运行流水线
        pipeline = RailwayInsurancePipeline(
            pdf_urls=pdf_urls,
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            task_id=task_id
        )
        
        # 运行完整流水线
        results = await pipeline.run_complete_pipeline()
        
        # 返回结果
        return results
    
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# 如果直接运行脚本，提供简单的测试功能
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Railway保险数据处理流水线')
    parser.add_argument('--pdf1', required=True, help='第一个保险方案PDF的URL')
    parser.add_argument('--pdf2', required=True, help='第二个保险方案PDF的URL')
    parser.add_argument('--supabase_url', help='Supabase URL')
    parser.add_argument('--supabase_key', help='Supabase Key')
    parser.add_argument('--task_id', help='任务ID')
    
    args = parser.parse_args()
    
    print("启动Railway版本保险数据处理流水线")
    print(f"   方案1: {args.pdf1}")
    print(f"   方案2: {args.pdf2}")
    
    result = asyncio.run(process_insurance_pdfs(
        pdf_urls=[args.pdf1, args.pdf2],
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
        task_id=args.task_id
    ))
    
    print("\n处理结果:")
    print(result)