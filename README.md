# 保险方案分析服务 - Railway部署版本

此文件夹包含了适用于Railway云平台部署的保险方案分析服务代码。

## 文件说明

- `railway_insurance_pipeline.py` - 核心处理逻辑，包含PDF解析、IRR计算、HTML生成等功能
- `app.py` - Flask Web API接口，提供HTTP服务
- `requirements.txt` - 项目依赖列表
- `Procfile` - Railway部署配置文件
- `两套方案对比HTML模板_含占位符_修正7.html` - HTML报告模板

## 部署说明

1. 将这些文件推送到GitHub仓库
2. 在Railway平台上创建新项目
3. 选择从GitHub导入
4. 选择包含这些文件的仓库

## API接口说明

### 健康检查
- 端点: `GET /health`
- 说明: 返回服务健康状态

### 处理保险PDF
- 端点: `POST /process`
- 请求体:
  ```json
  {
    "supabase_url": "https://your-supabase-project.supabase.co",
    "supabase_key": "your-supabase-anon-key",
    "pdf_urls": [
      "https://your-supabase-project.supabase.co/storage/v1/object/uploads/scheme1.pdf",
      "https://your-supabase-project.supabase.co/storage/v1/object/uploads/scheme2.pdf"
    ],
    "task_id": "unique-task-id"
  }
  ```
- 说明: 处理两个保险PDF文件，生成对比报告

## 主要修复

1. **Playwright异步API问题**：
   - 将`step4_generate_screenshot`函数完全转换为异步模式
   - 使用`playwright.async_api`替代`sync_playwright`
   - 解决在异步环境中使用同步API的冲突

2. **MIME类型错误**：
   - 增强`upload_file_to_supabase`方法
   - 添加全面的MIME类型检测
   - 提供更详细的日志输出
   
3. **IRR计算逻辑修复**：
   - 添加正确的`CorrectedIRRCalculator`类
   - 完整保留原始脚本的计算逻辑
   - 正确计算累计提取金额和剩余价值

## 注意事项

- 此服务需要Supabase进行存储和数据库操作
- 需要确保Supabase中存在`results`和`uploads`存储桶
- Playwright需要较大的内存空间运行，请确保Railway配置有足够资源