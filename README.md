# 保险方案分析系统 - Railway部署说明

## 准备工作

1. 在Supabase上已完成以下资源创建：
   - `uploads`存储桶：用于存储用户上传的PDF文件
   - `results`存储桶：用于存储生成的报告
   - `tasks`数据表：用于跟踪任务状态
   - `process-insurance` Edge Function：处理保险方案分析任务

2. Supabase项目信息：
   - 项目ID：qbbrkryynqnzxyagubln
   - 项目URL：https://qbbrkryynqnzxyagubln.supabase.co
   - Edge Function URL：https://qbbrkryynqnzxyagubln.supabase.co/functions/v1/process-insurance

## Railway部署步骤

1. 创建Railway账户并安装CLI
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. 创建新项目
   ```bash
   railway init
   ```

3. 部署应用
   ```bash
   cd /workspace/deploy
   railway up
   ```

4. 获取应用URL
   ```bash
   railway status
   ```

5. 更新Supabase Edge Function中的Railway URL
   - 编辑Edge Function代码
   - 将`RAILWAY_API_URL`变量更新为Railway生成的URL，例如：
     ```javascript
     const RAILWAY_API_URL = 'https://your-railway-app-url.railway.app/process';
     ```
   - 重新部署Edge Function

## 测试部署

1. 上传测试PDF文件到Supabase `uploads`存储桶

2. 创建测试任务记录
   ```sql
   INSERT INTO tasks (user_id, scheme1_file_id, scheme2_file_id, status)
   VALUES ('YOUR_USER_ID', 'uploads/test-file-1.pdf', 'uploads/test-file-2.pdf', 'pending')
   RETURNING id;
   ```

3. 调用Edge Function处理任务
   ```bash
   curl -X POST 'https://qbbrkryynqnzxyagubln.supabase.co/functions/v1/process-insurance' \
     -H 'Authorization: Bearer YOUR_ANON_KEY' \
     -H 'Content-Type: application/json' \
     -d '{
       "scheme1_file_id": "uploads/test-file-1.pdf",
       "scheme2_file_id": "uploads/test-file-2.pdf",
       "user_id": "YOUR_USER_ID",
       "task_id": "TASK_ID_FROM_PREVIOUS_STEP"
     }'
   ```

4. 检查任务状态
   ```sql
   SELECT * FROM tasks WHERE id = 'TASK_ID';
   ```

5. 查看生成的报告
   - 打开`result_urls.html_report`中的URL