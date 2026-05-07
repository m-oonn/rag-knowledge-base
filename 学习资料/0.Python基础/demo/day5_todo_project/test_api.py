"""
Day 5 附加：API 自动测试脚本

运行方式：
  1. 先启动服务器: uvicorn main:app --reload
  2. 新开一个终端运行: python test_api.py

这个脚本用 httpx 自动测试所有接口，
帮你验证每个功能是否正常工作。

学习要点：
- 怎么用代码调用自己写的 API（后面调用大模型 API 也是同样的方式）
- 怎么检查返回结果是否正确
"""

import httpx
import json

BASE_URL = "http://127.0.0.1:8000"


def print_response(label: str, response):
    """格式化打印响应"""
    status = response.status_code
    icon = "✅" if 200 <= status < 300 else "❌"
    print(f"\n{icon} {label}")
    print(f"   状态码: {status}")
    if response.text:
        try:
            data = response.json()
            print(f"   响应: {json.dumps(data, ensure_ascii=False, indent=2)[:300]}")
        except json.JSONDecodeError:
            print(f"   响应: {response.text[:200]}")
    print()


def main():
    print("=" * 60)
    print("TodoList API 自动测试")
    print("=" * 60)

    with httpx.Client(base_url=BASE_URL) as client:

        # --- 1. 健康检查 ---
        resp = client.get("/health")
        print_response("1. 健康检查 GET /health", resp)

        # --- 2. 获取全部待办 ---
        resp = client.get("/todos")
        print_response("2. 获取全部 GET /todos", resp)

        # --- 3. 创建新待办 ---
        resp = client.post("/todos", json={
            "title": "学习LangChain",
            "description": "RAG项目的核心框架",
            "priority": "high",
        })
        print_response("3. 创建待办 POST /todos", resp)
        new_id = resp.json().get("id") if resp.status_code == 201 else None

        # --- 4. 获取单个 ---
        if new_id:
            resp = client.get(f"/todos/{new_id}")
            print_response(f"4. 获取单个 GET /todos/{new_id}", resp)

        # --- 5. 更新状态 ---
        if new_id:
            resp = client.patch(f"/todos/{new_id}", json={
                "status": "in_progress",
            })
            print_response(f"5. 更新状态 PATCH /todos/{new_id}", resp)

        # --- 6. 搜索 ---
        resp = client.get("/todos", params={"q": "Python"})
        print_response("6. 搜索 GET /todos?q=Python", resp)

        # --- 7. 按状态过滤 ---
        resp = client.get("/todos", params={"status": "completed"})
        print_response("7. 过滤 GET /todos?status=completed", resp)

        # --- 8. 统计信息 ---
        resp = client.get("/todos/stats")
        print_response("8. 统计 GET /todos/stats", resp)

        # --- 9. 删除 ---
        if new_id:
            resp = client.delete(f"/todos/{new_id}")
            print_response(f"9. 删除 DELETE /todos/{new_id}", resp)

        # --- 10. 确认删除成功 ---
        if new_id:
            resp = client.get(f"/todos/{new_id}")
            print_response(f"10. 确认已删除 GET /todos/{new_id} (应返回404)", resp)

        # --- 11. 错误测试：创建无效数据 ---
        resp = client.post("/todos", json={
            "title": "",  # 空标题，应该验证失败
            "priority": "invalid",  # 无效优先级
        })
        print_response("11. 错误测试: 空标题+无效优先级 (应返回422)", resp)

        # --- 12. 错误测试：不存在的 ID ---
        resp = client.get("/todos/99999")
        print_response("12. 错误测试: 不存在的ID (应返回404)", resp)

    print("=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
