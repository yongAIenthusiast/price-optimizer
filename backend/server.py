from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
import time
from sklearn.metrics.pairwise import cosine_similarity

# 初始化 Flask 应用
app = Flask(__name__)
# 允许跨域请求 (CORS)，这样你的 React 前端 (通常在端口 3000) 才能访问这个 Python 后端 (端口 5000)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- 核心 AI 类 (复用之前的逻辑) ---

try:
    from sentence_transformers import SentenceTransformer

    print("正在加载 AI 模型 (paraphrase-multilingual-MiniLM-L12-v2)...这可能需要一点时间")
    # 加载模型到内存中 (全局变量)，这样不用每次请求都重新加载
    GLOBAL_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✅ AI 模型加载完成！")
except ImportError:
    print("⚠️ 警告: 未安装 sentence-transformers。将使用随机向量模式。")
    GLOBAL_MODEL = None


class AmazonCompetitorMatcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.rainforestapi.com/request"
        self.model = GLOBAL_MODEL

    def get_embedding(self, text):
        if self.model and text:
            return self.model.encode(text)
        else:
            return np.random.rand(384)

    def _make_api_request(self, params):
        params['api_key'] = self.api_key
        if 'amazon_domain' not in params:
            params['amazon_domain'] = 'amazon.de'
        try:
            print(f"📡 发送 API 请求: {params.get('type')}...")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ API 请求失败: {e}")
            return None

    def get_product_details(self, asin):
        # 获取详细五点描述
        params = {'type': 'product', 'asin': asin}
        data = self._make_api_request(params)
        if not data or 'product' not in data:
            return ""
        p = data['product']
        text = f"{p.get('title', '')}. " + " ".join(p.get('feature_bullets', []))
        if not p.get('feature_bullets'):
            text += p.get('description', '')
        return text

    def search_and_match(self, my_desc, keyword):
        # 1. 搜索
        params = {'type': 'search', 'search_term': keyword, 'sort_by': 'featured'}
        data = self._make_api_request(params)

        candidates = []
        if data and 'search_results' in data:
            # 限制前 3 个以节省积分和时间
            for item in data['search_results'][:1]:
                candidates.append({
                    'id': item.get('asin'),
                    'title': item.get('title'),
                    'price': item.get('price', {}).get('value', 0.0),
                    'currency': item.get('price', {}).get('currency', 'EUR'),
                    'link': item.get('link'),
                    'sales': item.get('ratings_total', 0),
                    'image': item.get('image')
                })

        if not candidates:
            return None, []

        # 2. AI 比对
        my_vector = self.get_embedding(my_desc)
        best_match = None
        highest_score = -1
        results = []

        for item in candidates:
            # 获取详情 (真实场景下会消耗积分)
            # 为了演示速度和省钱，如果没有详情API权限，这里可以暂时只用标题
            # detailed_text = self.get_product_details(item['id'])
            # 暂时降级为使用标题，以确保快速响应
            detailed_text = item['title']

            item_vector = self.get_embedding(detailed_text)
            score = float(cosine_similarity(my_vector.reshape(1, -1), item_vector.reshape(1, -1))[0][0])

            item['similarity'] = score
            # 提取特性关键词 (简单的模拟)
            item['features'] = detailed_text[:50] + "..."

            results.append(item)
            if score > highest_score:
                highest_score = score
                best_match = item

        return best_match, results


# --- API 路由定义 ---

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "message": "OptiPrice AI Backend is active"}), 200


@app.route('/api/find-competitor', methods=['POST'])
def find_competitor():
    """
    前端发送 JSON: { "keyword": "Bodenstuhl", "description": "..." }
    """
    data = request.json
    keyword = data.get('keyword', '')
    description = data.get('description', '')

    # 🔴 请替换为你的真实 Key
    API_KEY = "BF906805A6BA464EB9F10AE1819CE777"

    if not API_KEY or "YOUR_API_KEY" in API_KEY:
        return jsonify({"error": "Server configuration error: API Key missing"}), 500

    matcher = AmazonCompetitorMatcher(API_KEY)

    try:
        best, all_results = matcher.search_and_match(description, keyword)
        return jsonify({
            "success": True,
            "best_match": best,
            "all_candidates": all_results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 启动 Flask 服务器 on http://localhost:5000")
    app.run(debug=True, port=5000)