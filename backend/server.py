from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


class AmazonCompetitorMatcher:
    def __init__(self, rainforest_api_key, hf_token):
        self.rainforest_api_key = rainforest_api_key
        self.hf_token = hf_token
        self.rainforest_url = "https://api.rainforestapi.com/request"
        # 使用 HF 云端模型，避免 Render 内存溢出
        self.hf_api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def _make_rainforest_request(self, params):
        params['api_key'] = self.rainforest_api_key
        if 'amazon_domain' not in params:
            params['amazon_domain'] = 'amazon.de'
        try:
            # 打印请求类型，方便调试
            print(f"📡 Rainforest API 请求: {params.get('type')} (ASIN: {params.get('asin', 'N/A')})")
            response = requests.get(self.rainforest_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Rainforest API 错误: {e}")
            return None

    def get_product_details(self, asin):
        """
        【核心修改】获取产品的详细五点描述和长描述
        注意：每次调用消耗 1 个 Rainforest 积分
        """
        params = {'type': 'product', 'asin': asin}
        data = self._make_rainforest_request(params)

        if not data or 'product' not in data:
            return ""

        p = data['product']

        # 提取标题
        title = p.get('title', '')

        # 提取五点描述 (这是最重要的比对信息)
        bullets = p.get('feature_bullets', [])
        bullets_text = " ".join(bullets) if bullets else ""

        # 提取长描述 (作为补充)
        description = p.get('description', '')

        # 组合成一个完整的语义文本
        full_text = f"{title}. {bullets_text} {description}"

        # 如果抓取到的信息太少，就只返回标题
        if len(full_text) < 20:
            return title

        return full_text

    def get_embeddings_from_hf(self, texts):
        """调用 Hugging Face 云端 API 计算向量"""
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }
        try:
            print(f"🧠 调用 Hugging Face AI 计算 {len(texts)} 个文本的向量...")
            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=30)
            return response.json()
        except Exception as e:
            print(f"❌ Hugging Face API 错误: {e}")
            return None

    def search_and_match(self, my_desc, keyword):
        # 1. 搜索竞品列表
        params = {'type': 'search', 'search_term': keyword, 'sort_by': 'featured'}
        data = self._make_rainforest_request(params)

        candidates = []
        if data and 'search_results' in data:
            # ⚠️ 限制为前 3 个结果以平衡成本 (每次运行消耗约 4 积分)
            # 如果你想省钱，可以改成 [:1]
            for item in data['search_results'][:3]:
                candidates.append({
                    'id': item.get('asin'),
                    'title': item.get('title'),
                    'price': item.get('price', {}).get('value', 0.0),
                    'currency': item.get('price', {}).get('currency', 'EUR'),
                    'link': item.get('link'),
                    'sales': item.get('ratings_total', 0),
                    'image': item.get('image'),
                    # 这里的 desc_text 暂时留空，下面会通过 API 填充详细版
                    'desc_text': ''
                })

        if not candidates:
            return None, []

        # 2. 【核心修改】遍历获取每个竞品的"详细描述"
        print("⏳ 正在深入抓取竞品详情 (这需要几秒钟)...")
        valid_candidates = []

        # 准备文本列表，第一个是"我的产品"
        all_texts = [my_desc]

        for item in candidates:
            # 调用详情 API
            detail_text = self.get_product_details(item['id'])

            if detail_text:
                item['desc_text'] = detail_text
                # 截取前 500 个字符用于 AI 分析 (太长可能会超过 API 限制，且 500 字足够判断语义)
                all_texts.append(detail_text[:1000])
                valid_candidates.append(item)

        if not valid_candidates:
            return None, []

        # 3. 云端计算向量
        embeddings = self.get_embeddings_from_hf(all_texts)

        # 错误处理
        if isinstance(embeddings, dict) and 'error' in embeddings:
            print(f"HF Error: {embeddings}")
            return None, []

        if not embeddings or len(embeddings) != len(all_texts):
            return None, []

        # 4. 计算相似度
        my_vector = np.array(embeddings[0]).reshape(1, -1)
        best_match = None
        highest_score = -1

        for i, item in enumerate(valid_candidates):
            # i+1 因为 all_texts[0] 是我的产品
            item_vector = np.array(embeddings[i + 1]).reshape(1, -1)
            score = float(cosine_similarity(my_vector, item_vector)[0][0])

            item['similarity'] = score
            # 在前端展示匹配到的关键特征 (取详细描述的前 100 个字)
            item['features'] = item['desc_text'][:100] + "..."

            if score > highest_score:
                highest_score = score
                best_match = item

        return best_match, valid_candidates


# --- API 路由 ---

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "provider": "Hugging Face + Rainforest Detailed"}), 200


@app.route('/api/find-competitor', methods=['POST'])
def find_competitor():
    data = request.json
    keyword = data.get('keyword', '')
    description = data.get('description', '')

    RAINFOREST_KEY = os.environ.get("RAINFOREST_API_KEY")
    HF_TOKEN = os.environ.get("HF_TOKEN")

    if not RAINFOREST_KEY or not HF_TOKEN:
        print("❌ 错误: 环境变量未配置")
        return jsonify({"error": "Server configuration error: Environment Variables missing"}), 500

    matcher = AmazonCompetitorMatcher(RAINFOREST_KEY, HF_TOKEN)

    try:
        best, all_results = matcher.search_and_match(description, keyword)
        return jsonify({
            "success": True,
            "best_match": best,
            "all_candidates": all_results
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)