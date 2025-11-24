import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 强力 CORS 配置
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


class AmazonCompetitorMatcher:
    def __init__(self, rainforest_api_key, hf_token):
        self.rainforest_api_key = rainforest_api_key
        self.hf_token = hf_token
        self.rainforest_url = "https://api.rainforestapi.com/request"

        self.model_id = "sentence-transformers/distiluse-base-multilingual-cased-v1"

        # ✅ 终极修复：
        # 1. 使用 router.huggingface.co 新域名 (解决 410 错误)
        # 2. 显式指定 /pipeline/feature-extraction/ 路径 (解决 'sentences' 参数缺失错误)
        self.hf_api_url = f"https://router.huggingface.co/hf-inference/pipeline/feature-extraction/{self.model_id}"

    def _make_rainforest_request(self, params):
        params['api_key'] = self.rainforest_api_key
        if 'amazon_domain' not in params:
            params['amazon_domain'] = 'amazon.de'
        try:
            print(f"📡 Calling Rainforest: {params.get('type')}")
            response = requests.get(self.rainforest_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Rainforest Error: {e}")
            return None

    def get_product_details(self, asin):
        params = {'type': 'product', 'asin': asin}
        data = self._make_rainforest_request(params)
        if not data or 'product' not in data:
            return ""
        p = data['product']
        return f"{p.get('title', '')}. " + " ".join(p.get('feature_bullets', [])) + str(p.get('description', ''))

    def get_embeddings_from_hf(self, texts):
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }
        try:
            print(f"🧠 Calling HuggingFace Router (Feature Extraction) for {len(texts)} texts...")
            # 打印 URL 以便调试，确保它是 feature-extraction
            print(f"   Endpoint: {self.hf_api_url}")

            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=30)

            if response.status_code != 200:
                print(f"❌ HF API Error {response.status_code}: {response.text}")
                return None

            return response.json()

        except Exception as e:
            print(f"❌ HuggingFace Network Error: {e}")
            return None

    def search_and_match(self, my_desc, keyword):
        # 1. 搜索
        params = {'type': 'search', 'search_term': keyword, 'sort_by': 'featured'}
        data = self._make_rainforest_request(params)

        candidates = []
        if data and 'search_results' in data:
            # 限制前 3 个
            for item in data['search_results'][:3]:
                candidates.append({
                    'id': item.get('asin'),
                    'title': item.get('title'),
                    'price': item.get('price', {}).get('value', 0.0),
                    'currency': item.get('price', {}).get('currency', 'EUR'),
                    'link': item.get('link'),
                    'sales': item.get('ratings_total', 0),
                    'desc_text': ''
                })

        if not candidates:
            return None, []

        # 2. 详情
        all_texts = [my_desc]
        valid_candidates = []

        print("⏳ Fetching details...")
        for item in candidates:
            dt = self.get_product_details(item['id'])
            if dt:
                item['desc_text'] = dt
                all_texts.append(dt[:800])
                valid_candidates.append(item)

        if not valid_candidates: return None, []

        # 3. 向量
        embeddings = self.get_embeddings_from_hf(all_texts)

        if not embeddings or isinstance(embeddings, dict):
            print(f"Embeddings failed. Response: {embeddings}")
            # 兜底模拟数据，防止前端白屏
            if valid_candidates:
                print("⚠️ Using fallback simulation data due to AI error")
                best = valid_candidates[0]
                best['similarity'] = 0.0
                best['features'] = "AI Service Unavailable"
                return best, valid_candidates
            return None, []

        if not isinstance(embeddings, list):
            print(f"Unexpected format: {type(embeddings)}")
            return None, []

        # 4. 计算
        try:
            # 处理 Hugging Face 返回的维度问题 (有时是 [N, 384], 有时是 [1, N, 384])
            embeddings_arr = np.array(embeddings)
            if embeddings_arr.ndim == 3:
                embeddings_arr = embeddings_arr[0]  # 降维

            my_vector = embeddings_arr[0].reshape(1, -1)

            best_match = None
            highest_score = -1

            for i, item in enumerate(valid_candidates):
                # i+1 因为第0个是我的文本
                item_vector = embeddings_arr[i + 1].reshape(1, -1)

                score = float(cosine_similarity(my_vector, item_vector)[0][0])
                item['similarity'] = score
                item['features'] = item['desc_text'][:100] + "..."

                if score > highest_score:
                    highest_score = score
                    best_match = item

            return best_match, valid_candidates
        except Exception as e:
            print(f"Math Error: {e}")
            return None, []


# --- Route ---

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "model": "DistilUSE (Forced Feature Extraction)"}), 200


@app.route('/api/find-competitor', methods=['POST', 'OPTIONS'])
def find_competitor():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    data = request.json
    keyword = data.get('keyword', '')
    description = data.get('description', '')

    r_key = os.environ.get("RAINFOREST_API_KEY")
    h_token = os.environ.get("HF_TOKEN")

    if not r_key or not h_token:
        print("❌ 错误: 环境变量缺失")
        response = jsonify({"error": "Missing Env Vars"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

    try:
        matcher = AmazonCompetitorMatcher(r_key, h_token)
        best, all_results = matcher.search_and_match(description, keyword)
        return jsonify({"success": True, "best_match": best, "all_candidates": all_results})
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)