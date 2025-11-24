import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 允许跨域
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


class AmazonCompetitorMatcher:
    def __init__(self, rainforest_api_key, hf_token):
        self.rainforest_api_key = rainforest_api_key
        self.hf_token = hf_token
        self.rainforest_url = "https://api.rainforestapi.com/request"

        # ✅ 修改 1: 切换到一个更稳定的多语言模型 (DistilUSE)
        # 这个模型通常不会强制进入 SentenceSimilarity 模式，更容易获取向量
        self.model_id = "sentence-transformers/distiluse-base-multilingual-cased-v1"

        # ✅ 修改 2: 使用标准 API URL (如果这个报错 router，我们再换，但通常 models 路径是通用的)
        self.hf_api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"

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
            print(f"🧠 Calling HuggingFace ({self.model_id}) for {len(texts)} texts...")
            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=30)

            # ✅ 核心调试：如果状态码不是 200，打印原始内容
            if response.status_code != 200:
                print(f"❌ HF API Error {response.status_code}: {response.text}")

                # 如果提示要用 router，我们在这里做个自动回退 (Failover)
                if "router.huggingface.co" in response.text:
                    print("⚠️ API requests redirect to Router. Retrying with Router URL...")
                    router_url = f"https://router.huggingface.co/hf-inference/models/{self.model_id}"
                    response = requests.post(router_url, headers=headers, json=payload, timeout=30)

            # 尝试解析 JSON
            try:
                result = response.json()
            except Exception:
                print(f"❌ Critical: Response is not JSON. Raw body: {response.text[:200]}...")
                return None

            return result

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
            for item in data['search_results'][:1]:
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
                all_texts.append(dt[:800])  # 稍微减少长度以防超限
                valid_candidates.append(item)

        if not valid_candidates: return None, []

        # 3. 向量
        embeddings = self.get_embeddings_from_hf(all_texts)

        # 错误处理
        if not embeddings or isinstance(embeddings, dict):
            print(f"Embeddings failed. Response: {embeddings}")
            # 兜底：如果 AI 挂了，返回模拟数据防止前端报错
            fallback_match = valid_candidates[0]
            fallback_match['similarity'] = 0.0
            fallback_match['features'] = "AI Service Error - Check Logs"
            return fallback_match, valid_candidates

        # 维度检查 (确保返回的是向量列表)
        if not isinstance(embeddings, list):
            print(f"Unexpected format: {type(embeddings)}")
            return None, []

        # 4. 计算
        try:
            my_vector = np.array(embeddings[0])
            # 处理嵌套情况 [[...]]
            if my_vector.ndim > 1: my_vector = my_vector[0]
            my_vector = my_vector.reshape(1, -1)

            best_match = None
            highest_score = -1

            for i, item in enumerate(valid_candidates):
                item_vector = np.array(embeddings[i + 1])
                if item_vector.ndim > 1: item_vector = item_vector[0]
                item_vector = item_vector.reshape(1, -1)

                score = float(cosine_similarity(my_vector, item_vector)[0][0])
                item['similarity'] = score
                item['features'] = item['desc_text'][:100] + "..."

                if score > highest_score:
                    highest_score = score
                    best_match = item

            return best_match, valid_candidates
        except Exception as e:
            print(f"Math Error during similarity calc: {e}")
            return None, []


# --- Route ---

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "model": "DistilUSE Multilingual"}), 200


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