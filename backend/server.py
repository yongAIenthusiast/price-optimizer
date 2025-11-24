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

        # ✅ 核心修复：强制指定 pipeline 任务为 'feature-extraction'
        # 使用 router 域名，但保留 pipeline 路径结构，防止 API 自动错误识别为 SentenceSimilarity
        self.hf_api_url = "https://router.huggingface.co/hf-inference/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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
        # 组合标题、五点描述和长描述
        return f"{p.get('title', '')}. " + " ".join(p.get('feature_bullets', [])) + str(p.get('description', ''))

    def get_embeddings_from_hf(self, texts):
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}  # 如果模型在休眠，强制唤醒
        }
        try:
            print(f"🧠 Calling HuggingFace (Feature Extraction) for {len(texts)} texts...")
            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=30)
            return response.json()
        except Exception as e:
            print(f"❌ HuggingFace Error: {e}")
            return None

    def search_and_match(self, my_desc, keyword):
        # 1. 搜索
        params = {'type': 'search', 'search_term': keyword, 'sort_by': 'featured'}
        data = self._make_rainforest_request(params)

        candidates = []
        if data and 'search_results' in data:
            # 限制前 3 个结果
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

        # 2. 获取详情
        all_texts = [my_desc]
        valid_candidates = []

        print("⏳ Fetching details for candidates...")
        for item in candidates:
            dt = self.get_product_details(item['id'])
            if dt:
                # 截断文本防止超出 API 限制
                item['desc_text'] = dt
                all_texts.append(dt[:1000])
                valid_candidates.append(item)

        if not valid_candidates:
            return None, []

        # 3. 计算向量
        embeddings = self.get_embeddings_from_hf(all_texts)

        # 错误处理
        if not embeddings or isinstance(embeddings, dict):
            # 如果 API 返回错误字典
            print(f"Embeddings failed: {embeddings}")
            # 兜底策略：如果向量计算失败，使用简单的长度/随机分作为模拟，防止前端崩溃
            # 这在生产环境应该报错，但在演示中可以保证流程跑通
            if isinstance(embeddings, dict) and 'error' in embeddings:
                print("⚠️ Falling back to dummy similarity due to AI error")
                # 返回第一个结果作为匹配，并标记
                best = valid_candidates[0]
                best['similarity'] = 0.0
                best['features'] = "AI Error: " + str(embeddings['error'])[:50]
                return best, valid_candidates
            return None, []

        if len(embeddings) != len(all_texts):
            print("Embeddings length mismatch")
            return None, []

        # 4. 计算相似度
        # 注意：HF API 有时返回的是 [ [dim], [dim] ]，有时是嵌套的，加个检查
        try:
            my_vector = np.array(embeddings[0])
            if my_vector.ndim > 1: my_vector = my_vector[0]  # 扁平化处理
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
            print(f"Math Error: {e}")
            return None, []


# --- 路由配置 ---

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "message": "Backend is running with Forced Feature Extraction"}), 200


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
        response = jsonify({"error": "Server Config Error: Missing Env Vars"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

    try:
        matcher = AmazonCompetitorMatcher(r_key, h_token)
        best, all_results = matcher.search_and_match(description, keyword)
        return jsonify({"success": True, "best_match": best, "all_candidates": all_results})
    except Exception as e:
        print(f"❌ Server Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)