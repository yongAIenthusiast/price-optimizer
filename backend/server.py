import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
# 强力 CORS 配置
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


class AmazonCompetitorMatcher:
    def __init__(self, rainforest_api_key):
        self.rainforest_api_key = rainforest_api_key
        self.rainforest_url = "https://api.rainforestapi.com/request"
        # 移除 Hugging Face 相关配置，我们不再需要它了

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

    def calculate_local_similarity(self, texts):
        """
        ✅ 核心替代方案：本地 TF-IDF 算法
        不依赖外部 API，利用统计学原理计算文本相似度。
        对于包含具体参数（如 '14 Stufen', '90kg'）的产品描述，这种方法非常精准。
        """
        try:
            print(f"🧠 Running Local TF-IDF for {len(texts)} texts...")
            # 初始化向量化器 (自动处理德语停用词需下载nltk，这里用默认配置足够)
            vectorizer = TfidfVectorizer()

            # 将文本转换为 TF-IDF 矩阵
            tfidf_matrix = vectorizer.fit_transform(texts)

            # 计算余弦相似度
            # 第一个向量(tfidf_matrix[0:1])是我的产品
            # 后面的向量(tfidf_matrix[1:])是竞品
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            return cosine_similarities
        except Exception as e:
            print(f"❌ Local Algo Error: {e}")
            # 如果只有一段文本（没有竞品），会报错，返回空
            return [0.0] * (len(texts) - 1)

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

        # 2. 获取详情
        all_texts = [my_desc]
        valid_candidates = []

        print("⏳ Fetching details...")
        for item in candidates:
            dt = self.get_product_details(item['id'])
            if dt:
                item['desc_text'] = dt
                all_texts.append(dt)  # 本地算法没有长度限制，可以使用全文！
                valid_candidates.append(item)

        if not valid_candidates: return None, []

        # 3. 本地计算相似度 (取代 HF API)
        similarity_scores = self.calculate_local_similarity(all_texts)

        best_match = None
        highest_score = -1

        # 4. 整理结果
        for i, item in enumerate(valid_candidates):
            # 获取分数
            if i < len(similarity_scores):
                score = float(similarity_scores[i])
            else:
                score = 0.0

            item['similarity'] = score
            # 截取一段描述用于前端展示
            item['features'] = item['desc_text'][:100] + "..."

            if score > highest_score:
                highest_score = score
                best_match = item

        return best_match, valid_candidates


# --- Route ---

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "algorithm": "Local TF-IDF (Stable)"}), 200


@app.route('/api/find-competitor', methods=['POST', 'OPTIONS'])
def find_competitor():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    data = request.json
    keyword = data.get('keyword', '')
    description = data.get('description', '')

    r_key = os.environ.get("RAINFOREST_API_KEY")
    # 注意：我们不再检查 HF_TOKEN，因为不需要了

    if not r_key:
        response = jsonify({"error": "Missing RAINFOREST_API_KEY"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

    try:
        matcher = AmazonCompetitorMatcher(r_key)
        best, all_results = matcher.search_and_match(description, keyword)
        return jsonify({"success": True, "best_match": best, "all_candidates": all_results})
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)