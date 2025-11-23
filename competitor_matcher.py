import numpy as np
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity

# 尝试导入 sentence-transformers，如果没有安装则使用随机向量（仅供测试流程）
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("⚠️ 警告: 未安装 sentence-transformers。正在使用模拟向量模式，匹配结果将不准确。")
    print("请运行: pip install sentence-transformers")
    SentenceTransformer = None


class AmazonCompetitorMatcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.rainforestapi.com/request"

        print("正在初始化 AI 模型 (第一次运行可能需要几秒钟下载模型)...")
        if SentenceTransformer:
            # 使用多语言模型，支持德语、英语等
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        else:
            self.model = None

    def get_embedding(self, text):
        """将文本转换为向量"""
        if self.model and text:
            return self.model.encode(text)
        else:
            return np.random.rand(384)

    def _make_api_request(self, params):
        """发送请求给 Rainforest API 的通用函数"""
        params['api_key'] = self.api_key
        # 默认使用德国站点 (因为你的输入文本是德语)，如果是美国请改为 amazon.com
        if 'amazon_domain' not in params:
            params['amazon_domain'] = 'amazon.de'

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # 检查 HTTP 错误
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API 请求失败: {e}")
            return None

    def get_product_details_from_api(self, asin):
        """
        [第二步] 根据 ASIN 获取单个产品的详细信息（主要是五点描述）
        消耗: 1 个 API 积分
        """
        print(f"   ↳ 正在深入分析竞品详情 (ASIN: {asin})...")
        params = {
            'type': 'product',
            'asin': asin
        }
        data = self._make_api_request(params)

        if not data or 'product' not in data:
            return ""

        product = data['product']

        # 提取标题和五点描述 (Feature Bullets)
        # 五点描述是判断是否为“同款”的最关键信息
        title = product.get('title', '')
        bullets = product.get('feature_bullets', [])
        description = product.get('description', '')

        # 将标题和五点描述合并成一个长文本，用于 AI 向量分析
        full_text = f"{title}. " + " ".join(bullets)
        if not bullets:
            full_text += description  # 如果没有五点描述，用长描述兜底

        return full_text

    def search_amazon_real(self, keyword, limit=1):
        """
        [第一步] 在亚马逊搜索关键词，获取候选列表
        消耗: 1 个 API 积分
        """
        print(f"🔍 正在亚马逊 (amazon.de) 搜索: '{keyword}' ...")
        params = {
            'type': 'search',
            'search_term': keyword,
            'sort_by': 'featured'  # 或者 'price_low_to_high'
        }

        data = self._make_api_request(params)

        candidates = []
        if data and 'search_results' in data:
            # 只取前 N 个结果，为了节省 API 额度 (因为后面还要查详情)
            for item in data['search_results'][:limit]:
                # 提取价格
                price_val = 0.0
                currency = "EUR"
                if 'price' in item and item['price']:
                    price_val = item['price'].get('value', 0.0)
                    currency = item['price'].get('currency', 'EUR')

                candidates.append({
                    'id': item.get('asin'),
                    'title': item.get('title'),
                    'price': price_val,
                    'currency': currency,
                    'link': item.get('link'),
                    'sales_estimation': item.get('ratings_total', 0),  # 用评论数粗略估算热度
                    'image': item.get('image'),
                    'basic_text': item.get('title')  # 搜索结果通常只有标题，没有详细描述
                })

        print(f"✅ 找到 {len(candidates)} 个潜在竞品，准备进行深度比对...")
        return candidates

    def find_best_competitor(self, my_product_desc, search_keyword="Bodenstuhl"):
        """
        核心流程：
        1. 搜索关键词 -> 得到列表
        2. 遍历列表 -> 获取每个产品的详细描述 (五点描述)
        3. AI 比对 -> 找到最相似的
        """
        # 1. 获取竞品列表 (Top 3)
        # 注意：这里我们限制只取前 3 个，因为每深入分析一个都需要消耗额外的 API 积分
        competitors = self.search_amazon_real(search_keyword, limit=1)

        if not competitors:
            return None, []

        # 2. 计算我的产品向量
        my_vector = self.get_embedding(my_product_desc)

        print("\n--- 开始 AI 深度语义匹配 (耗时操作) ---")
        best_match = None
        highest_score = -1

        results = []

        for item in competitors:
            # [关键步骤] 调用 API 获取该竞品的详细“五点描述”
            # 因为仅靠标题无法区分“14档调节”和“5档调节”
            detailed_text = self.get_product_details_from_api(item['id'])

            if not detailed_text:
                detailed_text = item['title']  # 降级处理

            # 计算竞品向量
            item_vector = self.get_embedding(detailed_text)

            # 计算相似度
            score = cosine_similarity(my_vector.reshape(1, -1), item_vector.reshape(1, -1))[0][0]

            item['similarity_score'] = score
            item['detailed_desc_preview'] = detailed_text[:100] + "..."  # 仅用于打印预览
            results.append(item)

            print(f"👉 竞品: {item['title'][:20]}... | 价格: {item['price']} | 相似度: {score:.4f}")

            if score > highest_score:
                highest_score = score
                best_match = item

        print("-----------------------")
        return best_match, results


# --- 主程序入口 ---

if __name__ == "__main__":
    # 🔴 请在这里填入你的 Rainforest API Key
    API_KEY = "BF906805A6BA464EB9F10AE1819CE777"

    # 你的产品详细描述 (包含关键参数：14档调节, 90-180度, 90kg承重等)
    my_product_text = """
    14 Stufen einstellbar – Von 90° bis 180° lässt sich dieser Bodenstuhl leicht in 14 Stufen einstellen. 
    Stellen Sie den Stuhl auf den Boden, heben Sie die Rückenlehne an und stellen Sie sie nach Bedarf in eine bequeme Position.
    Multifunktionales Bodensofa – Egal ob Sie lesen, das Handyspiel spielen, fernsehen, meditieren oder bei Ihrem Haustier bleiben, dieser gepolsterte Bodenstuhl bietet optimalen Komfort.
    Optimaler Sitz- & Liegekomfort – Mit der Sitztiefe von 48 cm und der Dicke von 14 cm bietet dieses gepolsterte Kissen mit hochdichtem Schaumstoff den optimalen Sitz- und Liegekomfort.
    Platzsparend – Dank des klappbaren Designs ist dieses Sofa leicht zu transportieren und verstauen. Bei Nichtgebrauch können Sie es unter dem Bett oder im Schrank aufbewahren.
    Robust & dauerhaft – Dieses Bodensofa besteht aus robustem Plüsch-Gewebe, hochdichtem Schaumstoff und dem soliden Metallrahmen. Mit einer maximalen Belastbarkeit von 90 kg ist dieses Sofa eher für mittelgroßen Benutzer.
    """

    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ 错误: 请先在代码中填入你的 Rainforest API Key！")
    else:
        # 初始化匹配器
        matcher = AmazonCompetitorMatcher(API_KEY)

        # 运行匹配 (搜索关键词: Bodenstuhl)
        winner, all_data = matcher.find_best_competitor(my_product_text, search_keyword="Bodenstuhl")

        if winner:
            print(f"\n✅ 找到最精准竞品 (ASIN: {winner['id']}):")
            print(f"标题: {winner['title']}")
            print(f"价格: {winner['price']} {winner['currency']}")
            print(f"相似度得分: {winner['similarity_score']:.4f}")
            print(f"链接: {winner['link']}")

            print(f"\n💡 定价建议: ")
            if winner['price'] > 0:
                print(f"市场最相似竞品定价为 {winner['price']} {winner['currency']}。")
                print(f"建议定价范围: {winner['price'] - 1:.2f} - {winner['price'] + 2:.2f} {winner['currency']}")
            else:
                print("竞品当前缺货或无价格，无法提供具体建议。")
        else:
            print("未找到匹配竞品。")