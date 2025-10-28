import jieba

def extract_core_words(text):
    stop_words = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "个", "也", "很"}
    words = set([w for w in jieba.lcut(text) if len(w) >= 2 and w not in stop_words])
    food_core_words = {"菜", "饭", "面", "汤", "肉", "鱼", "蛋", "虾", "咸", "淡", "辣", "凉", "热", "冷", "硬", "软", "多", "少", "足", "够", "快", "慢", "好", "差", "优", "劣", "鲜", "腥", "臭", "香", "贵", "便宜", "值", "不值", "包装", "分量", "味道", "口感", "服务", "态度", "配送", "时效", "保温", "卫生"}
    return words & food_core_words
