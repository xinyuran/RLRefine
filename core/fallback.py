import jieba
import jieba.posseg as pseg
import logging


def extract_keywords_with_jieba(text, 
                                 extract_nouns=True, 
                                 extract_adjectives=True,
                                 extract_verbs=False,
                                 min_word_length=2,
                                 max_keywords=20,
                                 default_score=0.5):
    """
    使用 jieba 分词提取关键词（名词和形容词）
    
    Args:
        text: 待处理的文本
        extract_nouns: 是否提取名词
        extract_adjectives: 是否提取形容词
        extract_verbs: 是否提取动词
        min_word_length: 最小词长（字符数）
        max_keywords: 最多返回的关键词数量
        default_score: 默认重要性分数
        
    Returns:
        关键词列表，格式为 [["jieba兜底", "关键词", 分数], ...]
    """
    if not isinstance(text, str) or not text.strip():
        logging.warning("jieba兜底: 输入文本为空")
        return []
    
    try:
        # 使用 jieba 进行词性标注
        words = pseg.cut(text)
        
        # 定义要提取的词性
        target_pos = []
        if extract_nouns:
            # n: 名词, nr: 人名, ns: 地名, nt: 机构名, nz: 其他专名
            target_pos.extend(['n', 'nr', 'ns', 'nt', 'nz', 'vn', 'an'])
        if extract_adjectives:
            # a: 形容词, ad: 副形词, an: 名形词
            target_pos.extend(['a', 'ad'])
        if extract_verbs:
            # v: 动词, vd: 副动词, vn: 名动词
            target_pos.extend(['v', 'vd'])
        
        # 提取符合条件的词
        keywords = []
        seen = set()  # 去重
        
        for word, flag in words:
            # 过滤条件
            if (flag in target_pos and 
                len(word) >= min_word_length and 
                word not in seen and
                word.strip()):  # 确保不是空白
                
                keywords.append(word)
                seen.add(word)
        
        # 限制关键词数量
        if len(keywords) > max_keywords:
            keywords = keywords[:max_keywords]
        
        # 转换为标准格式: ["jieba兜底", "关键词", 分数]
        result = [["jieba兜底提取", keyword, default_score] for keyword in keywords]
        
        logging.info(f"jieba兜底成功提取 {len(result)} 个关键词")
        
        return result
        
    except Exception as e:
        logging.error(f"jieba兜底提取失败: {e}")
        return []


def extract_keywords_with_jieba_tfidf(text, topK=20, default_score=0.6):
    """
    使用 jieba 的 TF-IDF 算法提取关键词（更智能的兜底方案）
    
    Args:
        text: 待处理的文本
        topK: 返回前 K 个关键词
        default_score: 默认重要性分数基准
        
    Returns:
        关键词列表，格式为 [["jieba-TFIDF", "关键词", 分数], ...]
    """
    if not isinstance(text, str) or not text.strip():
        logging.warning("jieba-TFIDF兜底: 输入文本为空")
        return []
    
    try:
        import jieba.analyse
        
        # 使用 TF-IDF 提取关键词（带权重）
        keywords_with_weights = jieba.analyse.extract_tags(
            text, 
            topK=topK, 
            withWeight=True
        )
        
        # 转换为标准格式: ["jieba-TFIDF", "关键词", 分数]
        # TF-IDF 权重通常在 0-1 之间，可以直接作为分数
        result = [
            ["jieba-TFIDF提取", keyword, min(weight * 2, 1.0)]  # 权重*2但不超过1.0
            for keyword, weight in keywords_with_weights
        ]
        
        logging.info(f"jieba-TFIDF兜底成功提取 {len(result)} 个关键词")
        
        return result
        
    except Exception as e:
        logging.error(f"jieba-TFIDF兜底提取失败: {e}")
        # 降级到普通 jieba 分词
        return extract_keywords_with_jieba(text, max_keywords=topK, default_score=default_score)


def extract_keywords_with_jieba_textrank(text, topK=20, default_score=0.6):
    """
    使用 jieba 的 TextRank 算法提取关键词（图算法，更适合长文本）
    
    Args:
        text: 待处理的文本
        topK: 返回前 K 个关键词
        default_score: 默认重要性分数基准
        
    Returns:
        关键词列表，格式为 [["jieba-TextRank", "关键词", 分数], ...]
    """
    if not isinstance(text, str) or not text.strip():
        logging.warning("jieba-TextRank兜底: 输入文本为空")
        return []
    
    try:
        import jieba.analyse
        
        # 使用 TextRank 提取关键词（带权重）
        keywords_with_weights = jieba.analyse.textrank(
            text, 
            topK=topK, 
            withWeight=True
        )
        
        # 转换为标准格式: ["jieba-TextRank", "关键词", 分数]
        result = [
            ["jieba-TextRank提取", keyword, min(weight * 2, 1.0)]
            for keyword, weight in keywords_with_weights
        ]
        
        logging.info(f"jieba-TextRank兜底成功提取 {len(result)} 个关键词")
        
        return result
        
    except Exception as e:
        logging.error(f"jieba-TextRank兜底提取失败: {e}")
        # 降级到普通 jieba 分词
        return extract_keywords_with_jieba(text, max_keywords=topK, default_score=default_score)


# 配置：选择使用哪种 jieba 兜底方法
JIEBA_FALLBACK_METHOD = "tfidf"  # 可选: "simple", "tfidf", "textrank"


def jieba_fallback_extract(text, method=None, topK=20):
    """
    统一的 jieba 兜底接口
    
    Args:
        text: 待处理的文本
        method: 使用的方法 ("simple", "tfidf", "textrank")，None 则使用默认配置
        topK: 返回的关键词数量
        
    Returns:
        关键词列表，格式为 [["方法说明", "关键词", 分数], ...]
    """
    if method is None:
        method = JIEBA_FALLBACK_METHOD
    
    logging.info(f"🔧 启动 jieba 兜底提取，方法: {method}")
    
    if method == "tfidf":
        return extract_keywords_with_jieba_tfidf(text, topK=topK)
    elif method == "textrank":
        return extract_keywords_with_jieba_textrank(text, topK=topK)
    elif method == "simple":
        return extract_keywords_with_jieba(text, max_keywords=topK)
    else:
        logging.warning(f"未知的 jieba 方法: {method}，使用 TF-IDF")
        return extract_keywords_with_jieba_tfidf(text, topK=topK)


if __name__ == "__main__":
    # 测试代码
    test_text = "这套快速衣穿着非常好，一点不显胖，透性很好，水里一洗非常方便，一会会就干，实惠到家了"
    
    print("=" * 80)
    print("测试文本:", test_text)
    print("=" * 80)
    
    print("\n1. 简单分词提取（名词+形容词）:")
    result1 = extract_keywords_with_jieba(test_text)
    for item in result1:
        print(f"  {item}")
    
    print("\n2. TF-IDF 提取:")
    result2 = extract_keywords_with_jieba_tfidf(test_text, topK=10)
    for item in result2:
        print(f"  {item}")
    
    print("\n3. TextRank 提取:")
    result3 = extract_keywords_with_jieba_textrank(test_text, topK=10)
    for item in result3:
        print(f"  {item}")
    
    print("\n4. 统一接口:")
    result4 = jieba_fallback_extract(test_text, method="tfidf", topK=10)
    for item in result4:
        print(f"  {item}")

