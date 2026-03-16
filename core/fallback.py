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
    Extract keywords using jieba word segmentation (nouns and adjectives)

    Args:
        text: Input text to process
        extract_nouns: Whether to extract nouns
        extract_adjectives: Whether to extract adjectives
        extract_verbs: Whether to extract verbs
        min_word_length: Minimum word length (in characters)
        max_keywords: Maximum number of keywords to return
        default_score: Default importance score

    Returns:
        Keyword list in format [["jieba-fallback", "keyword", score], ...]
    """
    if not isinstance(text, str) or not text.strip():
        logging.warning("jieba fallback: input text is empty")
        return []
    
    try:
        # POS tagging with jieba
        words = pseg.cut(text)
        
        # Define target POS tags
        target_pos = []
        if extract_nouns:
            # n: noun, nr: person, ns: place, nt: org, nz: other proper noun
            target_pos.extend(['n', 'nr', 'ns', 'nt', 'nz', 'vn', 'an'])
        if extract_adjectives:
            # a: adjective, ad: adv-adj, an: noun-adj
            target_pos.extend(['a', 'ad'])
        if extract_verbs:
            # v: verb, vd: adv-verb, vn: noun-verb
            target_pos.extend(['v', 'vd'])
        
        # Extract matching words
        keywords = []
        seen = set()  # deduplication
        
        for word, flag in words:
            # Filter conditions
            if (flag in target_pos and 
                len(word) >= min_word_length and 
                word not in seen and
                word.strip()):  # ensure not blank
                
                keywords.append(word)
                seen.add(word)
        
        # Limit keyword count
        if len(keywords) > max_keywords:
            keywords = keywords[:max_keywords]
        
        # Convert to standard format: ["jieba-fallback", "keyword", score]
        result = [["jieba-fallback", keyword, default_score] for keyword in keywords]
        
        logging.info(f"jieba fallback extracted {len(result)} keywords")
        
        return result
        
    except Exception as e:
        logging.error(f"jieba fallback extraction failed: {e}")
        return []


def extract_keywords_with_jieba_tfidf(text, topK=20, default_score=0.6):
    """
    Extract keywords using jieba TF-IDF algorithm (smarter fallback)

    Args:
        text: Input text to process
        topK: Return top K keywords
        default_score: Default importance score baseline

    Returns:
        Keyword list in format [["jieba-TFIDF", "keyword", score], ...]
    """
    if not isinstance(text, str) or not text.strip():
        logging.warning("jieba-TFIDF fallback: input text is empty")
        return []
    
    try:
        import jieba.analyse
        
        # Extract keywords with TF-IDF (with weights)
        keywords_with_weights = jieba.analyse.extract_tags(
            text, 
            topK=topK, 
            withWeight=True
        )
        
        # Convert to standard format: ["jieba-TFIDF", "keyword", score]
        # TF-IDF weights are usually 0-1, can be used as scores directly
        result = [
            ["jieba-TFIDF", keyword, min(weight * 2, 1.0)]  # weight*2 capped at 1.0
            for keyword, weight in keywords_with_weights
        ]
        
        logging.info(f"jieba-TFIDF fallback extracted {len(result)} keywords")
        
        return result
        
    except Exception as e:
        logging.error(f"jieba-TFIDF fallback extraction failed: {e}")
        # Degrade to basic jieba segmentation
        return extract_keywords_with_jieba(text, max_keywords=topK, default_score=default_score)


def extract_keywords_with_jieba_textrank(text, topK=20, default_score=0.6):
    """
    Extract keywords using jieba TextRank algorithm (graph-based, better for long texts)

    Args:
        text: Input text to process
        topK: Return top K keywords
        default_score: Default importance score baseline

    Returns:
        Keyword list in format [["jieba-TextRank", "keyword", score], ...]
    """
    if not isinstance(text, str) or not text.strip():
        logging.warning("jieba-TextRank fallback: input text is empty")
        return []
    
    try:
        import jieba.analyse
        
        # Extract keywords with TextRank (with weights)
        keywords_with_weights = jieba.analyse.textrank(
            text, 
            topK=topK, 
            withWeight=True
        )
        
        # Convert to standard format: ["jieba-TextRank", "keyword", score]
        result = [
            ["jieba-TextRank", keyword, min(weight * 2, 1.0)]
            for keyword, weight in keywords_with_weights
        ]
        
        logging.info(f"jieba-TextRank fallback extracted {len(result)} keywords")
        
        return result
        
    except Exception as e:
        logging.error(f"jieba-TextRank fallback extraction failed: {e}")
        # Degrade to basic jieba segmentation
        return extract_keywords_with_jieba(text, max_keywords=topK, default_score=default_score)


# Config: select which jieba fallback method to use
JIEBA_FALLBACK_METHOD = "tfidf"  # Options: "simple", "tfidf", "textrank"


def jieba_fallback_extract(text, method=None, topK=20):
    """
    Unified jieba fallback interface

    Args:
        text: Input text to process
        method: Method to use ("simple", "tfidf", "textrank"), None uses default config
        topK: Number of keywords to return

    Returns:
        Keyword list in format [["method_label", "keyword", score], ...]
    """
    if method is None:
        method = JIEBA_FALLBACK_METHOD
    
    logging.info(f"Starting jieba fallback extraction, method: {method}")
    
    if method == "tfidf":
        return extract_keywords_with_jieba_tfidf(text, topK=topK)
    elif method == "textrank":
        return extract_keywords_with_jieba_textrank(text, topK=topK)
    elif method == "simple":
        return extract_keywords_with_jieba(text, max_keywords=topK)
    else:
        logging.warning(f"Unknown jieba method: {method}, using TF-IDF")
        return extract_keywords_with_jieba_tfidf(text, topK=topK)


if __name__ == "__main__":
    # Test code
    test_text = "这套快速衣穿着非常好，一点不显胖，透性很好，水里一洗非常方便，一会会就干，实惠到家了"
    
    print("=" * 80)
    print("Test text:", test_text)
    print("=" * 80)
    
    print("\n1. Simple segmentation (nouns+adjectives):")
    result1 = extract_keywords_with_jieba(test_text)
    for item in result1:
        print(f"  {item}")
    
    print("\n2. TF-IDF extraction:")
    result2 = extract_keywords_with_jieba_tfidf(test_text, topK=10)
    for item in result2:
        print(f"  {item}")
    
    print("\n3. TextRank extraction:")
    result3 = extract_keywords_with_jieba_textrank(test_text, topK=10)
    for item in result3:
        print(f"  {item}")
    
    print("\n4. Unified interface:")
    result4 = jieba_fallback_extract(test_text, method="tfidf", topK=10)
    for item in result4:
        print(f"  {item}")
