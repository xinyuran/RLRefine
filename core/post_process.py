# Keyword post-processing module
# For deduplication, sorting, filtering of model-returned keywords

import os
import re
import logging


# Stopword cache dict: {file_path: stopword_set}
_stopwords_cache = {}


def find_min_span_in_text(keyword, original_text):
    """
    Find the minimum contiguous span in the original text that matches all characters of the keyword.
    
    Uses a greedy algorithm: starting from each occurrence of the keyword's first character
    in the text, sequentially searches forward for remaining characters to compute the
    minimum span needed.
    
    Args:
        keyword: The keyword to match
        original_text: The original text
        
    Returns:
        Length of the minimum matching span, or -1 if no match is found
    """
    if not keyword or not original_text:
        return -1
    
    # Filter out spaces from keyword
    keyword_chars = [c for c in keyword if c != ' ']
    if not keyword_chars:
        return -1
    
    min_span = float('inf')
    
    # Find all positions of the first character in the text
    first_char = keyword_chars[0]
    start_positions = [i for i, c in enumerate(original_text) if c == first_char]
    
    # For each start position, attempt greedy matching
    for start_pos in start_positions:
        current_pos = start_pos
        matched = True
        
        # Match remaining characters sequentially
        for i, char in enumerate(keyword_chars):
            if i == 0:
                continue  # First character already matched
            
            # Search forward from current position for the next character
            found = False
            for j in range(current_pos + 1, len(original_text)):
                if original_text[j] == char:
                    current_pos = j
                    found = True
                    break
            
            if not found:
                matched = False
                break
        
        if matched:
            # Calculate the span of this match (from start to last matched position)
            span = current_pos - start_pos + 1
            min_span = min(min_span, span)
    
    return min_span if min_span != float('inf') else -1


def validate_keyword_chars_in_text(keyword, original_text, max_span_ratio=2):
    """
    Validate that each character of the keyword exists compactly in the original text.
    
    This function filters out keywords that the model "inferred/summarized on its own"
    and do not actually exist in the original text.
    
    Validation rules:
    1. Every character of the keyword must exist in the original text
    2. The minimum matching span in the text must not exceed max_span_ratio times the keyword length
    
    Examples (assuming max_span_ratio=2):
    - Text "已经退货", keyword "没退货" -> invalid ("没" not in text)
    - Text "衣服不怎么粘肉", keyword "不粘肉" -> valid (span=5, keyword length=3, ratio 1.67<2)
    - Text "聊个不停...一天", keyword "聊天" -> invalid (characters too scattered, span far exceeds 2x)
    
    Args:
        keyword: The keyword to validate
        original_text: The original text (preprocessed text recommended)
        max_span_ratio: Maximum span ratio (default 2, i.e. matching span must not exceed 2x keyword length)
        
    Returns:
        True if the keyword passes validation, False otherwise
    """
    if not isinstance(keyword, str) or not isinstance(original_text, str):
        return False
    
    if not keyword or not original_text:
        return False
    
    # Filter out spaces from keyword
    keyword_chars = [c for c in keyword if c != ' ']
    if not keyword_chars:
        return False
    
    keyword_len = len(keyword_chars)
    
    # 1. First check that every character exists in the original text
    original_chars = set(original_text)
    for char in keyword_chars:
        if char not in original_chars:
            return False
    
    # 2. Check compactness of characters in the text (minimum matching span)
    min_span = find_min_span_in_text(keyword, original_text)
    
    if min_span < 0:
        # No match found, character order is wrong or characters don't exist
        return False
    
    # Calculate the maximum allowed span
    max_allowed_span = keyword_len * max_span_ratio
    
    if min_span > max_allowed_span:
        logging.debug(f"[Text validation] Keyword '{keyword}' span too large: min_span={min_span}, max_allowed={max_allowed_span}")
        return False
    
    return True


def filter_keywords_not_in_original(keywords_data, original_text, keyword_idx=1, max_span_ratio=2):
    """
    Filter out keywords that do not exist in the original text.
    
    Iterates through the keyword list, checking that all characters of each keyword
    exist compactly in the original text. Filters out keywords containing characters
    not in the text, as well as keywords with characters scattered too far apart.
    
    Args:
        keywords_data: Keyword data list, format [[reasoning, keyword, score], ...]
        original_text: The original text (preprocessed text recommended)
        keyword_idx: Index position of keyword in the list (1 for new format, 0 for old format)
        max_span_ratio: Maximum span ratio (default 2, i.e. matching span must not exceed 2x keyword length)
        
    Returns:
        Filtered keyword data list
    """
    if not keywords_data or not original_text:
        return keywords_data
    
    filtered_data = []
    filtered_out = []  # Track filtered keywords (for debugging)
    
    for item in keywords_data:
        if len(item) > keyword_idx:
            keyword = item[keyword_idx]
            
            # Ensure keyword is a string type
            if not isinstance(keyword, str):
                continue
            
            # Validate that each character of the keyword exists compactly in the text
            if validate_keyword_chars_in_text(keyword, original_text, max_span_ratio):
                filtered_data.append(item)
            else:
                filtered_out.append(keyword)
    
    # Debug output
    if filtered_out:
        logging.info(f"[Text validation filter] Before: {len(keywords_data)}, after: {len(filtered_data)}, filtered out: {filtered_out}")
    
    return filtered_data



# ==================== Common Chinese numeral patterns ====================
# Includes: 零一二三四五六七八九十百千万亿 (simplified)
#           壹贰叁肆伍陆柒捌玖拾佰仟萬億 (traditional/uppercase)
#           〇两 (special numerals)
CHINESE_NUM_PATTERN = r'[零〇一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億两]+'
# Arabic numeral pattern
ARABIC_NUM_PATTERN = r'\d+'
# Any numeral pattern (Chinese or Arabic)
ANY_NUM_PATTERN = f'(?:{CHINESE_NUM_PATTERN}|{ARABIC_NUM_PATTERN})'


def is_date_keyword(keyword):
    """
    Determine whether a keyword is a date-related term.
    
    Detection rules:
    - Year: 2024年, 二零二四年, 二千零二十四年
    - Month: 1月, 一月, 01月
    - Day: 1号, 一号, 01号, 1日, 一日
    - Full date: 5月15, 五月十五号, 2024年11月
    - Day of week: 周一, 星期一, 礼拜一
    - Relative date: 今天, 昨天, 明天, 前天, 后天
    
    Args:
        keyword: The keyword to check
        
    Returns:
        True if it is a date keyword, False otherwise
    """
    if not isinstance(keyword, str):
        return False
    
    # ===== 1. Year detection =====
    
    # 1.1 Numeric year: 2024年, 24年, any digits + 年
    if re.search(ARABIC_NUM_PATTERN + r'\s*年', keyword):
        return True
    
    # 1.2 Chinese year: 二零二四年, 二〇二四年, 二千零二十四年
    if re.search(CHINESE_NUM_PATTERN + r'\s*年', keyword):
        return True
    
    # ===== 2. Month detection =====
    
    # 2.1 Numeric month: 1月, 01月, 12月
    if re.search(ARABIC_NUM_PATTERN + r'\s*月', keyword):
        return True
    
    # 2.2 Chinese month: 一月, 十二月, any Chinese numeral + 月
    if re.search(CHINESE_NUM_PATTERN + r'\s*月', keyword):
        return True
    
    # ===== 3. Day detection =====
    
    # 3.1 Numeric day: 1号, 01号, 1日, 01日
    if re.search(ARABIC_NUM_PATTERN + r'\s*[号日]', keyword):
        return True
    
    # 3.2 Chinese day: 一号, 二十七号, 三十一日
    if re.search(CHINESE_NUM_PATTERN + r'\s*[号日]', keyword):
        return True
    
    # 3.3 Standalone "号" or "日" (in specific contexts)
    if keyword == '号' or keyword == '日':
        return True
    
    # ===== 4. Full date formats =====
    
    # 4.1 Month-day combination: 5月15, 11.15, 11-15, 11/15
    if re.search(ARABIC_NUM_PATTERN + r'\s*月\s*' + ARABIC_NUM_PATTERN, keyword):
        return True
    if re.search(r'\d{1,2}[\./-]\d{1,2}', keyword):
        return True
    
    # 4.2 Chinese month-day combination: 五月十五
    if re.search(CHINESE_NUM_PATTERN + r'\s*月\s*' + CHINESE_NUM_PATTERN, keyword):
        return True
    
    # 4.3 Year-month-day combination: 2024年11月15日
    if re.search(ANY_NUM_PATTERN + r'\s*年\s*' + ANY_NUM_PATTERN + r'\s*月\s*' + ANY_NUM_PATTERN + r'\s*[日号]?', keyword):
        return True
    
    # ===== 5. Day of week detection =====
    
    # 5.1 Day of week: 周X, 星期X, 礼拜X (using generic patterns)
    if re.search(r'周[一二三四五六日天]', keyword):
        return True
    if re.search(r'星期[一二三四五六日天]', keyword):
        return True
    if re.search(r'礼拜[一二三四五六日天]', keyword):
        return True
    
    # 5.2 Workday, weekend
    if '工作日' in keyword or '周末' in keyword or '双休' in keyword:
        return True
    
    # ===== 6. Relative dates =====
    
    relative_dates = [
        '今天', '今日', '今儿',
        '昨天', '昨日', '昨儿',
        '明天', '明日', '明儿',
        '前天', '前日',
        '后天', '后日',
        '大前天', '大后天'
    ]
    for date in relative_dates:
        if date in keyword:
            return True
    
    # ===== 7. Other date expressions =====
    
    # 7.1 Beginning/middle/end of month: 月初, 月中, 月末, 月底
    if re.search(r'月[初中末底]', keyword):
        return True
    
    # 7.2 First/middle/last ten days: 上旬, 中旬, 下旬
    if '旬' in keyword and any(x in keyword for x in ['上', '中', '下']):
        return True
    
    # 7.3 Quarter: any numeral + 季度, Q + digits
    if re.search(r'第?' + ANY_NUM_PATTERN + r'\s*季度', keyword):
        return True
    if re.search(r'[Qq]' + ARABIC_NUM_PATTERN, keyword):
        return True
    
    return False


def is_time_keyword(keyword):
    """
    Determine whether a keyword is a time-related term.
    
    Uses the same detection rules as the preprocessing stage for consistency.
    
    Detection rules (consistent with remove_time_expressions in preprocess_v2.py):
    - Standard time formats: 8:40, 08:40:30, 8.40, 8点40分30秒
    - Colloquial time: 8点多, 8点半, 8点左右, 差5分8点
    - Time periods: 早上, 上午, 中午, 下午, 晚上, 凌晨, 夜里
    - Fuzzy time: 刚才, 现在, 马上, 立刻, 稍后
    
    Args:
        keyword: The keyword to check
        
    Returns:
        True if it is a time keyword, False otherwise
    """
    if not isinstance(keyword, str):
        return False
    
    # ===== 1. Standard time formats =====
    
    # 1.1 Colon-separated time: 8:40, 08:40:30, 23:59:59
    if re.search(r'\d{1,2}:\d{1,2}(:\d{1,2})?', keyword):
        return True
    
    # 1.2 Dot-separated time: 8.40, 8.40.30 (excludes prices)
    if re.search(r'(?<!\d)\d{1,2}\.\d{1,2}(\.\d{1,2})?(?!\d)', keyword):
        return True
    
    # 1.3 Chinese full time: 8点40分30秒, 8点40分, 8点40, 八点四十分 (using generic numeral pattern)
    if re.search(ANY_NUM_PATTERN + r'\s*点\s*' + ANY_NUM_PATTERN + r'\s*分\s*' + ANY_NUM_PATTERN + r'\s*秒', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*点\s*' + ANY_NUM_PATTERN + r'\s*分', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*点\s*' + ANY_NUM_PATTERN + r'(?![分秒])', keyword):
        return True
    
    # 1.4 Chinese time units: 8点, 40分, 30秒, 8时, 40分钟 (numeric or Chinese numeral)
    if re.search(ANY_NUM_PATTERN + r'\s*[点时]\s*(?:钟)?', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*分\s*(?:钟)?', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*秒\s*(?:钟)?', keyword):
        return True
    
    # ===== 2. Colloquial time expressions =====
    
    # 2.1 Approximate time: 8点多, 8点半, 8点左右, 8点钟左右 (numeric or Chinese numeral)
    if re.search(ANY_NUM_PATTERN + r'\s*[点时]\s*[多半来钟]', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*[点时]\s*(?:左右|上下)', keyword):
        return True
    
    # 2.2 "X minutes to Y o'clock": 差5分8点, 差一刻9点 (numeric or Chinese numeral)
    if re.search(r'差\s*' + ANY_NUM_PATTERN + r'\s*分\s*' + ANY_NUM_PATTERN + r'\s*[点时]', keyword):
        return True
    if re.search(r'差\s*' + CHINESE_NUM_PATTERN + r'\s*刻\s*' + ANY_NUM_PATTERN + r'\s*[点时]', keyword):
        return True
    
    # 2.3 "X o'clock and a quarter": 8点一刻, 9点三刻 (numeric or Chinese numeral)
    if re.search(ANY_NUM_PATTERN + r'\s*[点时]\s*' + CHINESE_NUM_PATTERN + r'\s*刻', keyword):
        return True
    
    # ===== 3. Time period expressions =====
    
    # 3.1 Time periods: 早上, 上午, 中午, 下午, 晚上, 夜里, 凌晨, 深夜
    time_periods = ['早上', '上午', '中午', '下午', '晚上', '夜里', '凌晨', '深夜', '早晨', '傍晚', '黄昏']
    for period in time_periods:
        if period in keyword:
            return True
    
    # 3.2 Full expression with time period: 上午8点, 晚上9点半 (numeric or Chinese numeral)
    if re.search(r'[早上中下晚夜凌深][上午里晨间夜]\s*' + ANY_NUM_PATTERN + r'\s*[点时]', keyword):
        return True
    
    # ===== 4. Fuzzy time words =====
    
    fuzzy_time_words = [
        '刚才', '刚刚', '现在', '此刻', '当前', '目前',
        '马上', '立刻', '立即', '立马', '即刻',
        '稍后', '待会', '一会儿', '过会', '等会',
        '随后', '之后', '然后', '接着', '紧接着'
    ]
    for word in fuzzy_time_words:
        if word in keyword:
            return True
    
    return False


def load_stopwords(stopwords_file="stopwords.txt"):
    """
    Load stopword list from file (with caching to avoid repeated loading).
    
    Args:
        stopwords_file: Path to stopword file (defaults to stopwords.txt in current directory)
        
    Returns:
        Set of stopwords
    """
    # If a relative path is provided, resolve it relative to the script directory
    if not os.path.isabs(stopwords_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stopwords_file = os.path.join(script_dir, stopwords_file)
    
    # Check if the file has already been loaded (cached)
    if stopwords_file in _stopwords_cache:
        return _stopwords_cache[stopwords_file]
    
    # First-time loading
    stopwords = set()
    
    # Check if file exists
    if not os.path.exists(stopwords_file):
        logging.warning(f"Stopword file not found: {stopwords_file}")
        _stopwords_cache[stopwords_file] = stopwords
        return stopwords
    
    try:
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip leading/trailing whitespace
                word = line.strip()
                # Skip empty lines and comment lines
                if word and not word.startswith('#'):
                    stopwords.add(word)
        logging.info(f"Loaded {len(stopwords)} stopwords (from: {os.path.basename(stopwords_file)})")
        # Cache the result
        _stopwords_cache[stopwords_file] = stopwords
    except Exception as e:
        logging.error(f"Failed to load stopword file: {e}")
        _stopwords_cache[stopwords_file] = stopwords
    
    return stopwords


def normalize_keywords_data(keywords_data, json_format="new"):
    """
    Normalize keyword data, ensuring scores are float type and keywords are strings.
    Supports automatic detection and conversion between 2-tuple and 3-tuple formats.
    
    Args:
        keywords_data: Raw keyword data
            - New format 3-tuple: [["reasoning", "keyword", score], ...]
            - New format 2-tuple: [["keyword", score], ...]  # missing reasoning
            - Old format: [["keyword", score, "reason"], ...]
        json_format: JSON format type, "new" (new format) or "old" (old format)
        
    Returns:
        Normalized keyword data (unified to 3-tuple format)
    """
    if not keywords_data:
        return []
    
    normalized_data = []
    for item in keywords_data:
        if not isinstance(item, list) or len(item) < 2:
            # Skip malformed data (at least 2 elements required)
            continue
        
        # Determine data structure based on length and format
        if json_format == "new":
            if len(item) == 3:
                # Standard 3-tuple: ["reasoning", "keyword", score]
                reasoning, keyword, score = item[0], item[1], item[2]
            elif len(item) == 2:
                # 2-tuple: ["keyword", score] - missing reasoning
                # Need to determine which is keyword and which is score
                if isinstance(item[0], str) and isinstance(item[1], (int, float, str)):
                    # Assume first is keyword, second is score
                    reasoning = ""  # Empty reasoning
                    keyword = item[0]
                    score = item[1]
                else:
                    # Ambiguous format, skip
                    continue
            else:
                # Unexpected length, skip
                continue
        else:
            # Old format: ["keyword", score, "reason"]
            if len(item) >= 2:
                keyword = item[0]
                score = item[1]
                reasoning = item[2] if len(item) > 2 else ""
            else:
                continue
        
        # Verify keyword is a string type
        if not isinstance(keyword, str):
            continue
        
        # Convert score to float
        try:
            score = float(score)
        except (ValueError, TypeError):
            # If conversion fails, use default score
            score = 0.5
        
        # Unified output as 3-tuple format: ["reasoning", "keyword", score]
        if json_format == "new":
            normalized_data.append([reasoning, keyword, score])
        else:
            normalized_data.append([keyword, score, reasoning])
    
    return normalized_data


def post_process_keywords(
    keywords_data,
    deduplicate=True,
    sort_by_importance=True,
    filter_low_score=False,
    score_threshold=0.0,
    top_n=False,
    n=10,
    return_full_info=False,
    json_format="new",
    remove_english=False,
    filter_stopwords=False,
    stopwords_exact_match=True,
    stopwords_contain_match=False,
    stopwords_file="stopwords.txt",
    filter_time_keywords=False,
    filter_date_keywords=False,
    filter_long_keywords=False,
    max_keyword_length=6,
    backfill_topn=True,
    filter_not_in_original=True,
    original_text=None,
    max_span_ratio=2
):
    """
    Post-process keyword extraction results.
    
    Args:
        keywords_data: Raw keyword data
            - New format: [["reasoning", "keyword", score], ...]
            - Old format: [["keyword", score, "reason"], ...]
        deduplicate: Whether to deduplicate (exact same words)
        sort_by_importance: Whether to sort by importance score descending
        filter_low_score: Whether to filter low-score keywords
        score_threshold: Score threshold (effective when filter_low_score=True)
        top_n: Whether to keep only top N keywords
        n: Number of keywords to keep (effective when top_n=True)
        return_full_info: Whether to return full info (including score and reason), False returns keyword list only
        json_format: JSON format type, "new" (new format) or "old" (old format)
        remove_english: Whether to remove keywords containing English letters
        filter_stopwords: Whether to enable stopword filtering
        stopwords_exact_match: Whether to enable exact match (filter when keyword exactly equals a stopword)
        stopwords_contain_match: Whether to enable contains match (filter when keyword contains a stopword)
        stopwords_file: Path to stopword file (defaults to stopwords.txt)
        filter_time_keywords: Whether to filter time-related keywords (e.g. "8点", "早上")
        filter_date_keywords: Whether to filter date-related keywords (e.g. "27号", "5月15")
        filter_long_keywords: Whether to filter overly long keywords (exceeding max_keyword_length)
        max_keyword_length: Maximum keyword length (effective when filter_long_keywords=True, default 6)
        backfill_topn: When top_n is enabled and filtered count is less than N, whether to backfill from filtered keywords
        filter_not_in_original: Whether to filter keywords not in original text (checks if each character exists in text)
        original_text: Original text (required when filter_not_in_original=True, preprocessed text recommended)
        max_span_ratio: Maximum span ratio of keyword characters in original text (default 2, prevents scattered character assembly)
        
    Returns:
        Processed keyword list or full info list
    """
    # If input is empty, return empty list directly
    if not keywords_data:
        return []
    
    # Deep copy data to avoid modifying the original
    processed_data = [item[:] for item in keywords_data]
    
    # Determine keyword and score positions (based on format)
    if json_format == "new":
        # New format: [reasoning, keyword, score]
        keyword_idx = 1
        score_idx = 2
    else:
        # Old format: [keyword, score, reason]
        keyword_idx = 0
        score_idx = 1
    
    # Define score getter function (used in multiple places)
    def get_score(item):
        """Safely get score with type conversion"""
        if len(item) <= score_idx:
            return 0
        score = item[score_idx]
        try:
            return float(score)
        except (ValueError, TypeError):
            return 0
    
    # 0. Filter empty keywords (executed first to prevent empty keywords from affecting later steps)
    # Remove items with empty string keywords
    filtered_data = []
    # logging.info(f"Before filtering empty keywords: len(processed_data) = {len(processed_data)}")
    for item in processed_data:
        if len(item) > keyword_idx:
            keyword = item[keyword_idx]
            # Ensure keyword is a non-empty string
            if isinstance(keyword, str) and keyword.strip():
                filtered_data.append(item)
    processed_data = filtered_data
    # logging.info(f"After filtering empty keywords: len(processed_data) = {len(processed_data)}")
    
    # 1. Sort by importance score (if enabled)
    if sort_by_importance:
        processed_data.sort(key=get_score, reverse=True)

    
    # 2. Deduplicate (if enabled)
    if deduplicate:
        seen_keywords = set()
        deduplicated_data = []
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # Keep only the first occurrence (since data is sorted, keeps the highest score)
                if keyword not in seen_keywords:
                    seen_keywords.add(keyword)
                    deduplicated_data.append(item)
        processed_data = deduplicated_data
    
    # 3. Filter low-score keywords (if enabled)
    if filter_low_score:
        def check_score(item):
            """Safely check if score meets the threshold"""
            if len(item) <= score_idx:
                return False
            try:
                score = float(item[score_idx])
                return score >= score_threshold
            except (ValueError, TypeError):
                return False
        
        processed_data = [item for item in processed_data if check_score(item)]
    
    # Note: top_n truncation is no longer done early here
    # 4. top_n truncation moved to after all filtering is complete (see below)
    
    # 5. Remove keywords containing English letters (if enabled)
    if remove_english:
        filtered_data = []
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # Ensure keyword is a string type
                if not isinstance(keyword, str):
                    continue  # Skip non-string keywords
                # Check if keyword contains English letters
                if not re.search(r'[a-zA-Z]', keyword):
                    filtered_data.append(item)
        processed_data = filtered_data
    
    # 6. Filter stopwords (if enabled)
    if filter_stopwords and (stopwords_exact_match or stopwords_contain_match):
        # Load stopword list
        stopwords = load_stopwords(stopwords_file)
        
        if stopwords:  # Only filter when stopwords are successfully loaded
            filtered_data = []
            filtered_out = []  # Track filtered keywords (for debugging)
            # logging.info(f"[Stopword filter] Keywords before filtering: {[item[keyword_idx] if len(item) > keyword_idx else '?' for item in processed_data]}")
            
            for item in processed_data:
                if len(item) > keyword_idx:
                    keyword = item[keyword_idx]
                    # Ensure keyword is a string type
                    if not isinstance(keyword, str):
                        # logging.info(f"[Stopword filter] Skipping non-string keyword: {keyword} (type: {type(keyword)})")
                        continue  # Skip non-string keywords
                    
                    should_filter = False
                    
                    # Exact match: keyword exactly equals a stopword
                    if stopwords_exact_match and keyword in stopwords:
                        # logging.info(f"[Stopword filter] '{keyword}' matched stopword (exact match)")
                        should_filter = True
                    
                    # Contains match: keyword contains a stopword
                    if stopwords_contain_match and not should_filter:
                        for stopword in stopwords:
                            if stopword in keyword:
                                should_filter = True
                                # logging.info(f"[Stopword filter] '{keyword}' contains stopword '{stopword}' (contains match)")
                                break
                    
                    # If not filtered, keep the keyword
                    if not should_filter:
                        filtered_data.append(item)
                    else:
                        filtered_out.append(keyword)

            # Debug output (optional, uncomment to enable)
            # logging.info(f"[Stopword filter] Before: {len(processed_data)}, after: {len(filtered_data)}, filtered out: {filtered_out}")
            
            processed_data = filtered_data
    
    # 7. Filter time keywords (if enabled)
    if filter_time_keywords:
        filtered_data = []
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # Ensure keyword is a string type
                if not isinstance(keyword, str):
                    continue  # Skip non-string keywords
                
                # Check if it is a time keyword
                if not is_time_keyword(keyword):
                    filtered_data.append(item)
        
        processed_data = filtered_data
    
    # 8. Filter date keywords (if enabled)
    if filter_date_keywords:
        filtered_data = []
        filtered_out_dates = []  # Track filtered date keywords
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # Ensure keyword is a string type
                if not isinstance(keyword, str):
                    continue  # Skip non-string keywords
                
                # Check if it is a date keyword
                if not is_date_keyword(keyword):
                    filtered_data.append(item)
                else:
                    filtered_out_dates.append(keyword)
        
        # Debug output (optional)
        # if filtered_out_dates:
        #     logging.info(f"[Date filter] Before: {len(processed_data)}, after: {len(filtered_data)}, filtered out: {filtered_out_dates}")
        
        processed_data = filtered_data
    
    # 9. Filter overly long keywords (if enabled)
    if filter_long_keywords and max_keyword_length > 0:
        filtered_data = []
        filtered_out_long = []  # Track filtered overly long keywords
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # Ensure keyword is a string type
                if not isinstance(keyword, str):
                    continue  # Skip non-string keywords
                
                # Check if keyword length exceeds the maximum
                if len(keyword) <= max_keyword_length:
                    filtered_data.append(item)
                else:
                    filtered_out_long.append(keyword)
        
        # Debug output (optional)
        # if filtered_out_long:
        #     logging.info(f"[Length filter] Before: {len(processed_data)}, after: {len(filtered_data)}, filtered out: {filtered_out_long}")
        
        processed_data = filtered_data

    # 10. Filter keywords not in the original text (if enabled)
    if filter_not_in_original and original_text:
        processed_data = filter_keywords_not_in_original(
            processed_data, 
            original_text, 
            keyword_idx=keyword_idx,
            max_span_ratio=max_span_ratio
        )


    # ========== Important: save clean data for backfilling after all filtering is done ==========
    # This ensures backfilling won't reintroduce keywords filtered by stopword/time/date/length filters
    clean_sorted_data = [item[:] for item in processed_data]
    
    # 11. Smart top_n processing (after all filtering is complete)
    if top_n and n > 0:
        # First truncate to top N
        if len(processed_data) > n:
            processed_data = processed_data[:n]
        
        current_count = len(processed_data)
        
        if current_count < n and backfill_topn:
            # Current count is less than N, backfilling needed
            # Calculate the number of items to backfill
            needed_count = n - current_count
            
            # Get the set of currently selected keywords (for deduplication)
            selected_keywords = set()
            for item in processed_data:
                if len(item) > keyword_idx:
                    keyword = item[keyword_idx]
                    if isinstance(keyword, str):
                        selected_keywords.add(keyword)
            
            # Find backfill candidates from clean data, not raw data; ensures backfilled keywords also passed stopword/time filters
            backfill_candidates = []
            for item in clean_sorted_data:
                if len(item) > keyword_idx:
                    keyword = item[keyword_idx]
                    # Ensure it's a string and not already selected
                    if isinstance(keyword, str) and keyword not in selected_keywords:
                        backfill_candidates.append(item)
            
            # Select highest-scoring candidates for backfilling
            backfill_items = backfill_candidates[:needed_count]
            
            # Append backfilled keywords to the end of results
            processed_data.extend(backfill_items)
            
            # Debug output
            if backfill_items:
                backfilled_keywords = [item[keyword_idx] for item in backfill_items if len(item) > keyword_idx]
                logging.info(f"[Smart backfill] Current: {current_count}, target: {n}, backfilled: {len(backfill_items)} -> {backfilled_keywords}")
        
        # Final truncation to N items (truncate if backfill exceeds N; normally should be exactly N or fewer)
        processed_data = processed_data[:n]
    
    # 12. Determine return format based on return_full_info
    if return_full_info:
        # Return full info
        return processed_data
    else:
        # Return keyword list only (ensure strings)
        keywords_list = []
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # Only add string-type keywords
                if isinstance(keyword, str):
                    keywords_list.append(keyword)
        return keywords_list


def extract_keywords_from_json(keywords_json, return_raw=False):
    """
    Extract keyword list from JSON-formatted keyword data.
    
    Args:
        keywords_json: JSON-formatted keyword data (dict)
        return_raw: Whether to return raw format (including score and reason)
        
    Returns:
        Keyword data list, format [["word1", 0.90, "reason"], ...] or []
    """
    # Handle returned keywords (compatible with both Chinese and English key names)
    if '关键词' in keywords_json:
        keywords_data = keywords_json['关键词']
    elif 'keywords' in keywords_json:
        keywords_data = keywords_json['keywords']
    else:
        return []
    
    return keywords_data

