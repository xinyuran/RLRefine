import re
import html
import unicodedata


def remove_time_expressions(text):
    """
    Remove time expressions from text (systematic processing)
    
    Supported time formats:
    - Standard time: 8:40, 08:40:30, 8.40, 8点40分30秒
    - Colloquial time: 8点多, 8点半, 8点左右, 差5分8点
    - Time periods: 早上, 上午, 中午, 下午, 晚上, 凌晨, 夜里
    - Fuzzy time: 刚才, 现在, 马上, 立刻, 稍后
    
    Args:
        text: Text to process
        
    Returns:
        Text with time expressions removed
    """
    if not isinstance(text, str):
        text = str(text)
    
    # ===== 1. Standard time formats =====
    
    # 1.1 Colon-separated time: 8:40, 08:40:30, 23:59:59
    text = re.sub(r'\d{1,2}:\d{1,2}(:\d{1,2})?', '', text)
    
    # 1.2 Dot-separated time: 8.40, 8.40.30 (ensure it's not a decimal number)
    # Use lookahead/lookbehind assertions to avoid matching prices or decimals
    text = re.sub(r'(?<!\d)\d{1,2}\.\d{1,2}(\.\d{1,2})?(?!\d)', '', text)
    
    # 1.3 Chinese full time: 8点40分30秒, 8点40分, 8点40
    # Match: number+点+number+分+number+秒
    text = re.sub(r'\d+\s*点\s*\d+\s*分\s*\d+\s*秒', '', text)
    # Match: number+点+number+分
    text = re.sub(r'\d+\s*点\s*\d+\s*分', '', text)
    # Match: number+点+number (not followed by a unit)
    text = re.sub(r'\d+\s*点\s*\d+(?![分秒])', '', text)
    
    # 1.4 Chinese time units: 8点, 40分, 30秒, 8时, 40分钟
    text = re.sub(r'\d+\s*[点时]\s*(?:钟)?', '', text)
    text = re.sub(r'\d+\s*分\s*(?:钟)?', '', text)
    text = re.sub(r'\d+\s*秒\s*(?:钟)?', '', text)
    
    # 1.5 Chinese numeral time: 八点, 四十分, 三十秒
    chinese_num_pattern = r'[零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億两]+'
    text = re.sub(chinese_num_pattern + r'\s*[点时]\s*(?:钟)?', '', text)
    text = re.sub(chinese_num_pattern + r'\s*分\s*(?:钟)?', '', text)
    text = re.sub(chinese_num_pattern + r'\s*秒\s*(?:钟)?', '', text)
    
    # ===== 2. Colloquial time expressions =====
    
    # 2.1 Fuzzy time: 8点多, 8点半, 8点左右, 8点钟左右
    text = re.sub(r'\d+\s*[点时]\s*[多半来钟]', '', text)
    text = re.sub(r'\d+\s*[点时]\s*(?:左右|上下)', '', text)
    
    # 2.2 "minutes to" pattern: 差5分8点, 差一刻9点
    text = re.sub(r'差\s*\d+\s*分\s*\d+\s*[点时]', '', text)
    text = re.sub(r'差\s*一刻\s*\d+\s*[点时]', '', text)
    
    # 2.3 Quarter hour: 8点一刻, 9点三刻
    text = re.sub(r'\d+\s*[点时]\s*[一二三]刻', '', text)
    
    # ===== 3. Time period expressions =====
    
    # 3.1 Time periods: 早上, 上午, 中午, 下午, 晚上, 夜里, 凌晨, 深夜
    text = re.sub(r'[早上中下晚夜凌深][上午里晨间夜]', '', text)
    
    # 3.2 Full expressions with time period: 上午8点, 晚上9点半
    text = re.sub(r'[早上中下晚夜凌深][上午里晨间夜]\s*\d+\s*[点时]', '', text)
    
    # ===== 4. Fuzzy time words =====
    
    fuzzy_time_words = [
        '刚才', '刚刚', '现在', '此刻', '当前', '目前',
        '马上', '立刻', '立即', '立马', '即刻',
        '稍后', '待会', '一会儿', '过会', '等会',
        '随后', '之后', '然后', '接着', '紧接着'
    ]
    for word in fuzzy_time_words:
        text = text.replace(word, '')
    
    return text


def remove_date_expressions(text):
    """
    Remove date expressions from text (systematic processing)
    
    Supported date formats:
    - Standard dates: 2024年11月10日, 2024-11-10, 11/10, 11.10
    - Year/month/day: 2024年, 11月, 10号, 10日
    - Relative dates: 昨天, 今天, 明天, 前天, 后天
    - Periods: 第一天, 第二周, 上个月, 去年
    
    Args:
        text: Text to process
        
    Returns:
        Text with date expressions removed
    """
    if not isinstance(text, str):
        text = str(text)
    
    # ===== 1. Standard date formats =====
    
    # 1.1 Full date: 2024年11月10日, 2024-11-10, 2024/11/10, 2024.11.10
    text = re.sub(r'\d{4}[-/年.]\d{1,2}[-/月.]\d{1,2}[日号]?', '', text)
    
    # 1.2 Month-day format: 11月10日, 11-10, 11/10, 11.10
    text = re.sub(r'\d{1,2}[-/月.]\d{1,2}[日号]?', '', text)
    
    # 1.3 Standalone year/month/day: 2024年, 11月, 10号, 10日
    text = re.sub(r'\d+\s*[年]', '', text)
    text = re.sub(r'\d+\s*[月]', '', text)
    text = re.sub(r'\d+\s*[日号]', '', text)
    
    # ===== 2. Chinese numeral dates =====
    
    chinese_num_pattern = r'[零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億两]+'
    
    # 2.1 Chinese numerals + year/month/day
    text = re.sub(chinese_num_pattern + r'\s*[年]', '', text)
    text = re.sub(chinese_num_pattern + r'\s*[月]', '', text)
    text = re.sub(chinese_num_pattern + r'\s*[日号]', '', text)
    
    # ===== 3. Relative date expressions =====
    
    # 3.1 Yesterday, today, tomorrow, day before yesterday, day after tomorrow, etc.
    relative_days = ['昨天', '今天', '明天', '前天', '后天', '大前天', '大后天', '昨日', '今日', '明日']
    for day in relative_days:
        text = text.replace(day, '')
    
    # 3.2 Last week, this week, next week, last month, this month, next month, last year, this year, next year
    relative_periods = [
        '上周', '本周', '下周', '这周', '上星期', '本星期', '下星期', '这星期',
        '上月', '本月', '下月', '这月', '上个月', '这个月', '下个月',
        '去年', '今年', '明年', '前年', '后年'
    ]
    for period in relative_periods:
        text = text.replace(period, '')
    
    # ===== 4. Periodic expressions =====
    
    # 4.1 Nth day/week/month/year
    text = re.sub(r'第' + chinese_num_pattern + r'[天周月年]', '', text)
    text = re.sub(r'第\d+[天周月年]', '', text)
    
    # 4.2 N days/weeks/months/years ago/later
    text = re.sub(chinese_num_pattern + r'[天周月年][前后]', '', text)
    text = re.sub(r'\d+[天周月年][前后]', '', text)
    
    # 4.3 Day of week (星期X, 周X)
    text = re.sub(r'[星期周][一二三四五六七日天]', '', text)
    text = re.sub(r'礼拜[一二三四五六七日天]', '', text)
    
    return text


def remove_dates(text):
    """
    Remove date and time expressions from text (unified entry point)
    
    This is a compatibility function that internally calls more fine-grained
    processing functions. Consider using remove_date_expressions() and
    remove_time_expressions() directly.
    
    Args:
        text: Text to process
        
    Returns:
        Text with dates and times removed
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove time first (time may be embedded in dates, e.g. "2024年11月10日8点")
    text = remove_time_expressions(text)
    
    # Then remove dates
    text = remove_date_expressions(text)
    
    return text


# ===== Below are other existing preprocessing functions =====

def keep_chinese_only(text, keep_numbers=True, keep_chinese_punctuation=True):
    """
    Keep only Chinese characters (optionally keep numbers and Chinese punctuation)
    
    Args:
        text: Text to process
        keep_numbers: Whether to keep digits (0-9)
        keep_chinese_punctuation: Whether to keep Chinese punctuation marks
        
    Returns:
        Text containing only Chinese characters
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Define character ranges to keep
    result = []
    
    for char in text:
        # 1. Keep Chinese characters (CJK Unified Ideographs)
        if '\u4e00' <= char <= '\u9fff':
            result.append(char)
        # 2. Keep Chinese punctuation (if enabled)
        elif keep_chinese_punctuation and char in '，。！？；：""''（）【】《》、…—·':
            result.append(char)
        # 3. Keep digits (if enabled)
        elif keep_numbers and char.isdigit():
            result.append(char)
        # 4. Keep spaces (for separation)
        elif char == ' ':
            result.append(char)
    
    return ''.join(result)


def clean_text(text, remove_english=True, deduplicate_punctuation=True, 
               remove_html_entities=True, normalize_whitespace=True,
               remove_control_chars=True):
    """
    Main text preprocessing function
    
    Args:
        text: Text to process
        remove_english: Whether to remove English letters (keep digits)
        deduplicate_punctuation: Whether to remove consecutive duplicate punctuation
        remove_html_entities: Whether to remove HTML entities
        normalize_whitespace: Whether to normalize whitespace characters
        remove_control_chars: Whether to remove control characters
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Remove HTML entities (e.g. &hellip; -> ..., &nbsp; -> space)
    if remove_html_entities:
        text = html.unescape(text)
        # Further handle common HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&quot;', '"')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
    
    # 2. Remove all English letters (a-z, A-Z), keep digits
    if remove_english:
        text = re.sub(r'[a-zA-Z]+', '', text)
    
    # 3. Remove control characters (e.g. \x00-\x1f, but keep common whitespace)
    if remove_control_chars:
        # Keep common whitespace: space, newline, tab
        text = ''.join(char for char in text 
                      if unicodedata.category(char)[0] != 'C' 
                      or char in ['\n', '\r', '\t', ' '])
    
    # 4. Deduplicate consecutive punctuation (!!! -> !, ... -> .)
    if deduplicate_punctuation:
        # Match consecutive duplicate punctuation (both Chinese and English)
        text = re.sub(r'([!！？?。.，,、；;：:""\"\'\'（）()【】\[\]《》<>…~～@#￥$%^&*_+\-=｜|/\\])\1+', r'\1', text)
    
    # 5. Normalize whitespace (merge multiple spaces/newlines into one space)
    if normalize_whitespace:
        # Merge multiple consecutive whitespace characters into one space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading and trailing whitespace
        text = text.strip()
    
    return text


def preprocess_comment(comment, 
                       remove_english=True,
                       deduplicate_punctuation=True,
                       remove_html_entities=True,
                       normalize_whitespace=True,
                       remove_control_chars=True,
                       remove_dates_flag=True,
                       keep_chinese_only_flag=True,
                       keep_numbers=True,
                       keep_chinese_punctuation=True,
                       remove_whitespace_chars_flag=True,
                       max_length=None):
    """
    Comment preprocessing function (includes cleaning and length truncation)
    
    Args:
        comment: Comment text to process
        remove_english: Whether to remove English letters (effective when keep_chinese_only_flag=False)
        deduplicate_punctuation: Whether to remove consecutive duplicate punctuation
        remove_html_entities: Whether to remove HTML entities
        normalize_whitespace: Whether to normalize whitespace characters
        remove_control_chars: Whether to remove control characters
        remove_dates_flag: Whether to remove date expressions
        keep_chinese_only_flag: Whether to keep only Chinese (takes priority over remove_english)
        keep_numbers: Whether to keep digits (effective when keep_chinese_only_flag=True)
        keep_chinese_punctuation: Whether to keep Chinese punctuation (effective when keep_chinese_only_flag=True)
        remove_whitespace_chars_flag: Whether to remove newlines, tabs, and other whitespace characters
        max_length: Maximum length limit (None means no limit)
        
    Returns:
        Preprocessed comment text
    """
    # Ensure input is a string
    if not isinstance(comment, str):
        comment = str(comment)

    # 0. Remove newlines, tabs, and other whitespace characters (process first)
    if remove_whitespace_chars_flag:
        comment = remove_whitespace_chars(comment)
    
    
    # 1. Remove date expressions (process early)
    if remove_dates_flag:
        comment = remove_dates(comment)
    
    # 2. Keep only Chinese (if enabled, overrides other language processing options)
    if keep_chinese_only_flag:
        comment = keep_chinese_only(
            comment,
            keep_numbers=keep_numbers,
            keep_chinese_punctuation=keep_chinese_punctuation
        )
    else:
        # Use original cleaning logic
        comment = clean_text(
            comment,
            remove_english=remove_english,
            deduplicate_punctuation=deduplicate_punctuation,
            remove_html_entities=remove_html_entities,
            normalize_whitespace=normalize_whitespace,
            remove_control_chars=remove_control_chars
        )
    
    # 3. Deduplicate punctuation (if enabled and not using keep_chinese_only)
    if deduplicate_punctuation and not keep_chinese_only_flag:
        comment = re.sub(r'([!！？?。.，,、；;：:""\"\'\'（）()【】\[\]《》<>…~～@#￥$%^&*_+\-=｜|/\\])\1+', r'\1', comment)
    elif deduplicate_punctuation and keep_chinese_only_flag:
        # Only deduplicate Chinese punctuation
        comment = re.sub(r'([，。！？；：""''（）【】《》、…—·])\1+', r'\1', comment)
    
    # 4. Normalize whitespace
    if normalize_whitespace:
        comment = re.sub(r'\s+', ' ', comment)
        comment = comment.strip()
    
    # 5. Truncate overly long text (if max_length is specified)
    if max_length is not None and len(comment) > max_length:
        comment = comment[:max_length]
    
    return comment


# Additional helper functions

def remove_whitespace_chars(text):
    """
    Remove whitespace characters and newlines from text
    
    Characters removed:
    - \\n Line Feed
    - \\r Carriage Return
    - \\t Tab
    - \\v Vertical Tab
    - \\f Form Feed
    - \\r\\n Windows newline (CRLF)
    
    Args:
        text: Text to process
        
    Returns:
        Text with whitespace characters removed
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Define whitespace characters to remove
    whitespace_chars = [
        '\r\n',  # Windows newline (CRLF), process first to avoid splitting
        '\n',    # Line Feed
        '\r',    # Carriage Return
        '\t',    # Tab
        '\v',    # Vertical Tab
        '\f',    # Form Feed
    ]
    
    for char in whitespace_chars:
        text = text.replace(char, '')
    
    return text


def remove_urls(text):
    """Remove URL links from text"""
    # Match URLs starting with http/https/ftp
    text = re.sub(r'https?://\S+|ftp://\S+', '', text)
    # Match URLs starting with www
    text = re.sub(r'www\.\S+', '', text)
    return text


def remove_emails(text):
    """Remove email addresses from text"""
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    return text


def remove_phone_numbers(text):
    """Remove phone numbers from text"""
    # Match Chinese mobile numbers (11 digits)
    text = re.sub(r'1[3-9]\d{9}', '', text)
    return text


def normalize_numbers(text):
    """Normalize number representation (convert full-width digits to half-width)"""
    # Full-width to half-width digits
    full_width = '０１２３４５６７８９'
    half_width = '0123456789'
    trans_table = str.maketrans(full_width, half_width)
    return text.translate(trans_table)


def remove_emojis(text):
    """
    Remove emoji characters from text
    
    Includes:
    - Standard emojis (😀😁😂 etc.)
    - Emoticon symbols (☺️♥️ etc.)
    - Special symbols (🔥⭐ etc.)
    
    Args:
        text: Text to process
        
    Returns:
        Text with emojis removed
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Emoji Unicode ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Extended Symbols A
        "\U0001FA70-\U0001FAFF"  # Extended Symbols B
        "\U0001F000-\U0001F02F"  # Mahjong Tiles
        "\U0001F0A0-\U0001F0FF"  # Playing Cards
        "]+",
        flags=re.UNICODE
    )
    
    return emoji_pattern.sub(r'', text)


def remove_garbled_text(text):
    """
    Remove garbled/corrupted characters
    
    Includes:
    - Invisible characters (zero-width characters, etc.)
    - Special control characters
    - Abnormal Unicode characters
    - Common garbled patterns (e.g.: 锟斤拷, 烫烫烫, etc.)
    
    Args:
        text: Text to process
        
    Returns:
        Text with garbled characters removed
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Remove zero-width characters
    zero_width_chars = [
        '\u200b',  # Zero Width Space
        '\u200c',  # Zero Width Non-Joiner
        '\u200d',  # Zero Width Joiner
        '\ufeff',  # Zero Width No-Break Space / BOM
        '\u2060',  # Word Joiner
    ]
    for char in zero_width_chars:
        text = text.replace(char, '')
    
    # 2. Remove common garbled patterns
    garbled_patterns = [
        '锟斤拷',  # UTF-8 encoding issue
        '烫烫烫',  # Uninitialized memory display
        '屯屯屯',  # Similar garbled pattern
        '�',      # Unicode Replacement Character
    ]
    for pattern in garbled_patterns:
        text = text.replace(pattern, '')
    
    # 3. Remove special format characters (Variation Selectors)
    text = re.sub(r'[\uFE00-\uFE0F]', '', text)  # Variation Selectors
    text = re.sub(r'[\U000E0100-\U000E01EF]', '', text)  # Variation Selectors Supplement
    
    # 4. Remove other invisible or special Unicode characters
    text = re.sub(r'[\u200e\u200f]', '', text)  # Left-to-Right/Right-to-Left Mark
    text = re.sub(r'[\u202a-\u202e]', '', text)  # Directional Formatting
    
    # 5. Remove Private Use Area characters (may display as garbled text)
    text = re.sub(r'[\ue000-\uf8ff]', '', text)  # Private Use Area
    text = re.sub(r'[\U000F0000-\U000FFFFD]', '', text)  # Supplementary Private Use Area-A
    text = re.sub(r'[\U00100000-\U0010FFFD]', '', text)  # Supplementary Private Use Area-B
    
    return text


def remove_special_symbols(text):
    """
    Remove special symbols (keep common punctuation)
    
    Removes:
    - Mathematical symbols (±×÷≈ etc.)
    - Currency symbols (€£¥$ etc., except commonly used ￥)
    - Arrow symbols (→←↑↓ etc.)
    - Geometric shapes (■□●○ etc.)
    - Musical symbols (♪♫ etc.)
    
    Args:
        text: Text to process
        
    Returns:
        Text with special symbols removed
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Define special symbol ranges to remove
    special_symbols_pattern = re.compile(
        "["
        "\u2190-\u21FF"  # Arrows
        "\u2200-\u22FF"  # Mathematical Operators
        "\u2300-\u23FF"  # Miscellaneous Technical
        "\u2500-\u257F"  # Box Drawing
        "\u2580-\u259F"  # Block Elements
        "\u25A0-\u25FF"  # Geometric Shapes
        "\u2600-\u26FF"  # Miscellaneous Symbols (includes some emoji)
        "\u2700-\u27BF"  # Dingbats
        "\u2B00-\u2BFF"  # Miscellaneous Symbols and Arrows
        "\u20A0-\u20CF"  # Currency Symbols (but keep ￥)
        "]+",
        flags=re.UNICODE
    )
    
    text = special_symbols_pattern.sub(r'', text)
    
    # Additional: remove some common special symbols not in the ranges above
    text = text.replace('™', '')  # Trademark
    text = text.replace('®', '')  # Registered Trademark
    text = text.replace('©', '')  # Copyright
    text = text.replace('§', '')  # Section Sign
    text = text.replace('¶', '')  # Pilcrow Sign
    text = text.replace('†', '')  # Dagger
    text = text.replace('‡', '')  # Double Dagger
    text = text.replace('•', '')  # Bullet
    text = text.replace('◦', '')  # White Bullet
    text = text.replace('‣', '')  # Triangular Bullet
    
    return text


def advanced_preprocess(text, 
                       remove_urls_flag=True,
                       remove_emails_flag=True,
                       remove_phones_flag=True,
                       normalize_numbers_flag=True,
                       remove_emojis_flag=True,
                       remove_garbled_flag=True,
                       remove_special_symbols_flag=True):
    """
    Advanced preprocessing (optional features)
    
    Args:
        text: Text to process
        remove_urls_flag: Whether to remove URLs
        remove_emails_flag: Whether to remove email addresses
        remove_phones_flag: Whether to remove phone numbers
        normalize_numbers_flag: Whether to normalize digits
        remove_emojis_flag: Whether to remove emoji characters
        remove_garbled_flag: Whether to remove garbled characters
        remove_special_symbols_flag: Whether to remove special symbols
        
    Returns:
        Processed text
    """
    # Prioritize removing garbled and invisible characters
    if remove_garbled_flag:
        text = remove_garbled_text(text)
    
    if remove_urls_flag:
        text = remove_urls(text)
    
    if remove_emails_flag:
        text = remove_emails(text)
    
    if remove_phones_flag:
        text = remove_phone_numbers(text)
    
    if remove_emojis_flag:
        text = remove_emojis(text)
    
    if remove_special_symbols_flag:
        text = remove_special_symbols(text)
    
    if normalize_numbers_flag:
        text = normalize_numbers(text)
    
    return text
