import re
import html
import unicodedata


def remove_time_expressions(text):
    """
    去除文本中的时间表达式（系统化处理）
    
    支持的时间格式：
    - 标准时间：8:40、08:40:30、8.40、8点40分30秒
    - 口语时间：8点多、8点半、8点左右、差5分8点
    - 时间段：早上、上午、中午、下午、晚上、凌晨、夜里
    - 模糊时间：刚才、现在、马上、立刻、稍后
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除时间表达后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # ===== 1. 标准时间格式 =====
    
    # 1.1 冒号分隔的时间：8:40、08:40:30、23:59:59
    text = re.sub(r'\d{1,2}:\d{1,2}(:\d{1,2})?', '', text)
    
    # 1.2 点号分隔的时间：8.40、8.40.30（需要确保不是小数）
    # 使用前瞻/后顾断言确保不是价格等小数
    text = re.sub(r'(?<!\d)\d{1,2}\.\d{1,2}(\.\d{1,2})?(?!\d)', '', text)
    
    # 1.3 中文完整时间：8点40分30秒、8点40分、8点40
    # 匹配：数字+点+数字+分+数字+秒
    text = re.sub(r'\d+\s*点\s*\d+\s*分\s*\d+\s*秒', '', text)
    # 匹配：数字+点+数字+分
    text = re.sub(r'\d+\s*点\s*\d+\s*分', '', text)
    # 匹配：数字+点+数字（后面没有单位）
    text = re.sub(r'\d+\s*点\s*\d+(?![分秒])', '', text)
    
    # 1.4 中文时间：8点、40分、30秒、8时、40分钟
    text = re.sub(r'\d+\s*[点时]\s*(?:钟)?', '', text)
    text = re.sub(r'\d+\s*分\s*(?:钟)?', '', text)
    text = re.sub(r'\d+\s*秒\s*(?:钟)?', '', text)
    
    # 1.5 中文数字时间：八点、四十分、三十秒
    chinese_num_pattern = r'[零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億两]+'
    text = re.sub(chinese_num_pattern + r'\s*[点时]\s*(?:钟)?', '', text)
    text = re.sub(chinese_num_pattern + r'\s*分\s*(?:钟)?', '', text)
    text = re.sub(chinese_num_pattern + r'\s*秒\s*(?:钟)?', '', text)
    
    # ===== 2. 口语化时间表达 =====
    
    # 2.1 模糊时间：8点多、8点半、8点左右、8点钟左右
    text = re.sub(r'\d+\s*[点时]\s*[多半来钟]', '', text)
    text = re.sub(r'\d+\s*[点时]\s*(?:左右|上下)', '', text)
    
    # 2.2 差几分几点：差5分8点、差一刻9点
    text = re.sub(r'差\s*\d+\s*分\s*\d+\s*[点时]', '', text)
    text = re.sub(r'差\s*一刻\s*\d+\s*[点时]', '', text)
    
    # 2.3 几点几刻：8点一刻、9点三刻
    text = re.sub(r'\d+\s*[点时]\s*[一二三]刻', '', text)
    
    # ===== 3. 时间段表达 =====
    
    # 3.1 时间段：早上、上午、中午、下午、晚上、夜里、凌晨、深夜
    text = re.sub(r'[早上中下晚夜凌深][上午里晨间夜]', '', text)
    
    # 3.2 带时间段的完整表达：上午8点、晚上9点半
    text = re.sub(r'[早上中下晚夜凌深][上午里晨间夜]\s*\d+\s*[点时]', '', text)
    
    # ===== 4. 模糊时间词 =====
    
    # 刚才、现在、马上、立刻、稍后、待会、一会儿
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
    去除文本中的日期表达式（系统化处理）
    
    支持的日期格式：
    - 标准日期：2024年11月10日、2024-11-10、11/10、11.10
    - 年月日：2024年、11月、10号、10日
    - 相对日期：昨天、今天、明天、前天、后天
    - 周期：第一天、第二周、上个月、去年
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除日期表达后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # ===== 1. 标准日期格式 =====
    
    # 1.1 完整日期：2024年11月10日、2024-11-10、2024/11/10、2024.11.10
    text = re.sub(r'\d{4}[-/年.]\d{1,2}[-/月.]\d{1,2}[日号]?', '', text)
    
    # 1.2 月日格式：11月10日、11-10、11/10、11.10
    text = re.sub(r'\d{1,2}[-/月.]\d{1,2}[日号]?', '', text)
    
    # 1.3 单独的年月日号：2024年、11月、10号、10日
    text = re.sub(r'\d+\s*[年]', '', text)
    text = re.sub(r'\d+\s*[月]', '', text)
    text = re.sub(r'\d+\s*[日号]', '', text)
    
    # ===== 2. 中文数字日期 =====
    
    chinese_num_pattern = r'[零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億两]+'
    
    # 2.1 中文数字+年月日号
    text = re.sub(chinese_num_pattern + r'\s*[年]', '', text)
    text = re.sub(chinese_num_pattern + r'\s*[月]', '', text)
    text = re.sub(chinese_num_pattern + r'\s*[日号]', '', text)
    
    # ===== 3. 相对日期表达 =====
    
    # 3.1 昨天、今天、明天、前天、后天、大前天、大后天
    relative_days = ['昨天', '今天', '明天', '前天', '后天', '大前天', '大后天', '昨日', '今日', '明日']
    for day in relative_days:
        text = text.replace(day, '')
    
    # 3.2 上周、本周、下周、上月、本月、下月、去年、今年、明年
    relative_periods = [
        '上周', '本周', '下周', '这周', '上星期', '本星期', '下星期', '这星期',
        '上月', '本月', '下月', '这月', '上个月', '这个月', '下个月',
        '去年', '今年', '明年', '前年', '后年'
    ]
    for period in relative_periods:
        text = text.replace(period, '')
    
    # ===== 4. 周期表达 =====
    
    # 4.1 第X天/周/月/年
    text = re.sub(r'第' + chinese_num_pattern + r'[天周月年]', '', text)
    text = re.sub(r'第\d+[天周月年]', '', text)
    
    # 4.2 X天/周/月/年前/后
    text = re.sub(chinese_num_pattern + r'[天周月年][前后]', '', text)
    text = re.sub(r'\d+[天周月年][前后]', '', text)
    
    # 4.3 星期X、周X
    text = re.sub(r'[星期周][一二三四五六七日天]', '', text)
    text = re.sub(r'礼拜[一二三四五六七日天]', '', text)
    
    return text


def remove_dates(text):
    """
    去除文本中的日期和时间表达式（统一入口）
    
    这是一个兼容性函数，内部调用更细粒度的处理函数。
    建议直接使用 remove_date_expressions() 和 remove_time_expressions()。
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除日期和时间后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 先去除时间（因为时间可能包含在日期中，如"2024年11月10日8点"）
    text = remove_time_expressions(text)
    
    # 再去除日期
    text = remove_date_expressions(text)
    
    return text


# ===== 以下是原有的其他预处理函数（保持不变）=====

def keep_chinese_only(text, keep_numbers=True, keep_chinese_punctuation=True):
    """
    只保留中文字符（可选保留数字和中文标点）
    
    Args:
        text: 待处理的文本
        keep_numbers: 是否保留数字（0-9）
        keep_chinese_punctuation: 是否保留中文标点符号
        
    Returns:
        只包含中文的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 定义要保留的字符范围
    result = []
    
    for char in text:
        # 1. 保留中文汉字（CJK统一表意文字）
        if '\u4e00' <= char <= '\u9fff':  # 基本汉字
            result.append(char)
        # 2. 保留中文标点符号（如果启用）
        elif keep_chinese_punctuation and char in '，。！？；：""''（）【】《》、…—·':
            result.append(char)
        # 3. 保留数字（如果启用）
        elif keep_numbers and char.isdigit():
            result.append(char)
        # 4. 保留空格（用于分隔）
        elif char == ' ':
            result.append(char)
    
    return ''.join(result)


def clean_text(text, remove_english=True, deduplicate_punctuation=True, 
               remove_html_entities=True, normalize_whitespace=True,
               remove_control_chars=True):
    """
    文本预处理主函数
    
    Args:
        text: 待处理的文本
        remove_english: 是否去除英文字母（保留数字）
        deduplicate_punctuation: 是否去除连续重复的标点符号
        remove_html_entities: 是否去除 HTML 实体
        normalize_whitespace: 是否规范化空白字符
        remove_control_chars: 是否去除控制字符
        
    Returns:
        清洗后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. 去除 HTML 实体（如 &hellip; → ...，&nbsp; → 空格）
    if remove_html_entities:
        text = html.unescape(text)
        # 进一步处理常见的 HTML 实体
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&quot;', '"')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
    
    # 2. 去除所有英文字母（a-z, A-Z），保留数字
    if remove_english:
        text = re.sub(r'[a-zA-Z]+', '', text)
    
    # 3. 去除控制字符（如 \x00-\x1f，但保留常用的换行、制表符）
    if remove_control_chars:
        # 保留常用空白字符：空格、换行、制表符
        text = ''.join(char for char in text 
                      if unicodedata.category(char)[0] != 'C' 
                      or char in ['\n', '\r', '\t', ' '])
    
    # 4. 去除连续重复的标点符号（!!! → !，... → .）
    if deduplicate_punctuation:
        # 匹配连续重复的标点符号（包括中英文标点）
        # 标点符号类别：P (Punctuation)
        text = re.sub(r'([!！？?。.，,、；;：:""\"\'\'（）()【】\[\]《》<>…~～@#￥$%^&*_+\-=｜|/\\])\1+', r'\1', text)
    
    # 5. 规范化空白字符（多个空格/换行合并为一个空格）
    if normalize_whitespace:
        # 将多个连续的空白字符（空格、换行、制表符等）合并为一个空格
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空白
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
    评论预处理函数（包含清洗和长度截断）
    
    Args:
        comment: 待处理的评论文本
        remove_english: 是否去除英文字母（当 keep_chinese_only_flag=False 时生效）
        deduplicate_punctuation: 是否去除连续重复的标点符号
        remove_html_entities: 是否去除 HTML 实体
        normalize_whitespace: 是否规范化空白字符
        remove_control_chars: 是否去除控制字符
        remove_dates_flag: 是否去除日期表达
        keep_chinese_only_flag: 是否只保留中文（优先级高于 remove_english）
        keep_numbers: 是否保留数字（当 keep_chinese_only_flag=True 时生效）
        keep_chinese_punctuation: 是否保留中文标点（当 keep_chinese_only_flag=True 时生效）
        remove_whitespace_chars_flag: 是否去除换行符、制表符等空白字符
        max_length: 最大长度限制（None 表示不限制）
        
    Returns:
        预处理后的评论文本
    """
    # 确保输入是字符串
    if not isinstance(comment, str):
        comment = str(comment)

    # 0. 去除换行符、制表符等空白字符（最先处理）
    if remove_whitespace_chars_flag:
        comment = remove_whitespace_chars(comment)
    
    
    # 1. 去除日期表达（优先处理）
    if remove_dates_flag:
        comment = remove_dates(comment)
    
    # 2. 只保留中文（如果启用，会覆盖其他语言处理选项）
    if keep_chinese_only_flag:
        comment = keep_chinese_only(
            comment,
            keep_numbers=keep_numbers,
            keep_chinese_punctuation=keep_chinese_punctuation
        )
    else:
        # 使用原有的清洗逻辑
        comment = clean_text(
            comment,
            remove_english=remove_english,
            deduplicate_punctuation=deduplicate_punctuation,
            remove_html_entities=remove_html_entities,
            normalize_whitespace=normalize_whitespace,
            remove_control_chars=remove_control_chars
        )
    
    # 3. 去除重复标点（如果启用且未使用 keep_chinese_only）
    if deduplicate_punctuation and not keep_chinese_only_flag:
        comment = re.sub(r'([!！？?。.，,、；;：:""\"\'\'（）()【】\[\]《》<>…~～@#￥$%^&*_+\-=｜|/\\])\1+', r'\1', comment)
    elif deduplicate_punctuation and keep_chinese_only_flag:
        # 只处理中文标点的重复
        comment = re.sub(r'([，。！？；：""''（）【】《》、…—·])\1+', r'\1', comment)
    
    # 4. 规范化空白字符
    if normalize_whitespace:
        comment = re.sub(r'\s+', ' ', comment)
        comment = comment.strip()
    
    # 5. 截断过长文本（如果指定了最大长度）
    if max_length is not None and len(comment) > max_length:
        comment = comment[:max_length]
    
    return comment


# 一些额外的辅助函数（保持不变）

def remove_whitespace_chars(text):
    """
    去除文本中的空白字符和换行符
    
    去除的字符包括：
    - \n 换行符 (Line Feed)
    - \r 回车符 (Carriage Return)
    - \t 制表符 (Tab)
    - \v 垂直制表符 (Vertical Tab)
    - \f 换页符 (Form Feed)
    - \r\n Windows换行 (CRLF)
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除空白字符后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 定义需要去除的空白字符
    whitespace_chars = [
        '\r\n',  # Windows换行（CRLF），需要先处理，避免被拆分
        '\n',    # 换行符 (Line Feed)
        '\r',    # 回车符 (Carriage Return)
        '\t',    # 制表符 (Tab)
        '\v',    # 垂直制表符 (Vertical Tab)
        '\f',    # 换页符 (Form Feed)
    ]
    
    for char in whitespace_chars:
        text = text.replace(char, '')
    
    return text


def remove_urls(text):
    """去除文本中的 URL 链接"""
    # 匹配 http/https/ftp 开头的 URL
    text = re.sub(r'https?://\S+|ftp://\S+', '', text)
    # 匹配 www 开头的 URL
    text = re.sub(r'www\.\S+', '', text)
    return text


def remove_emails(text):
    """去除文本中的邮箱地址"""
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    return text


def remove_phone_numbers(text):
    """去除文本中的手机号码"""
    # 匹配中国手机号（11位数字）
    text = re.sub(r'1[3-9]\d{9}', '', text)
    return text


def normalize_numbers(text):
    """规范化数字表示（将全角数字转为半角）"""
    # 全角数字转半角
    full_width = '０１２３４５６７８９'
    half_width = '0123456789'
    trans_table = str.maketrans(full_width, half_width)
    return text.translate(trans_table)


def remove_emojis(text):
    """
    去除文本中的emoji表情符号
    
    包括：
    - 标准emoji (😀😁😂等)
    - 表情符号 (☺️♥️等)
    - 特殊符号 (🔥⭐等)
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除emoji后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Emoji Unicode 范围
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号 (Emoticons)
        "\U0001F300-\U0001F5FF"  # 符号和象形文字 (Symbols & Pictographs)
        "\U0001F680-\U0001F6FF"  # 交通和地图符号 (Transport & Map Symbols)
        "\U0001F1E0-\U0001F1FF"  # 旗帜 (Flags)
        "\U0001F900-\U0001F9FF"  # 补充符号和象形文字 (Supplemental Symbols and Pictographs)
        "\U0001FA00-\U0001FA6F"  # 扩展符号A (Extended Symbols A)
        "\U0001FA70-\U0001FAFF"  # 扩展符号B (Extended Symbols B)
        "\U0001F000-\U0001F02F"  # 麻将牌 (Mahjong Tiles)
        "\U0001F0A0-\U0001F0FF"  # 扑克牌 (Playing Cards)
        "]+",
        flags=re.UNICODE
    )
    
    return emoji_pattern.sub(r'', text)


def remove_garbled_text(text):
    """
    去除乱码字符
    
    包括：
    - 不可见字符（零宽字符等）
    - 特殊控制字符
    - 异常Unicode字符
    - 常见乱码模式（如：锟斤拷、烫烫烫等）
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除乱码后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. 去除零宽字符（Zero-Width Characters）
    # 零宽空格、零宽连接符、零宽非连接符等
    zero_width_chars = [
        '\u200b',  # 零宽空格 (Zero Width Space)
        '\u200c',  # 零宽非连接符 (Zero Width Non-Joiner)
        '\u200d',  # 零宽连接符 (Zero Width Joiner)
        '\ufeff',  # 零宽非断空格 (Zero Width No-Break Space / BOM)
        '\u2060',  # 字连接符 (Word Joiner)
    ]
    for char in zero_width_chars:
        text = text.replace(char, '')
    
    # 2. 去除常见乱码模式
    garbled_patterns = [
        '锟斤拷',  # UTF-8编码问题导致的乱码
        '烫烫烫',  # 未初始化内存显示
        '屯屯屯',  # 类似乱码
        '�',      # Unicode替换字符 (Replacement Character)
    ]
    for pattern in garbled_patterns:
        text = text.replace(pattern, '')
    
    # 3. 去除特殊格式字符（Variation Selectors）
    # 用于改变前面字符的显示方式
    text = re.sub(r'[\uFE00-\uFE0F]', '', text)  # Variation Selectors
    text = re.sub(r'[\U000E0100-\U000E01EF]', '', text)  # Variation Selectors Supplement
    
    # 4. 去除其他不可见或特殊Unicode字符
    # 格式控制字符
    text = re.sub(r'[\u200e\u200f]', '', text)  # Left-to-Right/Right-to-Left Mark
    text = re.sub(r'[\u202a-\u202e]', '', text)  # Directional Formatting
    
    # 5. 去除私有使用区字符（Private Use Area）
    # 这些字符通常用于自定义符号，可能显示为乱码
    text = re.sub(r'[\ue000-\uf8ff]', '', text)  # Private Use Area
    text = re.sub(r'[\U000F0000-\U000FFFFD]', '', text)  # Supplementary Private Use Area-A
    text = re.sub(r'[\U00100000-\U0010FFFD]', '', text)  # Supplementary Private Use Area-B
    
    return text


def remove_special_symbols(text):
    """
    去除特殊符号（保留常用标点）
    
    去除：
    - 数学符号 (±×÷≈等)
    - 货币符号 (€£¥$等，除了常用的￥)
    - 箭头符号 (→←↑↓等)
    - 几何图形 (■□●○等)
    - 音乐符号 (♪♫等)
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除特殊符号后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 定义要去除的特殊符号范围
    special_symbols_pattern = re.compile(
        "["
        "\u2190-\u21FF"  # 箭头 (Arrows)
        "\u2200-\u22FF"  # 数学运算符 (Mathematical Operators)
        "\u2300-\u23FF"  # 杂项技术符号 (Miscellaneous Technical)
        "\u2500-\u257F"  # 制表符 (Box Drawing)
        "\u2580-\u259F"  # 方块元素 (Block Elements)
        "\u25A0-\u25FF"  # 几何图形 (Geometric Shapes)
        "\u2600-\u26FF"  # 杂项符号 (Miscellaneous Symbols) - 包含部分emoji
        "\u2700-\u27BF"  # 装饰符号 (Dingbats)
        "\u2B00-\u2BFF"  # 杂项符号和箭头 (Miscellaneous Symbols and Arrows)
        "\u20A0-\u20CF"  # 货币符号 (Currency Symbols) - 但保留￥
        "]+",
        flags=re.UNICODE
    )
    
    text = special_symbols_pattern.sub(r'', text)
    
    # 额外处理：去除一些常见的特殊符号（不在上述范围内的）
    text = text.replace('™', '')  # 商标
    text = text.replace('®', '')  # 注册商标
    text = text.replace('©', '')  # 版权
    text = text.replace('§', '')  # 章节符号
    text = text.replace('¶', '')  # 段落符号
    text = text.replace('†', '')  # 剑标
    text = text.replace('‡', '')  # 双剑标
    text = text.replace('•', '')  # 项目符号
    text = text.replace('◦', '')  # 空心项目符号
    text = text.replace('‣', '')  # 三角项目符号
    
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
    高级预处理（可选功能）
    
    Args:
        text: 待处理的文本
        remove_urls_flag: 是否去除 URL
        remove_emails_flag: 是否去除邮箱
        remove_phones_flag: 是否去除手机号
        normalize_numbers_flag: 是否规范化数字
        remove_emojis_flag: 是否去除emoji表情
        remove_garbled_flag: 是否去除乱码字符
        remove_special_symbols_flag: 是否去除特殊符号
        
    Returns:
        处理后的文本
    """
    # 优先去除乱码和不可见字符
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

