# 关键词后处理模块
# 用于对模型返回的关键词进行去重、排序、过滤等操作

import os
import re
import logging


# 停用词缓存字典：{文件路径: 停用词集合}
_stopwords_cache = {}


def find_min_span_in_text(keyword, original_text):
    """
    在原始文本中找到能匹配关键词所有字符的最小连续范围
    
    使用贪婪算法：从关键词第一个字符在原文中的每个出现位置开始，
    依次向后查找剩余字符，计算匹配所需的最小范围。
    
    Args:
        keyword: 待匹配的关键词
        original_text: 原始文本
        
    Returns:
        最小匹配范围的长度，如果无法匹配则返回 -1
    """
    if not keyword or not original_text:
        return -1
    
    # 过滤掉关键词中的空格
    keyword_chars = [c for c in keyword if c != ' ']
    if not keyword_chars:
        return -1
    
    min_span = float('inf')
    
    # 找到第一个字符在原文中的所有位置
    first_char = keyword_chars[0]
    start_positions = [i for i, c in enumerate(original_text) if c == first_char]
    
    # 对于每个起始位置，尝试贪婪匹配
    for start_pos in start_positions:
        current_pos = start_pos
        matched = True
        
        # 依次匹配剩余字符
        for i, char in enumerate(keyword_chars):
            if i == 0:
                continue  # 第一个字符已经匹配
            
            # 从当前位置向后查找下一个字符
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
            # 计算这次匹配的范围（从起始位置到最后匹配位置）
            span = current_pos - start_pos + 1
            min_span = min(min_span, span)
    
    return min_span if min_span != float('inf') else -1


def validate_keyword_chars_in_text(keyword, original_text, max_span_ratio=2):
    """
    验证关键词中的每个字符是否都在原始文本中紧凑地存在
    
    该函数用于过滤掉模型"自行推理总结"产生的、原文中不存在的关键词。
    
    验证规则：
    1. 关键词的每个字符都必须在原文中存在
    2. 这些字符在原文中的最小匹配范围不能超过关键词长度的 max_span_ratio 倍
    
    例如（假设 max_span_ratio=2）：
    - 原文"已经退货"，关键词"没退货" → 无效（"没"不在原文中）
    - 原文"衣服不怎么粘肉"，关键词"不粘肉" → 有效（范围5，关键词长度3，比例1.67<2）
    - 原文"聊个不停...一天"，关键词"聊天" → 无效（字符散落太远，范围远超2倍）
    
    Args:
        keyword: 待验证的关键词
        original_text: 原始文本（建议传入预处理后的文本）
        max_span_ratio: 最大跨度倍数（默认为2，即匹配范围不超过关键词长度的2倍）
        
    Returns:
        True 如果关键词验证通过，否则 False
    """
    if not isinstance(keyword, str) or not isinstance(original_text, str):
        return False
    
    if not keyword or not original_text:
        return False
    
    # 过滤掉关键词中的空格
    keyword_chars = [c for c in keyword if c != ' ']
    if not keyword_chars:
        return False
    
    keyword_len = len(keyword_chars)
    
    # 1. 首先检查每个字符是否都在原文中存在
    original_chars = set(original_text)
    for char in keyword_chars:
        if char not in original_chars:
            return False
    
    # 2. 检查字符在原文中的紧凑性（最小匹配范围）
    min_span = find_min_span_in_text(keyword, original_text)
    
    if min_span < 0:
        # 无法找到匹配，说明字符顺序不对或不存在
        return False
    
    # 计算允许的最大范围
    max_allowed_span = keyword_len * max_span_ratio
    
    if min_span > max_allowed_span:
        logging.debug(f"[原文验证] 关键词'{keyword}'跨度过大: 最小范围={min_span}, 允许最大={max_allowed_span}")
        return False
    
    return True


def filter_keywords_not_in_original(keywords_data, original_text, keyword_idx=1, max_span_ratio=2):
    """
    过滤掉不在原始文本中的关键词
    
    遍历关键词列表，检查每个关键词的所有字符是否都在原始文本中紧凑地存在，
    过滤掉包含原文中不存在字符的关键词，以及字符散落过远的拼凑关键词。
    
    Args:
        keywords_data: 关键词数据列表，格式为 [[推理, 关键词, 分数], ...]
        original_text: 原始文本（建议传入预处理后的文本）
        keyword_idx: 关键词在列表中的索引位置（新格式为1，旧格式为0）
        max_span_ratio: 最大跨度倍数（默认为2，即匹配范围不超过关键词长度的2倍）
        
    Returns:
        过滤后的关键词数据列表
    """
    if not keywords_data or not original_text:
        return keywords_data
    
    filtered_data = []
    filtered_out = []  # 记录被过滤的关键词（用于调试）
    
    for item in keywords_data:
        if len(item) > keyword_idx:
            keyword = item[keyword_idx]
            
            # 确保关键词是字符串类型
            if not isinstance(keyword, str):
                continue
            
            # 验证关键词中的每个字符是否都在原文中紧凑地存在
            if validate_keyword_chars_in_text(keyword, original_text, max_span_ratio):
                filtered_data.append(item)
            else:
                filtered_out.append(keyword)
    
    # 调试输出
    if filtered_out:
        logging.info(f"[原文验证过滤] 过滤前: {len(keywords_data)} 个, 过滤后: {len(filtered_data)} 个, 被过滤: {filtered_out}")
    
    return filtered_data



# ==================== 通用中文数字模式 ====================
# 包含：零一二三四五六七八九十百千万亿（简体）
#       壹贰叁肆伍陆柒捌玖拾佰仟萬億（繁体/大写）
#       〇两（特殊数字）
CHINESE_NUM_PATTERN = r'[零〇一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億两]+'
# 阿拉伯数字模式
ARABIC_NUM_PATTERN = r'\d+'
# 任意数字模式（中文或阿拉伯数字）
ANY_NUM_PATTERN = f'(?:{CHINESE_NUM_PATTERN}|{ARABIC_NUM_PATTERN})'


def is_date_keyword(keyword):
    """
    判断关键词是否为日期相关词汇
    
    检测规则：
    - 年份：2024年、二零二四年、二千零二十四年
    - 月份：1月、一月、01月
    - 日期：1号、一号、01号、1日、一日
    - 完整日期：5月15、五月十五号、2024年11月
    - 星期：周一、星期一、礼拜一
    - 相对日期：今天、昨天、明天、前天、后天
    
    Args:
        keyword: 待检测的关键词
        
    Returns:
        True 如果是日期关键词，否则 False
    """
    if not isinstance(keyword, str):
        return False
    
    # ===== 1. 年份检测 =====
    
    # 1.1 数字年份：2024年、24年、任意数字+年
    if re.search(ARABIC_NUM_PATTERN + r'\s*年', keyword):
        return True
    
    # 1.2 中文年份：二零二四年、二〇二四年、二千零二十四年
    if re.search(CHINESE_NUM_PATTERN + r'\s*年', keyword):
        return True
    
    # ===== 2. 月份检测 =====
    
    # 2.1 数字月份：1月、01月、12月
    if re.search(ARABIC_NUM_PATTERN + r'\s*月', keyword):
        return True
    
    # 2.2 中文月份：一月、十二月、任意中文数字+月
    if re.search(CHINESE_NUM_PATTERN + r'\s*月', keyword):
        return True
    
    # ===== 3. 日期检测 =====
    
    # 3.1 数字日期：1号、01号、1日、01日
    if re.search(ARABIC_NUM_PATTERN + r'\s*[号日]', keyword):
        return True
    
    # 3.2 中文日期：一号、二十七号、三十一日
    if re.search(CHINESE_NUM_PATTERN + r'\s*[号日]', keyword):
        return True
    
    # 3.3 只有"号"字（在特定上下文中）
    if keyword == '号' or keyword == '日':
        return True
    
    # ===== 4. 完整日期格式 =====
    
    # 4.1 月日组合：5月15、11.15、11-15、11/15
    if re.search(ARABIC_NUM_PATTERN + r'\s*月\s*' + ARABIC_NUM_PATTERN, keyword):
        return True
    if re.search(r'\d{1,2}[\./-]\d{1,2}', keyword):
        return True
    
    # 4.2 中文月日组合：五月十五
    if re.search(CHINESE_NUM_PATTERN + r'\s*月\s*' + CHINESE_NUM_PATTERN, keyword):
        return True
    
    # 4.3 年月日组合：2024年11月15日
    if re.search(ANY_NUM_PATTERN + r'\s*年\s*' + ANY_NUM_PATTERN + r'\s*月\s*' + ANY_NUM_PATTERN + r'\s*[日号]?', keyword):
        return True
    
    # ===== 5. 星期检测 =====
    
    # 5.1 星期：周X、星期X、礼拜X（使用通用模式）
    if re.search(r'周[一二三四五六日天]', keyword):
        return True
    if re.search(r'星期[一二三四五六日天]', keyword):
        return True
    if re.search(r'礼拜[一二三四五六日天]', keyword):
        return True
    
    # 5.2 工作日、周末
    if '工作日' in keyword or '周末' in keyword or '双休' in keyword:
        return True
    
    # ===== 6. 相对日期 =====
    
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
    
    # ===== 7. 其他日期表达 =====
    
    # 7.1 月初、月中、月末、月底
    if re.search(r'月[初中末底]', keyword):
        return True
    
    # 7.2 上旬、中旬、下旬
    if '旬' in keyword and any(x in keyword for x in ['上', '中', '下']):
        return True
    
    # 7.3 季度：任意数字+季度、Q+数字
    if re.search(r'第?' + ANY_NUM_PATTERN + r'\s*季度', keyword):
        return True
    if re.search(r'[Qq]' + ARABIC_NUM_PATTERN, keyword):
        return True
    
    return False


def is_time_keyword(keyword):
    """
    判断关键词是否为时间相关词汇
    
    使用与预处理阶段相同的检测规则，确保一致性。
    
    检测规则（与 preprocess_v2.py 中的 remove_time_expressions 保持一致）：
    - 标准时间格式：8:40、08:40:30、8.40、8点40分30秒
    - 口语化时间：8点多、8点半、8点左右、差5分8点
    - 时间段：早上、上午、中午、下午、晚上、凌晨、夜里
    - 模糊时间：刚才、现在、马上、立刻、稍后
    
    Args:
        keyword: 待检测的关键词
        
    Returns:
        True 如果是时间关键词，否则 False
    """
    if not isinstance(keyword, str):
        return False
    
    # ===== 1. 标准时间格式 =====
    
    # 1.1 冒号分隔的时间：8:40、08:40:30、23:59:59
    if re.search(r'\d{1,2}:\d{1,2}(:\d{1,2})?', keyword):
        return True
    
    # 1.2 点号分隔的时间：8.40、8.40.30（排除价格）
    if re.search(r'(?<!\d)\d{1,2}\.\d{1,2}(\.\d{1,2})?(?!\d)', keyword):
        return True
    
    # 1.3 中文完整时间：8点40分30秒、8点40分、8点40、八点四十分（统一使用通用数字模式）
    if re.search(ANY_NUM_PATTERN + r'\s*点\s*' + ANY_NUM_PATTERN + r'\s*分\s*' + ANY_NUM_PATTERN + r'\s*秒', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*点\s*' + ANY_NUM_PATTERN + r'\s*分', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*点\s*' + ANY_NUM_PATTERN + r'(?![分秒])', keyword):
        return True
    
    # 1.4 中文时间单位：8点、40分、30秒、8时、40分钟（数字或中文数字）
    if re.search(ANY_NUM_PATTERN + r'\s*[点时]\s*(?:钟)?', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*分\s*(?:钟)?', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*秒\s*(?:钟)?', keyword):
        return True
    
    # ===== 2. 口语化时间表达 =====
    
    # 2.1 模糊时间：8点多、8点半、8点左右、8点钟左右（数字或中文数字）
    if re.search(ANY_NUM_PATTERN + r'\s*[点时]\s*[多半来钟]', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*[点时]\s*(?:左右|上下)', keyword):
        return True
    
    # 2.2 差几分几点：差5分8点、差一刻9点（数字或中文数字）
    if re.search(r'差\s*' + ANY_NUM_PATTERN + r'\s*分\s*' + ANY_NUM_PATTERN + r'\s*[点时]', keyword):
        return True
    if re.search(r'差\s*' + CHINESE_NUM_PATTERN + r'\s*刻\s*' + ANY_NUM_PATTERN + r'\s*[点时]', keyword):
        return True
    
    # 2.3 几点几刻：8点一刻、9点三刻（数字或中文数字）
    if re.search(ANY_NUM_PATTERN + r'\s*[点时]\s*' + CHINESE_NUM_PATTERN + r'\s*刻', keyword):
        return True
    
    # ===== 3. 时间段表达 =====
    
    # 3.1 时间段：早上、上午、中午、下午、晚上、夜里、凌晨、深夜
    time_periods = ['早上', '上午', '中午', '下午', '晚上', '夜里', '凌晨', '深夜', '早晨', '傍晚', '黄昏']
    for period in time_periods:
        if period in keyword:
            return True
    
    # 3.2 带时间段的完整表达：上午8点、晚上9点半（数字或中文数字）
    if re.search(r'[早上中下晚夜凌深][上午里晨间夜]\s*' + ANY_NUM_PATTERN + r'\s*[点时]', keyword):
        return True
    
    # ===== 4. 模糊时间词 =====
    
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
    从文件加载停用词列表（带缓存机制，避免重复加载）
    
    Args:
        stopwords_file: 停用词文件路径（默认为当前目录下的 stopwords.txt）
        
    Returns:
        停用词集合（set）
    """
    # 如果提供的是相对路径，则相对于当前脚本所在目录
    if not os.path.isabs(stopwords_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stopwords_file = os.path.join(script_dir, stopwords_file)
    
    # 检查缓存中是否已经加载过该文件
    if stopwords_file in _stopwords_cache:
        return _stopwords_cache[stopwords_file]
    
    # 首次加载
    stopwords = set()
    
    # 检查文件是否存在
    if not os.path.exists(stopwords_file):
        logging.warning(f"停用词文件不存在: {stopwords_file}")
        _stopwords_cache[stopwords_file] = stopwords
        return stopwords
    
    try:
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 去除首尾空白字符
                word = line.strip()
                # 跳过空行和注释行
                if word and not word.startswith('#'):
                    stopwords.add(word)
        logging.info(f"成功加载 {len(stopwords)} 个停用词（来自: {os.path.basename(stopwords_file)}）")
        # 缓存结果
        _stopwords_cache[stopwords_file] = stopwords
    except Exception as e:
        logging.error(f"加载停用词文件失败: {e}")
        _stopwords_cache[stopwords_file] = stopwords
    
    return stopwords


def normalize_keywords_data(keywords_data, json_format="new"):
    """
    规范化关键词数据，确保分数为 float 类型，并验证关键词为字符串
    支持二元组和三元组格式的自动识别和转换
    
    Args:
        keywords_data: 原始关键词数据
            - 新格式三元组: [["推理", "关键词", 分数], ...]
            - 新格式二元组: [["关键词", 分数], ...]  # 缺少推理说明
            - 旧格式: [["关键词", 分数, "理由"], ...]
        json_format: JSON格式类型，"new"(新格式) 或 "old"(旧格式)
        
    Returns:
        规范化后的关键词数据（统一为三元组格式）
    """
    if not keywords_data:
        return []
    
    normalized_data = []
    for item in keywords_data:
        if not isinstance(item, list) or len(item) < 2:
            # 跳过格式不正确的数据（至少需要2个元素）
            continue
        
        # 根据长度和格式判断数据结构
        if json_format == "new":
            if len(item) == 3:
                # 标准三元组: ["推理", "关键词", 分数]
                reasoning, keyword, score = item[0], item[1], item[2]
            elif len(item) == 2:
                # 二元组: ["关键词", 分数] - 缺少推理说明
                # 需要判断哪个是关键词，哪个是分数
                if isinstance(item[0], str) and isinstance(item[1], (int, float, str)):
                    # 假设第一个是关键词，第二个是分数
                    reasoning = ""  # 空推理说明
                    keyword = item[0]
                    score = item[1]
                else:
                    # 格式不明确，跳过
                    continue
            else:
                # 长度不符合预期，跳过
                continue
        else:
            # 旧格式: ["关键词", 分数, "理由"]
            if len(item) >= 2:
                keyword = item[0]
                score = item[1]
                reasoning = item[2] if len(item) > 2 else ""
            else:
                continue
        
        # 验证关键词是否为字符串类型
        if not isinstance(keyword, str):
            continue
        
        # 转换分数为 float
        try:
            score = float(score)
        except (ValueError, TypeError):
            # 如果转换失败，设置默认分数
            score = 0.5
        
        # 统一输出为三元组格式: ["推理", "关键词", 分数]
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
    对关键词提取结果进行后处理
    
    Args:
        keywords_data: 原始关键词数据
            - 新格式: [["推理", "关键词", 分数], ...]
            - 旧格式: [["关键词", 分数, "理由"], ...]
        deduplicate: 是否去重（完全相同的词）
        sort_by_importance: 是否按 importance 分数从高到低排序
        filter_low_score: 是否过滤低分词
        score_threshold: 分数阈值（当 filter_low_score=True 时生效）
        top_n: 是否只保留前 N 个关键词
        n: 保留的关键词数量（当 top_n=True 时生效）
        return_full_info: 是否返回完整信息（包括分数和理由），False 则只返回关键词列表
        json_format: JSON格式类型，"new"(新格式) 或 "old"(旧格式)
        remove_english: 是否去除包含英文字母的关键词
        filter_stopwords: 是否启用停用词过滤
        stopwords_exact_match: 是否启用精确匹配（关键词完全等于停用词时过滤）
        stopwords_contain_match: 是否启用包含匹配（关键词包含停用词时过滤）
        stopwords_file: 停用词文件路径（默认为 stopwords.txt）
        filter_time_keywords: 是否过滤时间相关的关键词（如"8点"、"早上"等）
        filter_date_keywords: 是否过滤日期相关的关键词（如"27号"、"5月15"等）
        filter_long_keywords: 是否过滤超长关键词（长度超过max_keyword_length的关键词）
        max_keyword_length: 关键词最大长度（当 filter_long_keywords=True 时生效，默认为6）
        backfill_topn: 当启用top_n且过滤后数量不足N时，是否从被过滤的关键词中回填
        filter_not_in_original: 是否过滤原文中不存在的关键词（检查关键词每个字符是否都在原文中）
        original_text: 原始文本（当 filter_not_in_original=True 时必须提供，建议传入预处理后的文本）
        max_span_ratio: 关键词字符在原文中的最大跨度倍数（默认为2，防止字符散落拼凑）
        
    Returns:
        处理后的关键词列表或完整信息列表
    """
    # 如果输入为空，直接返回空列表
    if not keywords_data:
        return []
    
    # 深拷贝一份数据，避免修改原始数据
    processed_data = [item[:] for item in keywords_data]
    
    # 确定关键词和分数的位置（根据格式）
    if json_format == "new":
        # 新格式: [推理, 关键词, 分数]
        keyword_idx = 1
        score_idx = 2
    else:
        # 旧格式: [关键词, 分数, 理由]
        keyword_idx = 0
        score_idx = 1
    
    # 定义分数获取函数（后续多处使用）
    def get_score(item):
        """安全地获取分数，处理类型转换"""
        if len(item) <= score_idx:
            return 0
        score = item[score_idx]
        try:
            return float(score)
        except (ValueError, TypeError):
            return 0
    
    # 0. 过滤空关键词（最先执行，避免空关键词参与后续处理）
    # 移除关键词为空字符串的项
    filtered_data = []
    # logging.info(f"过滤空关键词前: len(processed_data) = {len(processed_data)}")
    for item in processed_data:
        if len(item) > keyword_idx:
            keyword = item[keyword_idx]
            # 确保关键词是字符串且非空
            if isinstance(keyword, str) and keyword.strip():
                filtered_data.append(item)
    processed_data = filtered_data
    # logging.info(f"过滤空关键词后: len(processed_data) = {len(processed_data)}")
    
    # 1. 按 importance 分数排序（如果启用）
    if sort_by_importance:
        processed_data.sort(key=get_score, reverse=True)

    
    # 2. 去重（如果启用）
    if deduplicate:
        seen_keywords = set()
        deduplicated_data = []
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # 只保留第一次出现的关键词（由于已排序，保留的是分数最高的）
                if keyword not in seen_keywords:
                    seen_keywords.add(keyword)
                    deduplicated_data.append(item)
        processed_data = deduplicated_data
    
    # 3. 过滤低分词（如果启用）
    if filter_low_score:
        def check_score(item):
            """安全地检查分数是否达到阈值"""
            if len(item) <= score_idx:
                return False
            try:
                score = float(item[score_idx])
                return score >= score_threshold
            except (ValueError, TypeError):
                return False
        
        processed_data = [item for item in processed_data if check_score(item)]
    
    # 注意：这里不再提前执行 top_n 截取
    # 4. top_n 截取移到所有过滤完成后（见后文）
    
    # 5. 去除包含英文字母的关键词（如果启用）
    if remove_english:
        filtered_data = []
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # 确保关键词是字符串类型
                if not isinstance(keyword, str):
                    continue  # 跳过非字符串类型的关键词
                # 检查关键词中是否包含英文字母
                if not re.search(r'[a-zA-Z]', keyword):
                    filtered_data.append(item)
        processed_data = filtered_data
    
    # 6. 过滤停用词（如果启用）
    if filter_stopwords and (stopwords_exact_match or stopwords_contain_match):
        # 加载停用词列表
        stopwords = load_stopwords(stopwords_file)
        
        if stopwords:  # 只有在成功加载停用词时才进行过滤
            filtered_data = []
            filtered_out = []  # 记录被过滤的关键词（用于调试）
            # logging.info(f"[停用词过滤] 过滤前关键词: {[item[keyword_idx] if len(item) > keyword_idx else '?' for item in processed_data]}")
            
            for item in processed_data:
                if len(item) > keyword_idx:
                    keyword = item[keyword_idx]
                    # 确保关键词是字符串类型
                    if not isinstance(keyword, str):
                        # logging.info(f"[停用词过滤] 跳过非字符串关键词: {keyword} (类型: {type(keyword)})")
                        continue  # 跳过非字符串类型的关键词
                    
                    should_filter = False
                    
                    # 精确匹配：关键词完全等于停用词
                    if stopwords_exact_match and keyword in stopwords:
                        # logging.info(f"[停用词过滤] '{keyword}' 匹配停用词（精确匹配）")
                        should_filter = True
                    
                    # 包含匹配：关键词包含停用词
                    if stopwords_contain_match and not should_filter:
                        for stopword in stopwords:
                            if stopword in keyword:
                                should_filter = True
                                # logging.info(f"[停用词过滤] '{keyword}' 包含停用词 '{stopword}'（包含匹配）")
                                break
                    
                    # 如果不需要过滤，则保留该关键词
                    if not should_filter:
                        filtered_data.append(item)
                    else:
                        filtered_out.append(keyword)

            # 调试输出（可选，取消注释以启用）
            # logging.info(f"[停用词过滤] 过滤前: {len(processed_data)} 个, 过滤后: {len(filtered_data)} 个, 被过滤: {filtered_out}")
            
            processed_data = filtered_data
    
    # 7. 过滤时间关键词（如果启用）
    if filter_time_keywords:
        filtered_data = []
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # 确保关键词是字符串类型
                if not isinstance(keyword, str):
                    continue  # 跳过非字符串类型的关键词
                
                # 检查是否为时间关键词
                if not is_time_keyword(keyword):
                    filtered_data.append(item)
        
        processed_data = filtered_data
    
    # 8. 过滤日期关键词（如果启用）
    if filter_date_keywords:
        filtered_data = []
        filtered_out_dates = []  # 记录被过滤的日期关键词
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # 确保关键词是字符串类型
                if not isinstance(keyword, str):
                    continue  # 跳过非字符串类型的关键词
                
                # 检查是否为日期关键词
                if not is_date_keyword(keyword):
                    filtered_data.append(item)
                else:
                    filtered_out_dates.append(keyword)
        
        # 调试输出（可选）
        # if filtered_out_dates:
        #     logging.info(f"[日期过滤] 过滤前: {len(processed_data)} 个, 过滤后: {len(filtered_data)} 个, 被过滤: {filtered_out_dates}")
        
        processed_data = filtered_data
    
    # 9. 过滤超长关键词（如果启用）
    if filter_long_keywords and max_keyword_length > 0:
        filtered_data = []
        filtered_out_long = []  # 记录被过滤的超长关键词
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # 确保关键词是字符串类型
                if not isinstance(keyword, str):
                    continue  # 跳过非字符串类型的关键词
                
                # 检查关键词长度是否超过最大长度
                if len(keyword) <= max_keyword_length:
                    filtered_data.append(item)
                else:
                    filtered_out_long.append(keyword)
        
        # 调试输出（可选）
        # if filtered_out_long:
        #     logging.info(f"[长度过滤] 过滤前: {len(processed_data)} 个, 过滤后: {len(filtered_data)} 个, 被过滤: {filtered_out_long}")
        
        processed_data = filtered_data

    # 10. 过滤原文中不存在的关键词（如果启用）
    if filter_not_in_original and original_text:
        processed_data = filter_keywords_not_in_original(
            processed_data, 
            original_text, 
            keyword_idx=keyword_idx,
            max_span_ratio=max_span_ratio
        )


    # ========== 重要：在所有过滤完成后，保存干净的数据用于回填 ==========
    # 这样回填时不会回填被停用词/时间词/日期词/超长关键词过滤掉的关键词
    clean_sorted_data = [item[:] for item in processed_data]
    
    # 11. 智能 top_n 处理（所有过滤完成后）
    if top_n and n > 0:
        # 先截取前 N 个
        if len(processed_data) > n:
            processed_data = processed_data[:n]
        
        current_count = len(processed_data)
        
        if current_count < n and backfill_topn:
            # 当前数量不足 N 个，需要回填
            # 计算需要回填的数量
            needed_count = n - current_count
            
            # 获取当前已选中的关键词集合（用于去重）
            selected_keywords = set()
            for item in processed_data:
                if len(item) > keyword_idx:
                    keyword = item[keyword_idx]
                    if isinstance(keyword, str):
                        selected_keywords.add(keyword)
            
            # 从干净的数据中找回填候选，而不是从原始数据;确保回填的关键词也是经过停用词和时间词过滤的
            backfill_candidates = []
            for item in clean_sorted_data:
                if len(item) > keyword_idx:
                    keyword = item[keyword_idx]
                    # 确保是字符串且未被选中
                    if isinstance(keyword, str) and keyword not in selected_keywords:
                        backfill_candidates.append(item)
            
            # 从候选中选择分数最高的进行回填
            backfill_items = backfill_candidates[:needed_count]
            
            # 将回填的关键词追加到结果末尾
            processed_data.extend(backfill_items)
            
            # 调试输出
            if backfill_items:
                backfilled_keywords = [item[keyword_idx] for item in backfill_items if len(item) > keyword_idx]
                logging.info(f"[智能回填] 当前: {current_count} 个, 目标: {n} 个, 回填: {len(backfill_items)} 个 → {backfilled_keywords}")
        
        # 最终截取到 N 个（如果回填后超过 N 个，需要截断；正常情况下应该正好是 N 个或少于 N 个）
        processed_data = processed_data[:n]
    
    # 11. 根据 return_full_info 决定返回格式
    if return_full_info:
        # 返回完整信息
        return processed_data
    else:
        # 只返回关键词列表（确保是字符串）
        keywords_list = []
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # 只添加字符串类型的关键词
                if isinstance(keyword, str):
                    keywords_list.append(keyword)
        return keywords_list


def extract_keywords_from_json(keywords_json, return_raw=False):
    """
    从 JSON 格式的关键词数据中提取关键词列表
    
    Args:
        keywords_json: JSON 格式的关键词数据（字典）
        return_raw: 是否返回原始格式（包含分数和理由）
        
    Returns:
        关键词数据列表，格式为 [["词1", 0.90, "理由"], ...] 或 []
    """
    # 统一处理返回的关键词部分（兼容中英文键名）
    if '关键词' in keywords_json:
        keywords_data = keywords_json['关键词']
    elif 'keywords' in keywords_json:
        keywords_data = keywords_json['keywords']
    else:
        return []
    
    return keywords_data

