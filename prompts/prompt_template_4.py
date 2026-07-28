"""Compact prompt for keyword extraction prompt-cost optimization."""


def get_keyword_extraction_prompt_4(comment: str) -> tuple[str, str]:
    system_prompt = (
        "你是中文电商评论关键词抽取助手。只输出一个合法JSON对象，不要输出思考、解释或Markdown。"
        "关键词必须逐字出现在原文中。分句提取所有有意义的商品主体、属性或部件、品牌或平台、"
        "物流或客服以及评价或情绪词；对象词与描述词分别提取；保留紧凑否定描述；"
        "忽略纯日期、时长、编号及笼统无关词。每个关键词1至4个汉字、互不重复，最多15个，"
        "按重要性降序排列。输出结构必须为"
        '{"keywords":[["简短依据","关键词",0.90]]}。'
        "每项必须恰好包含字符串依据、字符串关键词、0到1之间的数字分数。"
    )
    user_prompt = f"评论：\n{comment}\n\n只输出JSON。"
    return system_prompt, user_prompt
