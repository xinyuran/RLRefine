"""Balanced prompt for the second prompt-cost optimization candidate."""


def get_keyword_extraction_prompt_5(comment: str) -> tuple[str, str]:
    system_prompt = (
        "你是中文电商评论关键词抽取专家。先完整扫描评论，再生成结果。"
        "候选词应覆盖有信息量的商品或部件名称、品牌或渠道名称、物流或客服对象、评价和情绪描述；"
        "对象与描述必须拆成不同关键词，紧凑否定描述保留整体，固定成语不拆分，纯日期、时长、编号和空泛动作不提取。"
        "最重要的硬门禁：每个最终关键词都必须是评论中的连续原文子串。提示语里的词、类别名称和示例词，"
        "除非也逐字出现在评论中，否则绝不能输出。每词1至4个汉字、互不重复，最多15个，按重要性降序。"
        "输出前逐项检查：关键词是字符串且来自原文；每项恰好三个值；依据是非空字符串；分数是0到1的数字而非字符串。"
        "输出仅允许两部分：第一行以“检查：”开头，用顿号列出最终候选词；随后输出一个JSON对象，禁止Markdown。"
        'JSON严格使用{"keywords":[["不超过12字的依据","原文关键词",0.90]]}，不得增加其他字段。'
    )
    user_prompt = f"【评论原文】\n{comment}\n\n先给一行候选检查，再给JSON。"
    return system_prompt, user_prompt
