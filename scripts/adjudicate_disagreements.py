import csv
import json
import os

# Each entry: (sample_id, adjudicated_keywords, reason, notes)
# reason: primary_preferred | secondary_preferred | merged | new_decision | exclude
# All keywords MUST appear verbatim in source_text, 1-4 chars each, atomic, deduplicated, max 15

ADJUDICATIONS = [
    # Row 1: 手機還不錯...
    ("0443b9a584f050e6de95c9c9ac31ce91a28a0dadb7adc7262303d1d1a87b8d63",
     ["手機", "不錯", "攝像頭", "白點", "不良", "品質"],
     "merged", "P多'手機'(对象词有意义)和'三星'(品牌非评价关键词去掉); S的核心关键词完整"),

    # Row 2: 开机一看，屏幕碎了
    ("09ac30f684f688f069c416b2920b07975f5c6e5ff31f944150c97825b5a736b0",
     ["屏幕", "碎"],
     "secondary_preferred", "'开机'是动作非关键评价词"),

    # Row 3: 湾仔码头...口感和汤料
    ("0a392d4c80793c3f1f642c5c3f1cb44df1c7a0a7479f913ae18c1fe06fd39670",
     ["口感", "汤料", "好", "信赖"],
     "secondary_preferred", "P多了'湾仔码头'(品牌),'馄饨','早餐'(对象背景),'大品牌','值得信赖'(>4字); S简洁合理"),

    # Row 4: 全款前15预约...坑
    ("0cb443a513b72ce6759e257ea278d0b0a00cfacb8e82046d245f01fd9b5e1731",
     ["到货", "延保", "预定", "坑", "喜欢"],
     "secondary_preferred", "P多了'预约','无线充','礼品','绿色'等背景词; S简洁聚焦评价"),

    # Row 5: 电视机不错，音响效果好
    ("0e0bcb07fe32ffe957d6306a850600a32bb99426c372fae29e5e6b8f1641da4c",
     ["电视机", "不错", "音响", "好", "画质", "清晰", "投诉", "不开心", "安装工", "态度", "服务"],
     "secondary_preferred", "S补充了评价词'好'; P遗漏了该关键评价词"),

    # Row 6: 收到的是个次品
    ("0fa33cba8a3e394ab59f7b844a2acc9a8300ab0ece5cbaef07fac070be715bf7",
     ["次品", "屏幕", "黑点", "脏"],
     "secondary_preferred", "P多了'换回来','三星','手机','封包'等非核心评价词"),

    # Row 7: 差评，收到货3天就降价
    ("135f1c6b69c6d14e05d5cf33a832f67fc5048142ccef3fc19e8e238046041985",
     ["降价", "价格保护", "失望"],
     "secondary_preferred", "'差评'和'京东'非关键评价概念"),

    # Row 8: 运行内存...卡成狗
    ("15b38de5d25530688fd06f513ae05ce582d34bd60aea080af60604e6fc49013c",
     ["内存", "卡", "伪造"],
     "secondary_preferred", "P的'运行内存'>4字重复'内存','游戏','真假','苹果'为背景"),

    # Row 9: 包装简陋，非常粗糙
    ("1712bb20fe27c8b3bec5424dc4a71137102ae2684564a423a96f1a9cbd54cbe5",
     ["到货", "包装", "简陋", "粗糙", "差"],
     "merged", "保留P的'到货'(物流评价),S的'差'(明确评价);去掉品牌词和非核心对象词"),

    # Row 10: 快递很快，快递员也很负责
    ("1879f8925fd6583b0358e7bc8985a5707d1786c8debfbe6728685539aaf5e5dd",
     ["快递", "快", "负责", "手感", "好", "屏幕", "黄"],
     "secondary_preferred", "P多了'快递员','手机壳','兼容','原色模式'等非核心;S简洁且补充了评价词'好'"),

    # Row 11: 以为双12会有活动
    ("1a14a88cfd43b97480bcd26038b570764dbc764d33eb242e687d3bcec1837a5a",
     ["赠品", "退货", "运费", "坑"],
     "secondary_preferred", "P的'活动','补赠品'非核心;'补赠品'>4字"),

    # Row 12: 充电发烫
    ("1c301e26038660d35938ab8afefa1ce3b821b6e78a4ab3d1276b119dd24f6f03",
     ["充电", "发烫", "检测"],
     "secondary_preferred", "'京东','三星','无语'非核心评价词"),

    # Row 13: 真的很喜欢...质量很好
    ("1dfdf06ff20e95da85f7c0f90c4fe6909e15d8e862e9a86384c11cf3b9588fc5",
     ["喜欢", "质量", "好", "满意", "发货", "速度", "快", "包装", "仔细", "物流", "服务", "态度"],
     "secondary_preferred", "S原子化更好:拆分'服务态度'为'服务'+'态度';补充了'仔细','速度'等评价词"),

    # Row 14: 艾伦伯顿运动男士套装
    ("1f24bba8c84a8d10681bbecdf836c79aed98781b64c0f94c09a20a90d905b2b1",
     ["面料", "弹力", "手感", "好", "做工", "精细", "贴身", "不错", "物流", "快"],
     "secondary_preferred", "P多了品牌'艾伦伯顿'>4字,'运动','套装','五星'等背景词;S简洁聚焦"),

    # Row 15: 不怎么好，音量键都已经按不动了
    ("1fbd1bec650773df8e87d60996136ad44eac2ea912a2ddad01f334e5582816ab",
     ["音量键", "按不动", "锁屏键"],
     "primary_preferred", "'不怎么好'虽在原文但不够原子化,P的notes正确指出了这一点"),

    # Row 16: 手机确实好用...果冻屏
    ("251c60c3d9f6014e60218d1912ca216535fe96c47ac565453fd282ba0b2ac816",
     ["好用", "果冻屏", "碎", "曲面屏", "硬度", "碎屏", "质量", "售后", "维修"],
     "secondary_preferred", "S完整覆盖;P多了'手机','麻烦','屏','保护套','碎屏保'等词"),

    # Row 17: 耳机...售后
    ("29c29c2825737b0fde2163945aebcae7be9c797417bf9653ce09627475761095",
     ["耳机", "售后", "发票", "给力"],
     "secondary_preferred", "去掉品牌词'京东','三星'和'扯皮'(背景)"),

    # Row 18: 充电变这么长时间
    ("2ae87beb4e6aa879fe26c2529128dc5a66e840714893f03b385736668c1d6aa2",
     ["充电", "质量", "差", "客服", "快充", "慢"],
     "secondary_preferred", "P多了'手机','长时间','检测','反馈','麻烦'等背景词;S补充了'快充'"),

    # Row 19: 质量问题太严重了
    ("2ccc63e0cae1b6c73a8f55c8f43c9e5ffb2b7d9310ad59f935d6fe09aa422767",
     ["质量", "返回键", "失灵"],
     "secondary_preferred", "'问题','严重'太笼统"),

    # Row 20: 所有镜头都左歪的
    ("2d57a2bd0601842b17d6f7af93d69252c392933fc2836d985d66625fd9ec1b75",
     ["镜头", "歪", "换货", "品控"],
     "secondary_preferred", "P的'左歪'和'歪'重复,'问题'太笼统"),

    # Row 21: 差评给东的。预售没先发货
    ("2fdeb4e2e5ee60529a0206a289f729ac7e50e0f789f26bcd1158851e8286eb09",
     ["预售", "发货", "做工", "缝隙", "发热", "失望", "贴膜", "翘边", "掉色", "弧形屏幕"],
     "secondary_preferred", "S更好:补充了'发货','失望';去掉了'差评','返现','屏保','手机','山寨','无线充电'等非核心词"),

    # Row 22: 外观很不错...屏幕细腻
    ("302d8a4729f85e35ee21ad8e0d5588fe13a68f7bed26af3e44bfc73620cbb6bc",
     ["外观", "不错", "屏幕", "细腻", "舒适", "分辨率", "性能", "满意", "手感", "轻", "客服", "态度", "发货"],
     "secondary_preferred", "S补充了'轻','发货';P的'惊艳','单手','专员'非核心"),

    # Row 23: 小刘海...舒服...运行速度
    ("309bb4563a1c1c1371223c97c33eba748166b3b9b67b8a0e90b4689290cf869a",
     ["小刘海", "舒服", "运行", "速度", "快", "电量", "内存", "爽"],
     "secondary_preferred", "S原子化更好:拆分'运行速度'为'运行'+'速度';补充'爽';去掉'强迫症','苹果'"),

    # Row 24: 电视...很好...安装师傅
    ("323c51e821b825d314318a0f2162c92fba99e67799b1d24794a700f79f300a64",
     ["电视", "好", "质量", "色彩", "安装", "耐心"],
     "secondary_preferred", "S拆分'安装师傅'为'安装'更原子化;'棒棒'为语气词非核心评价"),

    # Row 25: 手机烫手 电池不耐用
    ("331493233e1366ec5e17b0c336077af3deb61e2443f1073380f12560eced1262",
     ["烫手", "电池", "不耐用"],
     "secondary_preferred", "'手机'(对象背景),'天气'(非评价)非核心"),

    # Row 26: 买之前客服说没拆封过
    ("35201f48235d708dd1dada442ef8a9f305c9afab4207e496c6d3d3729d708685",
     ["客服", "拆封", "不爽"],
     "secondary_preferred", "'机子','动过手脚'>4字,'慎重购买'>4字 非核心"),

    # Row 27: 不合格的电子产品
    ("367641baa0dacdffddd7ee6f40212d5b9b3568e5b7a8f0a7a598ab6b5521173d",
     ["不合格", "浪费"],
     "secondary_preferred", "'电子产品'>4字非必要;'时间','金钱'为'浪费'的宾语背景"),

    # Row 28: 三星手机烂...弯了
    ("367c85edfe1abaf3ef4b004cdd0abf4c76da7c7a5d37878e39f5c8dfb65fc463",
     ["烂", "弯", "质量", "维修", "屏幕", "金属环"],
     "secondary_preferred", "S简洁;P的'三星','手机','质量问题'>4字,'维修部'>合并为'维修'"),

    # Row 29: 速度很流畅，也不卡顿
    ("374058956c2a6f2084798c399acb6d07a2592fc519804142687d50bdda7c5113",
     ["速度", "流畅", "不卡顿", "外观"],
     "new_decision", "P的'卡顿'未保留否定;S的'不卡顿'更准确;'异常'在原文语义不明(P的notes也指出);去掉'京东'品牌词"),

    # Row 30: 质量特别好...颜值高
    ("3bac1bae89e3f92a094ba44bbcb801581db1dc93b3e9deaf9057ba8b06ea9d0f",
     ["质量", "好", "颜值", "高", "满意", "物流", "快", "不错", "价格", "实惠"],
     "secondary_preferred", "S补充了'高','不错';P多了'购物','值得购买'>4字"),

    # Row 31: 口感特别好，孩子爱吃
    ("3c7e374c9df34a225a5f165feeba2868454c110f62852494e7075200114524d9",
     ["口感", "好", "爱吃", "服务", "周到", "细致", "耐心", "物流", "速度", "快"],
     "secondary_preferred", "S原子化更好;P的'湾仔码头'品牌,'孩子','牌子','回购','卖家','建议','物流速度'>4字非核心"),

    # Row 32: 垃圾...屏幕不灵...卡
    ("3cef3fe0dec0d333d7a9b403053cc7ae4d30aea7a3286733b29fedadc10d731c",
     ["垃圾", "屏幕", "不灵", "卡", "内存"],
     "secondary_preferred", "P多了'不能删除'>4字,其余一致"),

    # Row 33: 屏幕...花屏
    ("3f332f919eb2d4bd2a5550eabb899678ab5bfdfd59d6f8185eca62c82f7e1077",
     ["屏幕", "花屏"],
     "secondary_preferred", "'花'和'花屏'重复提取同一概念,保留'花屏'更完整"),

    # Row 34: 写着12期免息
    ("452f8cac2b09cfcfcfcaa5ee630b8c08bf0945fc3372452ac3b99e9ce12deaa6",
     ["免息", "服务费", "客服", "心寒"],
     "secondary_preferred", "'还款','分期','证据'为背景信息"),

    # Row 35: 屏幕...坏屏...换货
    ("46762c6e657ac7110c76c8b9b339eb309a02c6d86a7b519e1aec730a7ca9758c",
     ["屏幕", "坏屏", "装配", "检测", "换货", "赠品", "不错"],
     "secondary_preferred", "P多了'黑','外力','手机'等背景词"),

    # Row 36: 京东商城网购...忽悠
    ("4d967a8c1cfe020ab624ac2a0a9f743c1e622e2dc46ae9dbd87e6ec99f812df7",
     ["忽悠"],
     "secondary_preferred", "'京东','网购','注意'为背景;'忽悠'是唯一核心评价"),

    # Row 37: 刚买没几天就多送了手机，变相降价
    ("62588f059cbd5842c0d545f30d8cbb36b8401583df96a5be6c67ea50074bf9e5",
     ["降价"],
     "secondary_preferred", "'手机'对象背景,'变相降价'>4字;'降价'已覆盖核心概念"),

    # Row 38: 京东...店大欺客...赠品
    ("6d7e1e21c33ebecc9bd98af8505335ad896f605818054263bc30291eae720773",
     ["店大欺客", "赠品", "客服", "信誉"],
     "secondary_preferred", "S简洁聚焦;P过多背景词'京东','手机','充电环','以次充好'>4字,'发错','拒收','文字游戏'>4字,'品牌信誉'>4字,'文字陷阱'>4字"),

    # Row 39: 不选择苹果6...质感...发烫
    ("6e76ac0b5c82500a6c59672647c139379d483e8f0baa5f59ee7dd8877641ea01",
     ["质感", "发烫", "阉割"],
     "secondary_preferred", "P多了品牌词'苹果','三星','手机','高通','消费者'等背景"),

    # Row 40: 物流发货快，包装好
    ("70ba983031d24a7952ba965ad65ea2719eb9a53b3f562f5d7ac13dcd63cbd2c4",
     ["物流", "发货", "快", "包装", "好", "服务", "态度", "衣服", "质量", "不错", "舒服"],
     "secondary_preferred", "S原子化更好:拆分'服务态度'为'服务'+'态度';补充评价词'好'"),

    # Row 41: 不满意...外观好...耗电
    ("72a8e4c29046729c4884b685d6cb0fc618cc832f445e510fc92445883eb2754c",
     ["不满意", "外观", "好", "耗电", "充电", "烫手", "内存", "价格", "高"],
     "merged", "两边基本一致;P多了'快'(耗电快的'快')但此处'快'修饰'耗电'不独立;S已足够完整"),

    # Row 42: 这6手机...按键...死键
    ("745dbe4ec287a5471f0ef8f7d28767a6a18fc065612984231511f44df3f62d69",
     ["按键", "死键"],
     "secondary_preferred", "'手机','问题','还可以'为背景或笼统词"),

    # Row 43: 尺码标准...物美价廉
    ("752ff76bba95eff2d3f813c3ad484f10465a4ba581864369d6f2e7140ad604e5",
     ["尺码", "合适", "款式", "好看", "柔软", "透气", "轻薄", "舒适", "包装", "物流", "快", "物美价廉", "满意"],
     "merged", "S补充了'快';P的'标准'和'合适'重复保留'合适';P的'面料'和'柔软'重复保留'柔软'"),

    # Row 44: 真心垃圾手机...刮花
    ("7864349bb6a471736810f1bd6fcab6bbe71b9a6a8d06b624b1e4370b23e26d4d",
     ["垃圾", "手机", "屏幕", "刮花", "误触", "恶心", "毛病"],
     "secondary_preferred", "S补充了'恶心'(明确评价);去掉'星钻黑'(型号),'三星'(品牌)"),

    # Row 45: 信。用 卡刷的钱...碎了
    ("7cacd9f4999ff4fe13899cfb53abf7a442b2e5c8ac4952b30e98d589d4ff1ced",
     ["碎"],
     "secondary_preferred", "P的'没还'不在原文连续出现;S正确仅提取'碎'"),

    # Row 46: 省电模式...耗电...卡
    ("806bb84be337ae6924fc1546005dd834556e618ca2903f5659290c55128fdc8b",
     ["省电", "耗电", "卡", "指纹识别", "延迟", "拍照片", "不错"],
     "merged", "P的'省电模式'>4字简化为'省电';S的'拍照片','不错'更完整;去掉'快','退了'"),

    # Row 47: 充电器充一次充不了
    ("80760499d9e652c6422ca25a292f3524b7cea34f6fada9102d3151093f11e6c0",
     ["充电器", "坏"],
     "secondary_preferred", "'手机'无评价义背景词"),

    # Row 48: 买来三天，出现多次死机
    ("815a53f6e96e8f35bdbadbf0716967c8c9ebafe63a973bb6e7104f7df39297d3",
     ["死机"],
     "secondary_preferred", "'中奖'为反讽非关键评价"),

    # Row 49: 买时明明标的送移动电源
    ("81c4768196c8e470cb72fe1d8d8955ad9761137d4bf6ccb5f26c877833db535e",
     ["移动电源"],
     "secondary_preferred", "'没有'太笼统"),

    # Row 50: 看起来还不错，电视剧很大的
    ("84b10cd2dfea3929206313dc73cae0787d5fae78493a4f1aa84821d47ce4e34f",
     ["不错", "大"],
     "secondary_preferred", "'电视剧'疑为'电视机'误写,不应作为关键词"),

    # Row 51: 太差劲了...退货
    ("868a2e86db7273af1296119b3ddd5f7ea04e6892715597a731f6580b691996ac",
     ["差劲", "退货"],
     "secondary_preferred", "P多了'银色','黑色','活动','京东','机子'等背景词"),

    # Row 52: 快递 太次了
    ("8ea141a78c477eff563151807f3e3e5c4cf5985dfea0b479f5297c7be631d711",
     ["快递", "次"],
     "secondary_preferred", "'人名'非评价关键词"),

    # Row 53: 白条...免息
    ("9094bf25a57285caf7c89a22de77f1e97eb1eb7e51f8111e2a40afc6055ad944",
     ["免息"],
     "secondary_preferred", "'白条'为金融产品名;'免息选项'>4字;'玩我'非标准评价"),

    # Row 54: 健身...运动服...质量
    ("91e2372c96d010f23b261ac9945fa3f483951f8a49365d8ff809a3353162767e",
     ["健身", "运动服", "质量", "好", "包装", "快递", "快", "价格", "实惠"],
     "secondary_preferred", "S补充了评价词'好';P的'暴赞'非标准评价"),

    # Row 55: 东西是不错的...没货
    ("95c1c3ed3b0f2ce7ccc2695f78c834e8481ef9937cfe4fe82be4715c94b4ecb2",
     ["不错", "没货", "退货", "失望"],
     "secondary_preferred", "P多了'东西','京东','差评','电商'等背景词"),

    # Row 56: 手机很漂亮...预定
    ("95cfa65e6d1833897b80583af1fabdccb4a74b3c5a70f273765e8642ed256975",
     ["手机", "漂亮", "预定", "后悔"],
     "merged", "保留P的'手机'(对象词);去掉'值得购买'>4字,'京东'"),

    # Row 57: 手机买了10天就黑屏了
    ("975c8bbed0b6081b0316c32ed34dcc8ae8fed6429a84910d3d3990be2ace0bd9",
     ["黑屏", "售后", "换货", "骗人", "信誉"],
     "secondary_preferred", "P多了'手机','协调','解决方案'>4字,'电子产品'>4字等背景词"),

    # Row 58: 物流就是快 东西也是正品
    ("99e8ba4b7f05c48454ccf3422224e2f8929c8324d3d6eca544ee648e908fc41f",
     ["物流", "快", "正品"],
     "secondary_preferred", "'东西'太笼统"),

    # Row 59: 无法开机...充电...发烫
    ("9ae0f8f8731b4616d824d6eb0bb558c55ce8463dc1ea463ea6f5e2aa3dd9aa73",
     ["无法开机", "充电", "发烫", "客服", "外观", "漂亮"],
     "secondary_preferred", "P多了品牌词'京东','白条','三星','手机','京豆'等;'换货'在原文为'没办法换货'背景"),

    # Row 60: 三星...手机...质感...相机抖动
    ("9ddb631fd87b1fb64c735e3c922a919562bb23d83378e7b9300310f44b14e269",
     ["质感", "提升", "相机", "抖动"],
     "secondary_preferred", "P多了品牌'三星','手机'背景;S简洁覆盖核心"),

    # Row 61: 伤心了...降...价格保护
    ("a105ed6e71bd5ba92cd154098bd3c6f96119b269ffcc0e182aa4e32fddbde637",
     ["伤心", "降", "价格保护", "不友好"],
     "secondary_preferred", "'老用户','京东','不错'非此处核心差异评价"),

    # Row 62: 多商品模板好评(长文本)
    ("a939ce04a113841f2fc038c78dee0cc4934125d6c8e809080dd1105c08d223c3",
     ["舒服", "效果", "不错", "服务", "热情", "质量", "正品", "做工", "细致", "修身", "发货", "快", "瑕疵"],
     "secondary_preferred", "S原子化更好,补充了'热情','细致','修身';P多了'宝贝','超赞','掌柜','鞋子','款式','物有所值'>4字,'行货'等"),

    # Row 63: 手机屏左下角就开裂了
    ("aabba1086032331583f8b78e15cdcc60cc4976674e4cf3e382e0d4fa0ecd732a",
     ["开裂", "镜面屏", "质量", "差", "保修"],
     "secondary_preferred", "P多了'手机','屏'(与'镜面屏'重复),'摔','倒霉','三思','不给保修'>4字;S的'保修'更原子化"),

    # Row 64: 返回键和选择键...死机
    ("aeb50afb4f6a6240e3eb7949752077d016a4434ef888da3a91a2cbd200fadb6b",
     ["返回键", "死机", "售后", "维修", "退货", "质量"],
     "secondary_preferred", "S补充了'维修';P的'三星','检测','选择键'次要"),

    # Row 65: 双11价格非常良心...质感...耳机音质
    ("b1f8537218c3d080ee2bdc4b46de166fcf0e48acc2f1f1bed3d64a68bcb0c7ab",
     ["价格", "良心", "质感", "喜欢", "速度", "快", "耳机", "音质", "丰富", "饱满", "力度"],
     "secondary_preferred", "S补充了'丰富','饱满','力度'(音质评价);P多了'发货','送达','惊喜','深空灰','屏','大','速度快'>4字,'数据接口'>4字"),

    # Row 66: 东西收到...发货快...贴心
    ("b326a86dd61e9ac9d2717f26f24ae097be60e71a2ec52b53c42332a1f338755d",
     ["喜欢", "发货", "速度", "快", "服务", "到位", "耐心", "贴心"],
     "secondary_preferred", "P多了'东西','图片','描述','一致','超级喜欢'>4字,'卖家','推荐','小礼物','五星好评'>4字等背景词"),

    # Row 67: 衣服收到了，质量不错
    ("b6753fbcbd3d7c9761515e789c6790efb1a087fd8fe96bc62cc0676239f17ed0",
     ["衣服", "质量", "不错", "合适", "快递", "快", "服务", "态度", "好"],
     "secondary_preferred", "S原子化更好:拆分'大小合适'>'合适','服务态度'>'服务'+'态度';P的'大小合适'>4字,'服务态度'>4字"),

    # Row 68: 质量非常好...发货速度快
    ("bf436522f9141bcd17ef5a235423ea35fedda70833bff5f15da2314d7364ae31",
     ["质量", "好", "发货", "速度", "快", "包装", "仔细", "严实", "满意"],
     "secondary_preferred", "S原子化更好,补充了'速度','仔细';P多了'超出','期望值','产品','超级棒'>4字"),

    # Row 69: 电视不错...降价...价格保护
    ("c1200fcc888fb0357090c7a30ec45550e89f62ee65ad375779b06311883bc07d",
     ["电视", "不错", "降价", "价格保护"],
     "primary_preferred", "P完整覆盖;S遗漏了'价格保护'"),

    # Row 70: 湾仔码头的忠实粉丝...口味好吃
    ("c3d310b832371c3d49f2086197f4de4fff6fff8ddc15bdb1b169d3a80cd167b7",
     ["口味", "好吃", "配料表", "干净", "放心", "物流", "快", "结实"],
     "secondary_preferred", "S补充了'结实';P的'湾仔码头'品牌,'粉丝','冻'为背景"),

    # Row 71: 降价了...价格保护...赠品
    ("c3e9f74d3e8cff59e28be411c5eb6ad456c64726d0d3d14c127f44184a2829c5",
     ["降价", "价格保护", "赠品"],
     "secondary_preferred", "P多了'京东','消费者','无爱'等非核心;S简洁聚焦"),

    # Row 72: 手机用了几天...满意...预定...订金
    ("c3f9af822eaff01f5ae724f87c236cfea77a59b9ad21d6a6c39b3f2cfb1c7fda",
     ["满意", "预定", "发货", "订金", "违约", "坑"],
     "merged", "保留S核心+P的'坑';去掉'手机','京东','供应商','采购','定金'(与'订金'重复),'数字游戏'>4字,'双倍赔付'>4字,'抱歉'"),

    # Row 73: 卡顿，无语，已退货
    ("c648a5571a3bcdffdcc649ac2088fa55ae57e24a089be45493116fbff8ff67e3",
     ["卡顿", "退货"],
     "secondary_preferred", "'无语'为情绪词非核心评价"),

    # Row 74: 三星屏幕就是垃圾
    ("ca255d10c27f36fd3df6b01ea96794ee18703b75e0ce6c7b8fd368423bf0b2d4",
     ["屏幕", "垃圾", "碎"],
     "secondary_preferred", "'三星'品牌词,'一塌糊涂'>4字"),

    # Row 75: 快递...赞...苹果系统...镜面
    ("cf2e882b95e9f8a692e15d513d3c90e75d0fead3a3b8535d3afdfcb3e1059106",
     ["快递", "赞", "快", "镜面", "手感", "好"],
     "secondary_preferred", "S补充了评价词'好';P多了'苹果','系统','安卓','手机'等背景词"),

    # Row 76: 左键频繁自己启动 屏幕乱跳
    ("d0d48ca6caabe88cd67d85585a1983b2097b91486621d5e5cfe7ffee910194f0",
     ["左键", "屏幕", "乱跳"],
     "secondary_preferred", "'自己启动'>4字不够原子化"),

    # Row 77: 新款的手机还是二手的
    ("d25dad3c1260171b23ecc2798ee9d1040a5cbe5e536e6a6d15d9c9d738c497cc",
     ["手机", "二手"],
     "secondary_preferred", "'新款'为背景描述非评价"),

    # Row 78: 东西是真品...色差严重...客服
    ("d27d5e8d3d752b03355bb23c26f83bf8d4daca7bbd3a20538787ec74af8be4cb",
     ["真品", "色差", "退货", "客服", "耐心"],
     "secondary_preferred", "P过多背景词:'颜色'(与'色差'重复),'图片','店家','不同意','电池','开机','不能退换'>4字,'无理由','值得信赖'>4字"),

    # Row 79: 发票是假的
    ("d5738850e09d4c4a578a369e07c25042453147d0b94f38acb4cb194274ffa91f",
     ["发票", "假"],
     "secondary_preferred", "P的'查不上','假票'(与'假'重复),'正规发票'>4字,'支支吾吾'>4字均冗余"),

    # Row 80: 不错很好就是太大了
    ("dc1eadbf72fb0bc6554e86f1b5253baf54960d01f17407937bbca9f9510e6505",
     ["不错", "好", "大"],
     "secondary_preferred", "'摆不下'为'大'的引申,非独立评价"),

    # Row 81: 宝贝收到了，质量很不错
    ("dd98af2a2efc35208935e6a443ea816685bc78060424273e0816048ce620f3f3",
     ["质量", "不错", "物流", "快"],
     "secondary_preferred", "'宝贝','老板','回复','支持'为背景或笼统词"),

    # Row 82: 客服要序列号没有...打电话容易断
    ("dde259ebe6253240a133ac01f203b862b0c5899f7af12bcd9b52feab8cc75bb3",
     ["客服", "序列号", "断", "套路", "质量"],
     "secondary_preferred", "P多了'手机','打电话','差评','官网','贵','质量保证'>4字;S简洁"),

    # Row 83: 开机正常,运行良好...花屏
    ("ed24d959e8d56075d8a00ec1e3818e60ca0de154566efe91eed21db8a8e2bcee",
     ["运行", "良好", "满意", "花屏", "换货"],
     "secondary_preferred", "S补充了'良好';P多了'开机','待机','息屏','心寒'等背景词"),

    # Row 84: 卡死了...安卓...次品
    ("eeca95102956a2b595975b55e19cddad79edad3c09dbf167a1880ab8371cd155",
     ["卡死", "安卓", "次品"],
     "secondary_preferred", "S的'卡死'更完整(保留原文连续);P的'卡'不如'卡死'完整;'信不过'>隐含态度非独立评价"),

    # Row 85: 果然超薄...送货...不错
    ("f3d2e75bb569445e79354829fe74eafb5251b162577098a1bfdbc5d2d36d277e",
     ["超薄", "发热", "送货", "久", "不错"],
     "secondary_preferred", "S补充了'久'(物流时效负向评价)"),

    # Row 86: 客服好！预定手机...赠品
    ("f3fb843fce2e75670b560cf0bdff5736054d0bf143438c1d22ec1a87086ac284",
     ["客服", "好", "预定", "赠品"],
     "secondary_preferred", "S补充了评价词'好';P的'手机','商家','承诺'为背景"),

    # Row 87: 屏幕有瑕疵...边框...划痕...不经用
    ("fe35e5d8ac1d23f64fd81d61736f635303cfd03c5aaa4198b8aa50b170abf1b7",
     ["屏幕", "瑕疵", "边框", "划痕", "不经用", "指纹键", "坑人"],
     "secondary_preferred", "S简洁完整;P多了'换新','贴膜'背景词"),

    # Row 88: 京东买的...苹果手机...品质...流畅
    ("fe83fee0d4780fa83c3fe2c2b3a25c6e2150ba9e2bf569744bdec7309f53feef",
     ["品质", "配送", "满意", "外观", "正品", "流畅", "屏幕", "大", "爽", "赠品"],
     "secondary_preferred", "S简洁;P多了'京东','苹果','手机','京东配送'>4字,'序列号','保修','无线','充电器'等背景词"),

    # Row 89: 拿到快递...宝贝...质量...态度
    ("ff00ff0334293562c654a37728cfb329f3020b130b21a9566ffb8c7f0b5c5346",
     ["客服", "态度", "好", "喜欢", "质量", "不错", "满意"],
     "secondary_preferred", "P多了'快递','宝贝','想象','一样','推荐','店家','还可以'等背景或笼统词"),

    # Row 90: 机身有使用过的痕迹
    ("ff5fb441d446ede387720b5c26de22a31c6b8812070fd318bc8b79804e299fac",
     ["机身", "痕迹", "标签", "污渍", "换货"],
     "secondary_preferred", "S补充了'痕迹';P的'使用过'>背景,'条形码'>属性非评价,'封包'>非评价"),
]

ADJUDICATOR_ID = "A03"


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base, "data", "annotation", "annotation_disagreements.csv")
    output_path = os.path.join(base, "data", "annotation", "annotation_disagreements_adjudicated.csv")

    with open(input_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    adj_map = {a[0]: a for a in ADJUDICATIONS}

    for row in rows:
        sid = row["sample_id"]
        adj = adj_map.get(sid)
        if adj is None:
            print(f"WARNING: no adjudication for {sid}")
            continue

        _, keywords, reason, notes = adj
        row["adjudicator_id"] = ADJUDICATOR_ID
        row["adjudicated_status"] = "complete"
        row["adjudicated_keywords_json"] = json.dumps(keywords, ensure_ascii=False)
        row["adjudication_reason"] = reason
        row["adjudication_notes"] = notes

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Adjudicated {len(rows)} rows -> {output_path}")

    missing = sum(1 for r in rows if not r.get("adjudicated_status"))
    if missing:
        print(f"WARNING: {missing} rows missing adjudicated_status")
    else:
        print("All rows adjudicated.")

    # Validate keywords
    errors = 0
    for row in rows:
        kw_str = row.get("adjudicated_keywords_json", "")
        if not kw_str:
            continue
        kws = json.loads(kw_str)
        src = row["source_text"]
        for k in kws:
            if len(k) > 4:
                print(f"ERROR: '{k}' > 4 chars in {row['sample_id'][:16]}")
                errors += 1
            if k not in src:
                print(f"ERROR: '{k}' not found in source_text of {row['sample_id'][:16]}")
                errors += 1
        if len(kws) > 15:
            print(f"ERROR: {len(kws)} keywords (>15) in {row['sample_id'][:16]}")
            errors += 1

    if errors == 0:
        print("Validation passed: all keywords in source_text, <=4 chars, <=15 per row.")
    else:
        print(f"Validation found {errors} errors.")

    # Count reasons
    reasons = {}
    for row in rows:
        r = row.get("adjudication_reason", "")
        reasons[r] = reasons.get(r, 0) + 1
    print(f"Reason distribution: {reasons}")


if __name__ == "__main__":
    main()
