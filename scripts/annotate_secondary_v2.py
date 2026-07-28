import csv
import json
import os

ANNOTATIONS = [
    {
        "sample_id": "f3d2e75bb569445e79354829fe74eafb5251b162577098a1bfdbc5d2d36d277e",
        "keywords": ["超薄", "发热", "送货", "久", "不错"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "eeca95102956a2b595975b55e19cddad79edad3c09dbf167a1880ab8371cd155",
        "keywords": ["卡死", "安卓", "次品"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "374058956c2a6f2084798c399acb6d07a2592fc519804142687d50bdda7c5113",
        "keywords": ["速度", "流畅", "不卡顿", "外观", "京东"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "7864349bb6a471736810f1bd6fcab6bbe71b9a6a8d06b624b1e4370b23e26d4d",
        "keywords": ["垃圾", "手机", "屏幕", "刮花", "误触", "恶心", "毛病"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "f3fd533fa229facfe673e520bdf67457f6ab2c380e0e5ae912766afd41faab20",
        "keywords": ["龟速", "没货", "预定", "垃圾"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "9f6c51a923baea160a7edb393c99103b3b901339b2d036ee7d97743991bc283a",
        "keywords": ["手机", "烫手", "发热"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "251c60c3d9f6014e60218d1912ca216535fe96c47ac565453fd282ba0b2ac816",
        "keywords": ["好用", "果冻屏", "碎", "曲面屏", "硬度", "碎屏", "质量", "售后", "维修"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "43ccce34e82cd957bc65afc8e3ec3b9f7fe1dd87bd4f42eaf058ef5b01fdbfb6",
        "keywords": ["电视机", "清晰", "不错"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "64fef8fbed556daa156f760999e508ebe53501f7a6c7eba22902b10bbefee64d",
        "keywords": ["手机", "发热", "卡"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "a939ce04a113841f2fc038c78dee0cc4934125d6c8e809080dd1105c08d223c3",
        "keywords": ["舒服", "效果", "不错", "服务", "热情", "质量", "正品", "做工", "细致", "修身", "发货", "快", "瑕疵"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "文本疑似多条评论拼接",
    },
    {
        "sample_id": "0e0bcb07fe32ffe957d6306a850600a32bb99426c372fae29e5e6b8f1641da4c",
        "keywords": ["电视机", "不错", "音响", "好", "画质", "清晰", "投诉", "不开心", "安装工", "态度", "服务"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "95c1c3ed3b0f2ce7ccc2695f78c834e8481ef9937cfe4fe82be4715c94b4ecb2",
        "keywords": ["不错", "没货", "退货", "失望"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "dc1eadbf72fb0bc6554e86f1b5253baf54960d01f17407937bbca9f9510e6505",
        "keywords": ["不错", "好", "大"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "91e2372c96d010f23b261ac9945fa3f483951f8a49365d8ff809a3353162767e",
        "keywords": ["健身", "运动服", "质量", "好", "包装", "快递", "快", "价格", "实惠"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "09ac30f684f688f069c416b2920b07975f5c6e5ff31f944150c97825b5a736b0",
        "keywords": ["屏幕", "碎"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "91bafebf1c95240ba58c7bbb4880bf8b62756b7ab2df302c2ff4a0a8b55e7c82",
        "keywords": ["电视", "不错", "好"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "309bb4563a1c1c1371223c97c33eba748166b3b9b67b8a0e90b4689290cf869a",
        "keywords": ["小刘海", "舒服", "运行", "速度", "快", "电量", "内存", "爽"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "c1200fcc888fb0357090c7a30ec45550e89f62ee65ad375779b06311883bc07d",
        "keywords": ["电视", "不错", "降价"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "806bb84be337ae6924fc1546005dd834556e618ca2903f5659290c55128fdc8b",
        "keywords": ["省电", "耗电", "卡", "指纹识别", "延迟", "拍照片", "不错"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "2fdeb4e2e5ee60529a0206a289f729ac7e50e0f789f26bcd1158851e8286eb09",
        "keywords": ["预售", "发货", "做工", "缝隙", "发热", "失望", "贴膜", "翘边", "掉色", "弧形屏幕"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "a105ed6e71bd5ba92cd154098bd3c6f96119b269ffcc0e182aa4e32fddbde637",
        "keywords": ["伤心", "降", "价格保护", "不友好"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "29c29c2825737b0fde2163945aebcae7be9c797417bf9653ce09627475761095",
        "keywords": ["耳机", "售后", "发票", "给力"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "ff5fb441d446ede387720b5c26de22a31c6b8812070fd318bc8b79804e299fac",
        "keywords": ["机身", "痕迹", "标签", "污渍", "换货"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "9ae0f8f8731b4616d824d6eb0bb558c55ce8463dc1ea463ea6f5e2aa3dd9aa73",
        "keywords": ["无法开机", "充电", "发烫", "客服", "外观", "漂亮"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "f3fb843fce2e75670b560cf0bdff5736054d0bf143438c1d22ec1a87086ac284",
        "keywords": ["客服", "好", "预定", "赠品"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "99e8ba4b7f05c48454ccf3422224e2f8929c8324d3d6eca544ee648e908fc41f",
        "keywords": ["物流", "快", "正品"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "bf436522f9141bcd17ef5a235423ea35fedda70833bff5f15da2314d7364ae31",
        "keywords": ["质量", "好", "发货", "速度", "快", "包装", "仔细", "严实", "满意"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "2d57a2bd0601842b17d6f7af93d69252c392933fc2836d985d66625fd9ec1b75",
        "keywords": ["镜头", "歪", "换货", "品控"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "d0d48ca6caabe88cd67d85585a1983b2097b91486621d5e5cfe7ffee910194f0",
        "keywords": ["左键", "屏幕", "乱跳"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "8ea141a78c477eff563151807f3e3e5c4cf5985dfea0b479f5297c7be631d711",
        "keywords": ["快递", "次"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "84b10cd2dfea3929206313dc73cae0787d5fae78493a4f1aa84821d47ce4e34f",
        "keywords": ["不错", "大"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "752ff76bba95eff2d3f813c3ad484f10465a4ba581864369d6f2e7140ad604e5",
        "keywords": ["尺码", "合适", "款式", "好看", "柔软", "透气", "轻薄", "舒适", "包装", "物流", "快", "物美价廉", "满意"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "cf2e882b95e9f8a692e15d513d3c90e75d0fead3a3b8535d3afdfcb3e1059106",
        "keywords": ["快递", "赞", "快", "镜面", "手感", "好"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "3c7e374c9df34a225a5f165feeba2868454c110f62852494e7075200114524d9",
        "keywords": ["口感", "好", "爱吃", "服务", "周到", "细致", "耐心", "物流", "速度", "快"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "367641baa0dacdffddd7ee6f40212d5b9b3568e5b7a8f0a7a598ab6b5521173d",
        "keywords": ["不合格", "浪费"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "95cfa65e6d1833897b80583af1fabdccb4a74b3c5a70f273765e8642ed256975",
        "keywords": ["手机", "漂亮", "预定", "后悔"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "2dc6f3d15a89e187780f6e93dc7f46375f3f3aa4dee56025b1aa5480a4145ed3",
        "keywords": ["大", "清晰度", "不错", "质量"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "1712bb20fe27c8b3bec5424dc4a71137102ae2684564a423a96f1a9cbd54cbe5",
        "keywords": ["包装", "简陋", "粗糙", "差"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "d25dad3c1260171b23ecc2798ee9d1040a5cbe5e536e6a6d15d9c9d738c497cc",
        "keywords": ["手机", "二手"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "aeb50afb4f6a6240e3eb7949752077d016a4434ef888da3a91a2cbd200fadb6b",
        "keywords": ["返回键", "死机", "售后", "维修", "退货", "质量"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "de08836094947970854225ca4d943778bb6038d811fbacda476c6e0e81a946f7",
        "keywords": ["电池", "不耐用"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "62588f059cbd5842c0d545f30d8cbb36b8401583df96a5be6c67ea50074bf9e5",
        "keywords": ["降价"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "c648a5571a3bcdffdcc649ac2088fa55ae57e24a089be45493116fbff8ff67e3",
        "keywords": ["卡顿", "退货"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "1f24bba8c84a8d10681bbecdf836c79aed98781b64c0f94c09a20a90d905b2b1",
        "keywords": ["面料", "弹力", "手感", "好", "做工", "精细", "贴身", "不错", "物流", "快"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "4d967a8c1cfe020ab624ac2a0a9f743c1e622e2dc46ae9dbd87e6ec99f812df7",
        "keywords": ["忽悠"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "0443b9a584f050e6de95c9c9ac31ce91a28a0dadb7adc7262303d1d1a87b8d63",
        "keywords": ["不錯", "攝像頭", "白點", "不良", "品質"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "原文为繁体中文，关键词保持原文字形",
    },
    {
        "sample_id": "1dfdf06ff20e95da85f7c0f90c4fe6909e15d8e862e9a86384c11cf3b9588fc5",
        "keywords": ["喜欢", "质量", "好", "满意", "发货", "速度", "快", "包装", "仔细", "物流", "服务", "态度"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "c3d310b832371c3d49f2086197f4de4fff6fff8ddc15bdb1b169d3a80cd167b7",
        "keywords": ["口味", "好吃", "配料表", "干净", "放心", "物流", "快", "结实"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "ca255d10c27f36fd3df6b01ea96794ee18703b75e0ce6c7b8fd368423bf0b2d4",
        "keywords": ["屏幕", "垃圾", "碎"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "745dbe4ec287a5471f0ef8f7d28767a6a18fc065612984231511f44df3f62d69",
        "keywords": ["按键", "死键"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "367c85edfe1abaf3ef4b004cdd0abf4c76da7c7a5d37878e39f5c8dfb65fc463",
        "keywords": ["烂", "弯", "质量", "维修", "屏幕", "金属环"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "15b38de5d25530688fd06f513ae05ce582d34bd60aea080af60604e6fc49013c",
        "keywords": ["内存", "卡", "伪造"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "b1f8537218c3d080ee2bdc4b46de166fcf0e48acc2f1f1bed3d64a68bcb0c7ab",
        "keywords": ["价格", "良心", "质感", "喜欢", "速度", "快", "耳机", "音质", "丰富", "饱满", "力度"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "c3e9f74d3e8cff59e28be411c5eb6ad456c64726d0d3d14c127f44184a2829c5",
        "keywords": ["降价", "价格保护", "赠品"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "b6753fbcbd3d7c9761515e789c6790efb1a087fd8fe96bc62cc0676239f17ed0",
        "keywords": ["衣服", "质量", "不错", "合适", "快递", "快", "服务", "态度", "好"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "6e76ac0b5c82500a6c59672647c139379d483e8f0baa5f59ee7dd8877641ea01",
        "keywords": ["质感", "发烫", "阉割"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "3cef3fe0dec0d333d7a9b403053cc7ae4d30aea7a3286733b29fedadc10d731c",
        "keywords": ["垃圾", "屏幕", "不灵", "卡", "内存"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "1c301e26038660d35938ab8afefa1ce3b821b6e78a4ab3d1276b119dd24f6f03",
        "keywords": ["充电", "发烫", "检测"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "ed24d959e8d56075d8a00ec1e3818e60ca0de154566efe91eed21db8a8e2bcee",
        "keywords": ["运行", "良好", "满意", "花屏", "换货"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "0fa33cba8a3e394ab59f7b844a2acc9a8300ab0ece5cbaef07fac070be715bf7",
        "keywords": ["次品", "屏幕", "黑点", "脏"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "452f8cac2b09cfcfcfcaa5ee630b8c08bf0945fc3372452ac3b99e9ce12deaa6",
        "keywords": ["免息", "服务费", "客服", "心寒"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "3f332f919eb2d4bd2a5550eabb899678ab5bfdfd59d6f8185eca62c82f7e1077",
        "keywords": ["屏幕", "花屏"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "9ddb631fd87b1fb64c735e3c922a919562bb23d83378e7b9300310f44b14e269",
        "keywords": ["提升", "质感", "相机", "抖动"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "aabba1086032331583f8b78e15cdcc60cc4976674e4cf3e382e0d4fa0ecd732a",
        "keywords": ["开裂", "镜面屏", "质量", "差", "保修"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "dd98af2a2efc35208935e6a443ea816685bc78060424273e0816048ce620f3f3",
        "keywords": ["质量", "不错", "物流", "快"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "7cacd9f4999ff4fe13899cfb53abf7a442b2e5c8ac4952b30e98d589d4ff1ced",
        "keywords": ["碎"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "文本开头可能截断，信息不完整但仍可提取关键词",
    },
    {
        "sample_id": "331493233e1366ec5e17b0c336077af3deb61e2443f1073380f12560eced1262",
        "keywords": ["烫手", "电池", "不耐用"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "70ba983031d24a7952ba965ad65ea2719eb9a53b3f562f5d7ac13dcd63cbd2c4",
        "keywords": ["物流", "发货", "快", "包装", "好", "服务", "态度", "衣服", "质量", "不错", "舒服"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "35201f48235d708dd1dada442ef8a9f305c9afab4207e496c6d3d3729d708685",
        "keywords": ["客服", "拆封", "不爽"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "c3f9af822eaff01f5ae724f87c236cfea77a59b9ad21d6a6c39b3f2cfb1c7fda",
        "keywords": ["满意", "预定", "发货", "订金", "违约"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "fe35e5d8ac1d23f64fd81d61736f635303cfd03c5aaa4198b8aa50b170abf1b7",
        "keywords": ["屏幕", "瑕疵", "边框", "划痕", "不经用", "指纹键", "坑人"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "302d8a4729f85e35ee21ad8e0d5588fe13a68f7bed26af3e44bfc73620cbb6bc",
        "keywords": ["外观", "不错", "屏幕", "细腻", "舒适", "分辨率", "性能", "满意", "手感", "轻", "客服", "态度", "发货"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "80760499d9e652c6422ca25a292f3524b7cea34f6fada9102d3151093f11e6c0",
        "keywords": ["充电器", "坏"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "1fbd1bec650773df8e87d60996136ad44eac2ea912a2ddad01f334e5582816ab",
        "keywords": ["不怎么好", "音量键", "按不动", "锁屏键"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "9094bf25a57285caf7c89a22de77f1e97eb1eb7e51f8111e2a40afc6055ad944",
        "keywords": ["免息"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "6d7e1e21c33ebecc9bd98af8505335ad896f605818054263bc30291eae720773",
        "keywords": ["店大欺客", "赠品", "客服", "信誉"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "868a2e86db7273af1296119b3ddd5f7ea04e6892715597a731f6580b691996ac",
        "keywords": ["差劲", "退货"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "975c8bbed0b6081b0316c32ed34dcc8ae8fed6429a84910d3d3990be2ace0bd9",
        "keywords": ["黑屏", "售后", "换货", "骗人", "信誉"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "7c75739964fb830ebab08aa1de9ab3725d05ac4f6e27096f5a483a5355bab289",
        "keywords": ["饥饿营销", "抢购", "性价比", "高", "质量"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "16b538338bf71e3544f9fb77369481e52141da1f799e3d7254db8746fe8d9873",
        "keywords": ["漂亮", "大气", "物超所值"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "135f1c6b69c6d14e05d5cf33a832f67fc5048142ccef3fc19e8e238046041985",
        "keywords": ["降价", "价格保护", "失望"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "d27d5e8d3d752b03355bb23c26f83bf8d4daca7bbd3a20538787ec74af8be4cb",
        "keywords": ["真品", "色差", "退货", "客服", "耐心"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "81c4768196c8e470cb72fe1d8d8955ad9761137d4bf6ccb5f26c877833db535e",
        "keywords": ["移动电源"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "1879f8925fd6583b0358e7bc8985a5707d1786c8debfbe6728685539aaf5e5dd",
        "keywords": ["快递", "快", "负责", "手感", "好", "屏幕", "黄"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "3bac1bae89e3f92a094ba44bbcb801581db1dc93b3e9deaf9057ba8b06ea9d0f",
        "keywords": ["质量", "好", "颜值", "高", "满意", "物流", "快", "不错", "价格", "实惠"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "dde259ebe6253240a133ac01f203b862b0c5899f7af12bcd9b52feab8cc75bb3",
        "keywords": ["客服", "序列号", "断", "套路", "质量"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "2ae87beb4e6aa879fe26c2529128dc5a66e840714893f03b385736668c1d6aa2",
        "keywords": ["充电", "质量", "差", "客服", "快充", "慢"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "2ccc63e0cae1b6c73a8f55c8f43c9e5ffb2b7d9310ad59f935d6fe09aa422767",
        "keywords": ["质量", "返回键", "失灵"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "fe83fee0d4780fa83c3fe2c2b3a25c6e2150ba9e2bf569744bdec7309f53feef",
        "keywords": ["品质", "配送", "满意", "外观", "正品", "流畅", "屏幕", "大", "爽", "赠品"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "815a53f6e96e8f35bdbadbf0716967c8c9ebafe63a973bb6e7104f7df39297d3",
        "keywords": ["死机"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "46762c6e657ac7110c76c8b9b339eb309a02c6d86a7b519e1aec730a7ca9758c",
        "keywords": ["屏幕", "坏屏", "装配", "检测", "换货", "赠品", "不错"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "0cb443a513b72ce6759e257ea278d0b0a00cfacb8e82046d245f01fd9b5e1731",
        "keywords": ["到货", "延保", "预定", "坑", "喜欢"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "5300b19ed15058751c427e4f16437b8dfd1d202aa2ad16fb2e890572b9635205",
        "keywords": ["发货", "快", "尺码", "合适"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "72a8e4c29046729c4884b685d6cb0fc618cc832f445e510fc92445883eb2754c",
        "keywords": ["不满意", "外观", "好", "耗电", "充电", "烫手", "内存", "价格", "高"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "b326a86dd61e9ac9d2717f26f24ae097be60e71a2ec52b53c42332a1f338755d",
        "keywords": ["喜欢", "发货", "速度", "快", "服务", "到位", "耐心", "贴心"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "ff00ff0334293562c654a37728cfb329f3020b130b21a9566ffb8c7f0b5c5346",
        "keywords": ["客服", "态度", "好", "喜欢", "质量", "不错", "满意"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "323c51e821b825d314318a0f2162c92fba99e67799b1d24794a700f79f300a64",
        "keywords": ["电视", "好", "质量", "色彩", "安装", "耐心"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "1a14a88cfd43b97480bcd26038b570764dbc764d33eb242e687d3bcec1837a5a",
        "keywords": ["赠品", "退货", "运费", "坑"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "0a392d4c80793c3f1f642c5c3f1cb44df1c7a0a7479f913ae18c1fe06fd39670",
        "keywords": ["口感", "汤料", "好", "信赖"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
    {
        "sample_id": "d5738850e09d4c4a578a369e07c25042453147d0b94f38acb4cb194274ffa91f",
        "keywords": ["发票", "假"],
        "status": "complete",
        "exclude_reason": "",
        "notes": "",
    },
]

ANNOTATOR_ID = "A02"


def main():
    input_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "annotation", "annotation_secondary_original.csv",
    )
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "annotation", "annotation_secondary_v2_annotated.csv",
    )

    with open(input_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    ann_map = {a["sample_id"]: a for a in ANNOTATIONS}

    annotated_rows = []
    for row in rows:
        sid = row["sample_id"]
        ann = ann_map.get(sid)
        if ann is None:
            print(f"WARNING: no annotation for sample_id={sid}")
            annotated_rows.append(row)
            continue

        row["annotator_id"] = ANNOTATOR_ID
        row["keywords_json"] = json.dumps(ann["keywords"], ensure_ascii=False)
        row["annotation_status"] = ann["status"]
        row["exclude_reason"] = ann["exclude_reason"]
        row["notes"] = ann["notes"]
        annotated_rows.append(row)

    fieldnames = [
        "sample_id", "source_text", "annotator_id",
        "keywords_json", "annotation_status", "exclude_reason", "notes",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotated_rows)

    print(f"Done. Annotated {len(annotated_rows)} rows -> {output_path}")

    missing = 0
    for row in annotated_rows:
        if not row.get("annotation_status"):
            missing += 1
    if missing:
        print(f"WARNING: {missing} rows have no annotation_status")
    else:
        print("All rows annotated successfully.")


if __name__ == "__main__":
    main()
