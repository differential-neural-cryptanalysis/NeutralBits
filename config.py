import numpy as np

# 通过修改 VER 来指定分析的对象，如：VER = 0 指分析 Speck32/64
VER = 3

# 定义一个versions列表，用来存储Speck各个成员版本信息
# n:     字长(比特个数), 分组长度为 2 * n
# alpha: 左分支上的移位参数
# beta:  右分支上的移位参数
# m:     密钥长度为 m * n
# T:     总轮数
# dtype: 算法实现时，存储一个字的 numpy 数据类型
# nb:    算法实现时，存储一个字的 numpy 数据类型的字节个数
# knb:   算法实现时，存储主密钥的字节个数
versions = [
  #[ n, alpha, beta,    m,  T,  dtype    , nb,  knb]
   [16,     7,    2,    4,  22, np.uint16,  2,    8], # 0  Speck  32/ 64,
   [24,     8,    3,    3,  22, np.uint32,  4,   12], # 1  Speck  48/ 72,
   [24,     8,    3,    4,  23, np.uint32,  4,   16], # 2  Speck  48/ 96,
   [32,     8,    3,    3,  26, np.uint32,  4,   12], # 3  Speck  64/ 96,
   [32,     8,    3,    4,  27, np.uint32,  4,   16], # 4  Speck  64/128,
   [48,     8,    3,    2,  28, np.uint64,  8,   16], # 5  Speck  96/ 96,
   [48,     8,    3,    3,  29, np.uint64,  8,   24], # 6  Speck  96/144,
   [64,     8,    3,    2,  32, np.uint64,  8,   16], # 7  Speck 128/128,
   [64,     8,    3,    3,  33, np.uint64,  8,   24], # 8  Speck 128/192,
   [64,     8,    3,    4,  34, np.uint64,  8,   32], # 9  Speck 128/256,
]
# Speck 成员 versions[VER] 的字长（比特个数）
WORD_SIZE = versions[VER][0];
# Speck 成员 versions[VER] 的分组长度（比特个数）
BLOCK_SIZE = 2 * WORD_SIZE
# Speck 成员 versions[VER] 的左分支上的移位参数 alpha
ALPHA     = versions[VER][1];
# Speck 成员 versions[VER] 的右分支上的移位参数 beta
BETA      = versions[VER][2];
# Speck 成员 versions[VER] 的字的全1掩码
MASK_VAL  = versions[VER][5](2 ** WORD_SIZE - 1);

# 程序支持一次搜索多个差分路径的中性比特
# 选择哪几个差分路径
START_DIFF_idx  = 0   # 比如从名称为 '32_8020_4101' 的差分路径开始
END_DIFF_idx    = 1   # 比如到名称为 '32_8021_4101' 的差分路径结束 (包含)
DIFF_idx        = range(START_DIFF_idx, END_DIFF_idx + 1)

# 程序支持搜索一个给定差分内部差分的 x，x-1, x-2...轮的中性比特
# 搜索给定一个n轮差分路径的从第 START_FirstRoundIdx 轮开始，START_NRound 轮差分路径的中性比特、START_NRound - 1 轮差分的中性比特、...、END_NRound 轮差分的中性比特，则如下设置：
# 比如：搜索给定一个 3 轮差分路径的从第 0 轮开始，3 轮差分路径的中性比特、2 轮差分的中性比特，则如下设置：
START_NRound = 3
END_NRound   = 2
START_FirstRoundIdx = 0
END_FirstRoundIdx = 0

# 搜索将 trail_base 每轮差分都旋转移位 ROT_LIST 中的参数之后得到的路径的中性比特
ROT_LIST = range(0, WORD_SIZE)

XORN = 3       # 检测同步中性比特集合中同时修改的最大比特个数
GPT  = 0.2     # 比特条件对条件中性比特的中性度影响阈值

# 控制中性率阈值
threshold_freq = 0.8
threshold_freq_low = 0.01

# 使用的线程个数 <= BLOCK_SIZE，且需要整除 BLOCK_SIZE
PN = BLOCK_SIZE // 2

# 如果搜索的是整个差分的中性比特，TEST_TRAIL = False，如果搜索一条差分路径的中性比特，TEST_TRAIL = True
TEST_TRAIL = False

#===============================================
# Set Test Cases
if VER == 0:                         # Speck 32/*
    START_DIFF_idx = 0
    END_DIFF_idx   = 1
    START_NRound   = 3
    END_NRound     = 3
    ROT_LIST       = range(0, WORD_SIZE)   # 可以检测所有可能的旋转移位后的差分路径，及其中性比特
    threshold_freq      = 0.8
    PN = BLOCK_SIZE
elif VER == 1 or VER == 2:           # Speck 48/*
    START_DIFF_idx = 45
    END_DIFF_idx   = 45
    START_NRound   = 3
    END_NRound     = 3
    ROT_LIST = [0, 23]               # 在攻击 Speck 48/* 时可用到的旋转移位差分路径, 可以根据需求进行调整
    threshold_freq      = 0.7
    PN = BLOCK_SIZE
elif VER == 3 or VER == 4:           # Speck 64/*
    START_DIFF_idx = 46
    END_DIFF_idx   = 46
    START_NRound   = 3
    END_NRound     = 3
    ROT_LIST = [1, 10, 15]           # 在攻击 Speck 64/* 时用到的旋转移位差分路径, 可以根据需求进行调整
    threshold_freq      = 0.8
    PN = BLOCK_SIZE
elif VER == 5 or VER == 6:           # Speck 96/*
    START_DIFF_idx = 59
    END_DIFF_idx   = 59
    START_NRound   = 4
    END_NRound     = 4
    ROT_LIST = [5, 17, 29, 41]       # 在攻击 Speck 96/* 时用到的旋转移位差分路径, 可以根据需求进行调整
    threshold_freq      = 0.99
    threshold_freq_low = 0.10
    PN = BLOCK_SIZE // 2
elif VER == 7 or VER == 8 or VER == 9: # Speck 128/*
    START_DIFF_idx = 89
    END_DIFF_idx   = 89
    START_NRound   = 6
    END_NRound     = 6
    ROT_LIST = [0, 12, 26, 41, 53]   # 在攻击 Speck 128/* 时用到的旋转移位差分路径, 可以根据需求进行调整
    threshold_freq      = 0.75
    threshold_freq_low = 0.10
    PN = BLOCK_SIZE // 2
else:
    print("Wrong version number, terminate.")
DIFF_idx            = range(START_DIFF_idx, END_DIFF_idx + 1)
START_FirstRoundIdx = 0
END_FirstRoundIdx   = 0
XORN                = 3
#===============================================


# 已知的一些差分路径的名称，如有新的差分路径可以在这里添加
# 路径的名字可以自己取，需要与get_trails里的路径相对应
DIFFs = [
    '32_8020_4101',                 #  0 在攻击Speck32/64 13轮中用到的路径
    '32_8060_4101',                 #  1 在攻击Speck32/64 13轮中用到的路径
    '32_8021_4101',                 #  2
    '32_8061_4101',                 #  3
    '32_0211_0A04',                 #  4 在攻击Speck32/64 11轮中用到的路径
    '32_0A20_4205',                 #  5
    '32_1488_1008',                 #  6
    '32_3448_7048',                 #  7
    '32_3408_7048',                 #  8
    '32_3428_7048',                 #  9
    '32_3468_7048',                 # 10
    '32_34C8_3048',                 # 11
    '32_80E0_C300',                 # 12
    '32_01E0_C202',                 # 13
    '32_80A0_C100',                 # 14
    '32_4429_4080',                 # 15
    '32_7448_B0F8',                 # 16
    '32_7458_B0F8',                 # 17
    '32_7428_B0F8',                 # 18
    '32_7468_B0F8',                 # 19
    '32_7478_B0F8',                 # 20
    '32_7438_B0F8',                 # 21
    '32_7C48_B0F8',                 # 22
    '32_7C58_B0F8',                 # 23
    '32_7C28_B0F8',                 # 24
    '32_7C78_B0F8',                 # 25
    '32_7C68_B0F8',                 # 26
    '32_7C38_B0F8',                 # 27
    '32_7449_B0F8',                 # 28
    '32_7C49_B0F8',                 # 29
    '32_7459_B0F8',                 # 30
    '32_7C59_B0F8',                 # 31
    '32_6448_B0F8',                 # 32
    '32_6C48_B0F8',                 # 33
    '32_6458_B0F8',                 # 34
    '32_6C58_B0F8',                 # 35
    '32_0448_1068',                 # 36
    '48_484008_000008',             # 37
    '48_0a4058_484a00',             # 38
    '48_0a40d8_484a00',             # 39
    '48_0a4048_484a00',             # 40
    '48_0a40c8_484a00',             # 41
    '48_602901_012128',             # 42
    '48_202901_012128',             # 43
    '48_80b014_940090',             # 44
    '48_809014_940090',             # 45 在攻击Speck48/*中用到的路径
    '64_81248000_00802084',         # 46 在攻击Speck64/*中用到的路径
    '64_200122c0_60082100',         # 47
    '64_20012268_78082100',         # 48
    '64_20012278_78082100',         # 49
    '64_20012240_60082100',         # 50
    '64_20012280_60082100',         # 51
    '64_20012200_60082100',         # 52
    '64_48800800_08401802',         # 53
    '64_48900800_08401802',         # 54
    '64_024400c0_104200c0',         # 55
    '64_024480c0_104200c0',         # 56
    '64_02448040_104200c0',         # 57
    '64_02440040_104200c0',         # 58
    '96_100100141010_901000249410', # 59 在攻击Speck96/*中用到的路径
    '96_0002401c0120_002004400c21', # 60
    '96_0006401c0120_002004400c21', # 61
    '96_0006403c0120_002004400c21', # 62
    '96_000640240120_002004400c21', # 63
    '96_0002403c0120_002004400c21', # 64
    '96_000240240120_002004400c21', # 65
    '96_1200002402c0_c21002004400', # 66
    '96_120000240240_c21002004400', # 67
    '96_1200006402c0_c21002004400', # 68
    '96_120000640240_c21002004400', # 69
    '96_3c01200002c0_400c21002004', # 70
    '96_3c01200006c0_400c21002004', # 71
    '96_3c0120000240_400c21002004', # 72
    '96_1c01200002c0_400c21002004', # 73
    '96_1c0120000240_400c21002004', # 74
    '96_3c0120000640_400c21002004', # 75
    '96_240120000240_400c21002004', # 76
    '96_1c01200006c0_400c21002004', # 77
    '96_2401200006c0_400c21002004', # 78
    '96_240120000640_400c21002004', # 79
    '96_2401200002c0_400c21002004', # 80
    '96_1c0120000640_400c21002004', # 81
    '96_640240120000_004400c21002', # 82
    '96_240240120000_004400c21002', # 83
    '96_6401c0120000_004400c21002', # 84
    '96_6401c0120000_004400c21002', # 85
    '96_2403c0120000_004400c21002', # 86
    '96_6403c0120000_004400c21002', # 87
    'ALLW14',                       # 88
    'LLJW21',                       # 89 在攻击Speck128/*中用到的路径
    'SHY16',                        # 90
    'DMS19',                        # 91
    '128_64',                       # 92
    '128_76',                       # 93
    '128_90',                       # 94
    '128_92',                       # 95
    '128_93',                       # 96
    '128_105',                      # 97
    '128_121',                      # 98
    ]

# 根据差分路径的名称设置差分路径，如有新的差分路径应在这里添加相应的记录
# 这里trail_base从上往下是差分路径从下往上的传播
def get_trails(DIFF):
    trail_base = []
    if VER == 0:
        if DIFF == '32_8020_4101':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0201), np.uint16(0x0604)),
                (np.uint16(0x8020), np.uint16(0x4101)),
                ] # 差分路径，该差分路径可以通过将各轮差分旋转移位i位，得到新的差分路径，因此称此差分路径为 trail_base，后同
        if DIFF == '32_8021_4101':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0201), np.uint16(0x0604)),
                (np.uint16(0x8021), np.uint16(0x4101)),
                ]
        if DIFF == '32_8060_4101':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0201), np.uint16(0x0604)),
                (np.uint16(0x8060), np.uint16(0x4101)),
                ]
        if DIFF == '32_8061_4101':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0201), np.uint16(0x0604)),
                (np.uint16(0x8061), np.uint16(0x4101)),
                ]
        if DIFF == '32_0211_0A04':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x0211), np.uint16(0x0A04)),
                (np.uint16(0x0A60), np.uint16(0x4205)),
                ]
        if DIFF == '32_0A20_4205':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x0211), np.uint16(0x0A04)),
                (np.uint16(0x0A20), np.uint16(0x4205)),
                (np.uint16(0x14AC), np.uint16(0x5209)),
                ]
        if DIFF == '32_0448_1068':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0207), np.uint16(0x0604)),
                (np.uint16(0x80a0), np.uint16(0xc100)),
                (np.uint16(0x0448), np.uint16(0x1068)),
                (np.uint16(0xa000), np.uint16(0x0508)),
                ]
        if DIFF == '32_1488_1008':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0601), np.uint16(0x0604)),
                (np.uint16(0x0021), np.uint16(0x4001)),
                (np.uint16(0x1488), np.uint16(0x1008)),
                ]
        if DIFF == '32_3448_7048':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0201), np.uint16(0x0604)),
                (np.uint16(0x8020), np.uint16(0x4101)),
                (np.uint16(0x3448), np.uint16(0x7048)),
                ]
        if DIFF == '32_3408_7048':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0201), np.uint16(0x0604)),
                (np.uint16(0x8020), np.uint16(0x4101)),
                (np.uint16(0x3408), np.uint16(0x7048)),
                ]
        if DIFF == '32_3428_7048':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0201), np.uint16(0x0604)),
                (np.uint16(0x8020), np.uint16(0x4101)),
                (np.uint16(0x3428), np.uint16(0x7048)),
                ]
        if DIFF == '32_3468_7048':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0201), np.uint16(0x0604)),
                (np.uint16(0x8020), np.uint16(0x4101)),
                (np.uint16(0x3468), np.uint16(0x7048)),
                ]
        if DIFF == '32_34C8_3048':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0201), np.uint16(0x0604)),
                (np.uint16(0x8021), np.uint16(0x4101)),
                (np.uint16(0x34C8), np.uint16(0x3048)),
                ]
        if DIFF == '32_80E0_C300':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x3800), np.uint16(0x0010)),
                (np.uint16(0x0207), np.uint16(0x0E04)),
                (np.uint16(0x80E0), np.uint16(0xC300)),
                (np.uint16(0xF448), np.uint16(0x10F8)),
                ]
        if DIFF == '32_01E0_C202':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7448), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_80A0_C100':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0207), np.uint16(0x0604)),
                (np.uint16(0x80a0), np.uint16(0xc100)),
                (np.uint16(0x0448), np.uint16(0x1068)),
                (np.uint16(0xa000), np.uint16(0x0508)),
                (np.uint16(0xa144), np.uint16(0x2942)),
                ]
        if DIFF == '32_4429_4080':
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x1800), np.uint16(0x0010)),
                (np.uint16(0x0601), np.uint16(0x0604)),
                (np.uint16(0x0025), np.uint16(0x4001)),
                (np.uint16(0x1208), np.uint16(0x1009)),
                (np.uint16(0x4429), np.uint16(0x4080)),
                ]
        #  ***100*011**1000
        #  1011000011111000
        #  0000000111100000
        if DIFF == '32_7448_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7448), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7458_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7458), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7428_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7428), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7468_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7468), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7478_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7478), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7438_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7438), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7C48_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7C48), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7C58_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7C58), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7C28_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7C28), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7C78_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7C78), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7C68_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7C68), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7C38_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7C38), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7449_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7449), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7C49_B0F8': # 011* *100 010* 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7C49), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7459_B0F8': # 011* *100 010* 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7459), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_7C59_B0F8': # 011* *100 010* 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x7C59), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_6448_B0F8': # 011* *100 0*** 100*
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x6448), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_6C48_B0F8': # 0110 *100 010* 1000
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x6C48), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_6458_B0F8': # 0110 *100 0101 1000
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x6458), np.uint16(0xB0F8)),
                ]
        if DIFF == '32_6C58_B0F8': # 0110 1100 0101 1000
            trail_base = [
                (np.uint16(0x0040), np.uint16(0x0000)),
                (np.uint16(0x2800), np.uint16(0x0010)),
                (np.uint16(0x020F), np.uint16(0x0A04)),
                (np.uint16(0x01E0), np.uint16(0xC202)),
                (np.uint16(0x6C58), np.uint16(0xB0F8)),
                ]
    elif VER == 1 or VER == 2:
        ## 3-round, Pr = 2^-11
        #if DIFF == '48_':
        #    input_diff =  ((one << versions[VER][5](1)) ^ (one << versions[VER][5](4)) ^ (one << versions[VER][5](6)) ^ (one << versions[VER][5](22)), (one << versions[VER][5](9)) ^ (one << versions[VER][5](14)) ^ (one << versions[VER][5](20)) ^ (one << versions[VER][5](22)))
        #    output_diff = ((one << versions[VER][5](7)), zeo)
        if DIFF == '48_484008_000008': 
            trail_base = [
                (np.uint32(0x000000), np.uint32(0x200000)),
                (np.uint32(0x000004), np.uint32(0x040000)),
                (np.uint32(0x800480), np.uint32(0x808000)),
                (np.uint32(0x1c1080), np.uint32(0x001090)),
                (np.uint32(0x918200), np.uint32(0x038002)),
                (np.uint32(0x424040), np.uint32(0x524040)),
                (np.uint32(0x404040), np.uint32(0x020000)),
                (np.uint32(0x084848), np.uint32(0x084808)),
                (np.uint32(0x484008), np.uint32(0x000008)),
                ]
        if DIFF == '48_0a4058_484a00':
            trail_base = [
                (np.uint32(0x000004), np.uint32(0x000000)),
                (np.uint32(0x000480), np.uint32(0x800000)),
                (np.uint32(0x041010), np.uint32(0x100090)),
                (np.uint32(0x920002), np.uint32(0x028210)),
                (np.uint32(0x104040), np.uint32(0x521042)),
                (np.uint32(0x0a4058), np.uint32(0x484a00)),
                ]
        if DIFF == '48_0a40d8_484a00':
            trail_base = [
                (np.uint32(0x000004), np.uint32(0x000000)),
                (np.uint32(0x000480), np.uint32(0x800000)),
                (np.uint32(0x041010), np.uint32(0x100090)),
                (np.uint32(0x920002), np.uint32(0x028210)),
                (np.uint32(0x104040), np.uint32(0x521042)),
                (np.uint32(0x0a40d8), np.uint32(0x484a00)),
                ]
        if DIFF == '48_0a4048_484a00':
            trail_base = [
                (np.uint32(0x000004), np.uint32(0x000000)),
                (np.uint32(0x000480), np.uint32(0x800000)),
                (np.uint32(0x041010), np.uint32(0x100090)),
                (np.uint32(0x920002), np.uint32(0x028210)),
                (np.uint32(0x104040), np.uint32(0x521042)),
                (np.uint32(0x0a4048), np.uint32(0x484a00)),
                ]
        if DIFF == '48_0a40c8_484a00':
            trail_base = [
                (np.uint32(0x000004), np.uint32(0x000000)),
                (np.uint32(0x000480), np.uint32(0x800000)),
                (np.uint32(0x041010), np.uint32(0x100090)),
                (np.uint32(0x920002), np.uint32(0x028210)),
                (np.uint32(0x104040), np.uint32(0x521042)),
                (np.uint32(0x0a40c8), np.uint32(0x484a00)),
                ]
        if DIFF == '48_602901_012128':
            trail_base = [
                (np.uint32(0x100000), np.uint32(0x000000)),
                (np.uint32(0x000012), np.uint32(0x020000)),
                (np.uint32(0x401040), np.uint32(0x404002)),
                (np.uint32(0x0a4800), np.uint32(0x400a08)),
                (np.uint32(0x004101), np.uint32(0x094841)),
                (np.uint32(0x602901), np.uint32(0x012128)),
                ]
        if DIFF == '48_202901_012128':
            trail_base = [
                (np.uint32(0x100000), np.uint32(0x000000)),
                (np.uint32(0x000012), np.uint32(0x020000)),
                (np.uint32(0x401040), np.uint32(0x404002)),
                (np.uint32(0x0a4800), np.uint32(0x400a08)),
                (np.uint32(0x004101), np.uint32(0x094841)),
                (np.uint32(0x202901), np.uint32(0x012128)),
                ]
        if DIFF == '48_80b014_940090':
            trail_base = [
                (np.uint32(0x000800), np.uint32(0x000000)),
                (np.uint32(0x090000), np.uint32(0x000100)),
                (np.uint32(0x202008), np.uint32(0x012020)),
                (np.uint32(0x000524), np.uint32(0x042005)),
                (np.uint32(0x808020), np.uint32(0x2084a4)),
                (np.uint32(0x80b014), np.uint32(0x940090)),
                ]
        if DIFF == '48_809014_940090':
            trail_base = [
                (np.uint32(0x000001), np.uint32(0x000000)), #(np.uint32(0x000800), np.uint32(0x000000)),
                (np.uint32(0x000120), np.uint32(0x200000)), #(np.uint32(0x090000), np.uint32(0x000100)),
                (np.uint32(0x010404), np.uint32(0x040024)), #(np.uint32(0x202008), np.uint32(0x012020)),
                (np.uint32(0xa48000), np.uint32(0x00a084)), #(np.uint32(0x000524), np.uint32(0x042005)),
                (np.uint32(0x041010), np.uint32(0x948410)), #(np.uint32(0x808020), np.uint32(0x2084a4)),
                (np.uint32(0x029012), np.uint32(0x121280)), #(np.uint32(0x809014), np.uint32(0x940090)),
                ]
    elif VER == 3 or VER == 4:
        # 3-round, Pr = 2^-12
        #input_diff =  ((one << versions[VER][5](5)) ^ (one << versions[VER][5](21)) ^ (one << versions[VER][5](24)) ^ (one << versions[VER][5](27)) ^ (one << versions[VER][5](30)), (one << versions[VER][5](8)) ^ (one << versions[VER][5](13)) ^ (one << versions[VER][5](19)) ^ (one << versions[VER][5](29)))
        #output_diff = ((one << versions[VER][5](6)), zeo)
        if DIFF == '64_81248000_00802084':
            trail_base = [
                (versions[VER][5](0x00000001), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00000120), versions[VER][5](0x20000000)),
                (versions[VER][5](0x00010404), versions[VER][5](0x04000024)),
                (versions[VER][5](0x81248000), versions[VER][5](0x00802084)),
                ]
        if DIFF == '64_200122c0_60082100':
            trail_base = [
                (versions[VER][5](0x00000002), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00000240), versions[VER][5](0x40000000)),
                (versions[VER][5](0x00020808), versions[VER][5](0x08000048)),
                (versions[VER][5](0x02490001), versions[VER][5](0x01004108)),
                (versions[VER][5](0x20282022), versions[VER][5](0x20692821)),
                (versions[VER][5](0x200122c0), versions[VER][5](0x60082100)),
                ]
        if DIFF == '64_20012268_78082100':
            trail_base = [
                (versions[VER][5](0x00000002), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00000240), versions[VER][5](0x40000000)),
                (versions[VER][5](0x00020808), versions[VER][5](0x08000048)),
                (versions[VER][5](0x02490001), versions[VER][5](0x01004108)),
                (versions[VER][5](0xe0282022), versions[VER][5](0x20692821)),
                (versions[VER][5](0x20012268), versions[VER][5](0x78082100)),
                ]
        if DIFF == '64_20012278_78082100':
            trail_base = [
                (versions[VER][5](0x00000002), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00000240), versions[VER][5](0x40000000)),
                (versions[VER][5](0x00020808), versions[VER][5](0x08000048)),
                (versions[VER][5](0x02490001), versions[VER][5](0x01004108)),
                (versions[VER][5](0xe0282022), versions[VER][5](0x20692821)),
                (versions[VER][5](0x20012278), versions[VER][5](0x78082100)),
                ]
        if DIFF == '64_20012240_60082100':
            trail_base = [
                (versions[VER][5](0x00000002), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00000240), versions[VER][5](0x40000000)),
                (versions[VER][5](0x00020808), versions[VER][5](0x08000048)),
                (versions[VER][5](0x02490001), versions[VER][5](0x01004108)),
                (versions[VER][5](0x20282022), versions[VER][5](0x20692821)),
                (versions[VER][5](0x20012240), versions[VER][5](0x60082100)),
                ]
        if DIFF == '64_20012280_60082100':
            trail_base = [
                (versions[VER][5](0x00000002), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00000240), versions[VER][5](0x40000000)),
                (versions[VER][5](0x00020808), versions[VER][5](0x08000048)),
                (versions[VER][5](0x02490001), versions[VER][5](0x01004108)),
                (versions[VER][5](0x20282022), versions[VER][5](0x20692821)),
                (versions[VER][5](0x20012280), versions[VER][5](0x60082100)),
                ]
        if DIFF == '64_20012200_60082100':
            trail_base = [
                (versions[VER][5](0x00000002), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00000240), versions[VER][5](0x40000000)),
                (versions[VER][5](0x00020808), versions[VER][5](0x08000048)),
                (versions[VER][5](0x02490001), versions[VER][5](0x01004108)),
                (versions[VER][5](0x20282022), versions[VER][5](0x20692821)),
                (versions[VER][5](0x20012200), versions[VER][5](0x60082100)),
                ]
        if DIFF == '64_48800800_08401802':
            trail_base = [
                (versions[VER][5](0x00008000), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00900000), versions[VER][5](0x00001000)),
                (versions[VER][5](0x82020000), versions[VER][5](0x00120200)),
                (versions[VER][5](0x40004092), versions[VER][5](0x10420040)),
                (versions[VER][5](0x0808880a), versions[VER][5](0x4a08481a)),
                (versions[VER][5](0x48800800), versions[VER][5](0x08401802)),
                ]
        if DIFF == '64_48900800_08401802':
            trail_base = [
                (versions[VER][5](0x00008000), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00900000), versions[VER][5](0x00001000)),
                (versions[VER][5](0x82020000), versions[VER][5](0x00120200)),
                (versions[VER][5](0x40004092), versions[VER][5](0x10420040)),
                (versions[VER][5](0x0808880a), versions[VER][5](0x4a08481a)),
                (versions[VER][5](0x48900800), versions[VER][5](0x08401802)),
                ]
        if DIFF == '64_024400c0_104200c0':
            trail_base = [
                (versions[VER][5](0x00000400), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00048000), versions[VER][5](0x00000080)),
                (versions[VER][5](0x04101000), versions[VER][5](0x00009010)),
                (versions[VER][5](0x92000204), versions[VER][5](0x00821002)),
                (versions[VER][5](0x50404440), versions[VER][5](0xd2504240)),
                (versions[VER][5](0x024400c0), versions[VER][5](0x104200c0)),
                ]
        if DIFF == '64_024480c0_104200c0':
            trail_base = [
                (versions[VER][5](0x00000400), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00048000), versions[VER][5](0x00000080)),
                (versions[VER][5](0x04101000), versions[VER][5](0x00009010)),
                (versions[VER][5](0x92000204), versions[VER][5](0x00821002)),
                (versions[VER][5](0x50404440), versions[VER][5](0xd2504240)),
                (versions[VER][5](0x024480c0), versions[VER][5](0x104200c0)),
                ]
        if DIFF == '64_02448040_104200c0':
            trail_base = [
                (versions[VER][5](0x00000400), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00048000), versions[VER][5](0x00000080)),
                (versions[VER][5](0x04101000), versions[VER][5](0x00009010)),
                (versions[VER][5](0x92000204), versions[VER][5](0x00821002)),
                (versions[VER][5](0x50404440), versions[VER][5](0xd2504240)),
                (versions[VER][5](0x02448040), versions[VER][5](0x104200c0)),
                ]
        if DIFF == '64_02440040_104200c0':
            trail_base = [
                (versions[VER][5](0x00000400), versions[VER][5](0x00000000)),
                (versions[VER][5](0x00048000), versions[VER][5](0x00000080)),
                (versions[VER][5](0x04101000), versions[VER][5](0x00009010)),
                (versions[VER][5](0x92000204), versions[VER][5](0x00821002)),
                (versions[VER][5](0x50404440), versions[VER][5](0xd2504240)),
                (versions[VER][5](0x02440040), versions[VER][5](0x104200c0)),
                ]
    elif VER == 5 or VER == 6:
        # 3-round, Pr = 2^-12  
        #input_diff =  ((one << versions[VER][5](6)) ^ (one << versions[VER][5](22)) ^ (one << versions[VER][5](25)) ^ (one << versions[VER][5](28)) ^ (one << versions[VER][5](31)), (one << versions[VER][5](9)) ^ (one << versions[VER][5](14)) ^ (one << versions[VER][5](20)) ^ (one << versions[VER][5](46)))
        #output_diff = ((one << versions[VER][5](7)), zeo)
        if DIFF == '96_100100141010_901000249410':
            trail_base = [
                (versions[VER][5](0x000000000001), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000000000120), versions[VER][5](0x200000000000)),
                (versions[VER][5](0x000000010404), versions[VER][5](0x040000000024)),
                (versions[VER][5](0x800001248000), versions[VER][5](0x008000002084)),
                (versions[VER][5](0x100100141010), versions[VER][5](0x901000249410)),
                ]
        if DIFF == '96_0002401c0120_002004400c21':
            trail_base = [
                (versions[VER][5](0x020000000000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x400000000002), versions[VER][5](0x004000000000)),
                (versions[VER][5](0x080000000208), versions[VER][5](0x480800000000)),
                (versions[VER][5](0x010000024900), versions[VER][5](0x080100000041)),
                (versions[VER][5](0x202002002820), versions[VER][5](0x212020004928)),
                (versions[VER][5](0x0002401c0120), versions[VER][5](0x002004400c21)),
                ]
        if DIFF == '96_0006401c0120_002004400c21':
            trail_base = [
                (versions[VER][5](0x020000000000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x400000000002), versions[VER][5](0x004000000000)),
                (versions[VER][5](0x080000000208), versions[VER][5](0x480800000000)),
                (versions[VER][5](0x010000024900), versions[VER][5](0x080100000041)),
                (versions[VER][5](0x202002002820), versions[VER][5](0x212020004928)),
                (versions[VER][5](0x0006401c0120), versions[VER][5](0x002004400c21)),
                ]
        if DIFF == '96_0006403c0120_002004400c21':
            trail_base = [
                (versions[VER][5](0x020000000000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x400000000002), versions[VER][5](0x004000000000)),
                (versions[VER][5](0x080000000208), versions[VER][5](0x480800000000)),
                (versions[VER][5](0x010000024900), versions[VER][5](0x080100000041)),
                (versions[VER][5](0x202002002820), versions[VER][5](0x212020004928)),
                (versions[VER][5](0x0006403c0120), versions[VER][5](0x002004400c21)),
                ]
        if DIFF == '96_000640240120_002004400c21':
            trail_base = [
                (versions[VER][5](0x020000000000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x400000000002), versions[VER][5](0x004000000000)),
                (versions[VER][5](0x080000000208), versions[VER][5](0x480800000000)),
                (versions[VER][5](0x010000024900), versions[VER][5](0x080100000041)),
                (versions[VER][5](0x202002002820), versions[VER][5](0x212020004928)),
                (versions[VER][5](0x000640240120), versions[VER][5](0x002004400c21)),
                ]
        if DIFF == '96_0002403c0120_002004400c21':
            trail_base = [
                (versions[VER][5](0x020000000000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x400000000002), versions[VER][5](0x004000000000)),
                (versions[VER][5](0x080000000208), versions[VER][5](0x480800000000)),
                (versions[VER][5](0x010000024900), versions[VER][5](0x080100000041)),
                (versions[VER][5](0x202002002820), versions[VER][5](0x212020004928)),
                (versions[VER][5](0x0002403c0120), versions[VER][5](0x002004400c21)),
                ]
        if DIFF == '96_000240240120_002004400c21':
            trail_base = [
                (versions[VER][5](0x020000000000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x400000000002), versions[VER][5](0x004000000000)),
                (versions[VER][5](0x080000000208), versions[VER][5](0x480800000000)),
                (versions[VER][5](0x010000024900), versions[VER][5](0x080100000041)),
                (versions[VER][5](0x202002002820), versions[VER][5](0x212020004928)),
                (versions[VER][5](0x000240240120), versions[VER][5](0x002004400c21)),
                ]
        if DIFF == '96_1200002402c0_c21002004400':
            trail_base = [
                (versions[VER][5](0x000020000000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x002400000000), versions[VER][5](0x000004000000)),
                (versions[VER][5](0x208080000000), versions[VER][5](0x000480800000)),
                (versions[VER][5](0x900010000024), versions[VER][5](0x041080100000)),
                (versions[VER][5](0x820202002002), versions[VER][5](0x928212020004)),
                (versions[VER][5](0x1200002402c0), versions[VER][5](0xc21002004400)),
                ]
        if DIFF == '96_120000240240_c21002004400':
            trail_base = [
                (versions[VER][5](0x000020000000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x002400000000), versions[VER][5](0x000004000000)),
                (versions[VER][5](0x208080000000), versions[VER][5](0x000480800000)),
                (versions[VER][5](0x900010000024), versions[VER][5](0x041080100000)),
                (versions[VER][5](0x820202002002), versions[VER][5](0x928212020004)),
                (versions[VER][5](0x120000240240), versions[VER][5](0xc21002004400)),
                ]
        if DIFF == '96_1200006402c0_c21002004400':
            trail_base = [
                (versions[VER][5](0x000020000000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x002400000000), versions[VER][5](0x000004000000)),
                (versions[VER][5](0x208080000000), versions[VER][5](0x000480800000)),
                (versions[VER][5](0x900010000024), versions[VER][5](0x041080100000)),
                (versions[VER][5](0x820202002002), versions[VER][5](0x928212020004)),
                (versions[VER][5](0x1200006402c0), versions[VER][5](0xc21002004400)),
                ]
        if DIFF == '96_120000640240_c21002004400':
            trail_base = [
                (versions[VER][5](0x000020000000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x002400000000), versions[VER][5](0x000004000000)),
                (versions[VER][5](0x208080000000), versions[VER][5](0x000480800000)),
                (versions[VER][5](0x900010000024), versions[VER][5](0x041080100000)),
                (versions[VER][5](0x820202002002), versions[VER][5](0x928212020004)),
                (versions[VER][5](0x120000640240), versions[VER][5](0xc21002004400)),
                ]
        if DIFF == '96_3c01200002c0_400c21002004':
            trail_base = [
                (versions[VER][5](0x000000020000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000002400000), versions[VER][5](0x000000004000)),
                (versions[VER][5](0x000208080000), versions[VER][5](0x000000480800)),
                (versions[VER][5](0x024900010000), versions[VER][5](0x000041080100)),
                (versions[VER][5](0x002820202002), versions[VER][5](0x004928212020)),
                (versions[VER][5](0x3c01200002c0), versions[VER][5](0x400c21002004)),
                ]
        if DIFF == '96_3c01200006c0_400c21002004':
            trail_base = [
                (versions[VER][5](0x000000020000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000002400000), versions[VER][5](0x000000004000)),
                (versions[VER][5](0x000208080000), versions[VER][5](0x000000480800)),
                (versions[VER][5](0x024900010000), versions[VER][5](0x000041080100)),
                (versions[VER][5](0x002820202002), versions[VER][5](0x004928212020)),
                (versions[VER][5](0x3c01200006c0), versions[VER][5](0x400c21002004)),
                ]
        if DIFF == '96_3c0120000240_400c21002004':
            trail_base = [
                (versions[VER][5](0x000000020000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000002400000), versions[VER][5](0x000000004000)),
                (versions[VER][5](0x000208080000), versions[VER][5](0x000000480800)),
                (versions[VER][5](0x024900010000), versions[VER][5](0x000041080100)),
                (versions[VER][5](0x002820202002), versions[VER][5](0x004928212020)),
                (versions[VER][5](0x3c0120000240), versions[VER][5](0x400c21002004)),
                ]
        if DIFF == '96_1c01200002c0_400c21002004':
            trail_base = [
                (versions[VER][5](0x000000020000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000002400000), versions[VER][5](0x000000004000)),
                (versions[VER][5](0x000208080000), versions[VER][5](0x000000480800)),
                (versions[VER][5](0x024900010000), versions[VER][5](0x000041080100)),
                (versions[VER][5](0x002820202002), versions[VER][5](0x004928212020)),
                (versions[VER][5](0x1c01200002c0), versions[VER][5](0x400c21002004)),
                ]
        if DIFF == '96_1c0120000240_400c21002004':
            trail_base = [
                (versions[VER][5](0x000000020000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000002400000), versions[VER][5](0x000000004000)),
                (versions[VER][5](0x000208080000), versions[VER][5](0x000000480800)),
                (versions[VER][5](0x024900010000), versions[VER][5](0x000041080100)),
                (versions[VER][5](0x002820202002), versions[VER][5](0x004928212020)),
                (versions[VER][5](0x1c0120000240), versions[VER][5](0x400c21002004)),
                ]
        if DIFF == '96_3c0120000640_400c21002004':
            trail_base = [
                (versions[VER][5](0x000000020000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000002400000), versions[VER][5](0x000000004000)),
                (versions[VER][5](0x000208080000), versions[VER][5](0x000000480800)),
                (versions[VER][5](0x024900010000), versions[VER][5](0x000041080100)),
                (versions[VER][5](0x002820202002), versions[VER][5](0x004928212020)),
                (versions[VER][5](0x3c0120000640), versions[VER][5](0x400c21002004)),
                ]
        if DIFF == '96_240120000240_400c21002004':
            trail_base = [
                (versions[VER][5](0x000000020000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000002400000), versions[VER][5](0x000000004000)),
                (versions[VER][5](0x000208080000), versions[VER][5](0x000000480800)),
                (versions[VER][5](0x024900010000), versions[VER][5](0x000041080100)),
                (versions[VER][5](0x002820202002), versions[VER][5](0x004928212020)),
                (versions[VER][5](0x240120000240), versions[VER][5](0x400c21002004)),
                ]
        if DIFF == '96_1c01200006c0_400c21002004':
            trail_base = [
                (versions[VER][5](0x000000020000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000002400000), versions[VER][5](0x000000004000)),
                (versions[VER][5](0x000208080000), versions[VER][5](0x000000480800)),
                (versions[VER][5](0x024900010000), versions[VER][5](0x000041080100)),
                (versions[VER][5](0x002820202002), versions[VER][5](0x004928212020)),
                (versions[VER][5](0x1c01200006c0), versions[VER][5](0x400c21002004)),
                ]
        if DIFF == '96_2401200006c0_400c21002004':
            trail_base = [
                (versions[VER][5](0x000000020000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000002400000), versions[VER][5](0x000000004000)),
                (versions[VER][5](0x000208080000), versions[VER][5](0x000000480800)),
                (versions[VER][5](0x024900010000), versions[VER][5](0x000041080100)),
                (versions[VER][5](0x002820202002), versions[VER][5](0x004928212020)),
                (versions[VER][5](0x2401200006c0), versions[VER][5](0x400c21002004)),
                ]
        if DIFF == '96_240120000640_400c21002004':
            trail_base = [
                (versions[VER][5](0x000000020000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000002400000), versions[VER][5](0x000000004000)),
                (versions[VER][5](0x000208080000), versions[VER][5](0x000000480800)),
                (versions[VER][5](0x024900010000), versions[VER][5](0x000041080100)),
                (versions[VER][5](0x002820202002), versions[VER][5](0x004928212020)),
                (versions[VER][5](0x240120000640), versions[VER][5](0x400c21002004)),
                ]
        if DIFF == '96_2401200002c0_400c21002004':
            trail_base = [
                (versions[VER][5](0x000000020000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000002400000), versions[VER][5](0x000000004000)),
                (versions[VER][5](0x000208080000), versions[VER][5](0x000000480800)),
                (versions[VER][5](0x024900010000), versions[VER][5](0x000041080100)),
                (versions[VER][5](0x002820202002), versions[VER][5](0x004928212020)),
                (versions[VER][5](0x2401200002c0), versions[VER][5](0x400c21002004)),
                ]
        if DIFF == '96_1c0120000640_400c21002004':
            trail_base = [
                (versions[VER][5](0x000000020000), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000002400000), versions[VER][5](0x000000004000)),
                (versions[VER][5](0x000208080000), versions[VER][5](0x000000480800)),
                (versions[VER][5](0x024900010000), versions[VER][5](0x000041080100)),
                (versions[VER][5](0x002820202002), versions[VER][5](0x004928212020)),
                (versions[VER][5](0x1c0120000640), versions[VER][5](0x400c21002004)),
                ]
        if DIFF == '96_640240120000_004400c21002':
            trail_base = [
                (versions[VER][5](0x000000000020), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000000002400), versions[VER][5](0x000000000004)),
                (versions[VER][5](0x000000208080), versions[VER][5](0x800000000480)),
                (versions[VER][5](0x000024900010), versions[VER][5](0x100000041080)),
                (versions[VER][5](0x002002820202), versions[VER][5](0x020004928212)),
                (versions[VER][5](0x640240120000), versions[VER][5](0x004400c21002)),
                ]
        if DIFF == '96_240240120000_004400c21002':
            trail_base = [
                (versions[VER][5](0x000000000020), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000000002400), versions[VER][5](0x000000000004)),
                (versions[VER][5](0x000000208080), versions[VER][5](0x800000000480)),
                (versions[VER][5](0x000024900010), versions[VER][5](0x100000041080)),
                (versions[VER][5](0x002002820202), versions[VER][5](0x020004928212)),
                (versions[VER][5](0x240240120000), versions[VER][5](0x004400c21002)),
                ]
        if DIFF == '96_6401c0120000_004400c21002':
            trail_base = [
                (versions[VER][5](0x000000000020), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000000002400), versions[VER][5](0x000000000004)),
                (versions[VER][5](0x000000208080), versions[VER][5](0x800000000480)),
                (versions[VER][5](0x000024900010), versions[VER][5](0x100000041080)),
                (versions[VER][5](0x002002820202), versions[VER][5](0x020004928212)),
                (versions[VER][5](0x6401c0120000), versions[VER][5](0x004400c21002)),
                ]
        if DIFF == '96_6401c0120000_004400c21002':
            trail_base = [
                (versions[VER][5](0x000000000020), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000000002400), versions[VER][5](0x000000000004)),
                (versions[VER][5](0x000000208080), versions[VER][5](0x800000000480)),
                (versions[VER][5](0x000024900010), versions[VER][5](0x100000041080)),
                (versions[VER][5](0x002002820202), versions[VER][5](0x020004928212)),
                (versions[VER][5](0x2401c0120000), versions[VER][5](0x004400c21002)),
                ]
        if DIFF == '96_2403c0120000_004400c21002':
            trail_base = [
                (versions[VER][5](0x000000000020), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000000002400), versions[VER][5](0x000000000004)),
                (versions[VER][5](0x000000208080), versions[VER][5](0x800000000480)),
                (versions[VER][5](0x000024900010), versions[VER][5](0x100000041080)),
                (versions[VER][5](0x002002820202), versions[VER][5](0x020004928212)),
                (versions[VER][5](0x2403c0120000), versions[VER][5](0x004400c21002)),
                ]
        if DIFF == '96_6403c0120000_004400c21002':
            trail_base = [
                (versions[VER][5](0x000000000020), versions[VER][5](0x000000000000)),
                (versions[VER][5](0x000000002400), versions[VER][5](0x000000000004)),
                (versions[VER][5](0x000000208080), versions[VER][5](0x800000000480)),
                (versions[VER][5](0x000024900010), versions[VER][5](0x100000041080)),
                (versions[VER][5](0x002002820202), versions[VER][5](0x020004928212)),
                (versions[VER][5](0x6403c0120000), versions[VER][5](0x004400c21002)),
                ]
    elif VER == 7 or VER == 8 or VER ==9:
        if DIFF == '128_64':
            trail_base = [
            (np.uint64(0x0000000000000001), np.uint64(0x0000000000000000)),
            (np.uint64(0x0000000000000120), np.uint64(0x2000000000000000)), # 2^-2
            (np.uint64(0x0000000000010404), np.uint64(0x0400000000000024)),
            (np.uint64(0x8000000001e48000), np.uint64(0x0080000000002084)),
            (np.uint64(0x10000000001c1010), np.uint64(0x90100000003c9410)),
            (np.uint64(0x0200000000009000), np.uint64(0x1002000000041080)),
            (np.uint64(0x4040000000001000), np.uint64(0x0240400000009010)),
            ]
        if DIFF == '128_76':
            trail_base = [
            (np.uint64(0x0000000000001000), np.uint64(0x0000000000000000)),
            (np.uint64(0x0000000000120000), np.uint64(0x0000000000000200)),
            (np.uint64(0x0000000010404000), np.uint64(0x0000000000024040)),
            (np.uint64(0x0000001e48000800), np.uint64(0x0000000002084008)),
            (np.uint64(0x00000001c1010100), np.uint64(0x00000003c9410901)),
            (np.uint64(0x0000000009000020), np.uint64(0x2000000041080100)),
            (np.uint64(0x0000000001000404), np.uint64(0x0400000009010024)),
            ]
        if DIFF == '128_90':
            trail_base = [
            (np.uint64(0x0000000004000000), np.uint64(0x0000000000000000)),
            (np.uint64(0x0000000480000000), np.uint64(0x0000000000800000)),
            (np.uint64(0x0000041010000000), np.uint64(0x0000000090100000)),
            (np.uint64(0x0007920002000000), np.uint64(0x0000008210020000)),
            (np.uint64(0x0000704040400000), np.uint64(0x0000f25042404000)),
            (np.uint64(0x0000024000080000), np.uint64(0x0000104200400800)),
            (np.uint64(0x0000004001010000), np.uint64(0x0000024040090100)),
            ]
        if DIFF == '128_92':
            trail_base = [
            (np.uint64(0x0000000010000000), np.uint64(0x0000000000000000)),
            (np.uint64(0x0000001200000000), np.uint64(0x0000000002000000)),
            (np.uint64(0x0000104040000000), np.uint64(0x0000000240400000)),
            (np.uint64(0x001e480008000000), np.uint64(0x0000020840080000)),
            (np.uint64(0x0001c10101000000), np.uint64(0x0003c94109010000)),
            (np.uint64(0x0000090000200000), np.uint64(0x0000410801002000)),
            (np.uint64(0x0000010004040000), np.uint64(0x0000090100240400)),
            ]
        if DIFF == '128_93':
            trail_base = [
            (np.uint64(0x0000000020000000), np.uint64(0x0000000000000000)),
            (np.uint64(0x0000002400000000), np.uint64(0x0000000004000000)),
            (np.uint64(0x0000208080000000), np.uint64(0x0000000480800000)),
            (np.uint64(0x003c900010000000), np.uint64(0x0000041080100000)),
            (np.uint64(0x0003820202000000), np.uint64(0x0007928212020000)),
            (np.uint64(0x0000120000400000), np.uint64(0x0000821002004000)),
            (np.uint64(0x0000020008080000), np.uint64(0x0000120200480800)),
            ]
        if DIFF == '128_105':
            trail_base = [
            (np.uint64(0x0000020000000000), np.uint64(0x0000000000000000)),
            (np.uint64(0x0002400000000000), np.uint64(0x0000004000000000)),
            (np.uint64(0x0208080000000000), np.uint64(0x0000480800000000)),
            (np.uint64(0xc900010000000002), np.uint64(0x0041080100000000)),
            (np.uint64(0x1820202000000200), np.uint64(0x5928212020000000)),
            (np.uint64(0x0120000400024000), np.uint64(0x0821002004000040)),
            (np.uint64(0x0020008082080800), np.uint64(0x0120200480804808)),
            ]
        if DIFF == '128_121':
            trail_base = [
            (np.uint64(0x0200000000000000), np.uint64(0x0000000000000000)),
            (np.uint64(0x4000000000000002), np.uint64(0x0040000000000000)),
            (np.uint64(0x0800000000000208), np.uint64(0x4808000000000000)),
            (np.uint64(0x010000000003c900), np.uint64(0x0801000000000041)),
            (np.uint64(0x2020000000003820), np.uint64(0x2120200000007928)),
            (np.uint64(0x0004000000000120), np.uint64(0x0020040000000821)),
            (np.uint64(0x0080800000000020), np.uint64(0x2004808000000120)),
            ]
        if DIFF == '128_105_2':
            trail_base = [
            (np.uint64(0x0000020000000000), np.uint64(0x0000000000000000)),
            (np.uint64(0x0002400000000000), np.uint64(0x0000004000000000)),
            (np.uint64(0x0208080000000000), np.uint64(0x0000480800000000)),
            (np.uint64(0xc900010000000003), np.uint64(0x0041080100000000)),
            (np.uint64(0x3820202000000000), np.uint64(0x7928212020000000)),
            (np.uint64(0x0120000400000000), np.uint64(0x0821002004000000)),
            (np.uint64(0x0020008080000000), np.uint64(0x0120200480800000)),
            ]
        if DIFF == '128_64_2':
            trail_base = [
            (np.uint64(0x0000000000000001), np.uint64(0x0000000000000000)),
            (np.uint64(0x0000000000000120), np.uint64(0x2000000000000000)),
            (np.uint64(0x0000000000010404), np.uint64(0x0400000000000024)),
            (np.uint64(0x8000000000e48000), np.uint64(0x0080000000002084)),
            (np.uint64(0x10000000001c1010), np.uint64(0x90100000001c9410)),
            (np.uint64(0x0200000004009000), np.uint64(0x1002000000001080)),
            (np.uint64(0x4040000480801000), np.uint64(0x0240400000801010)),
            ]
        if DIFF == '128_64_3':
            trail_base = [
            (np.uint64(0x0000000000000001), np.uint64(0x0000000000000000)),
            (np.uint64(0x0000000000000120), np.uint64(0x2000000000000000)),
            (np.uint64(0x0000000000010404), np.uint64(0x0400000000000024)),
            (np.uint64(0x8000000000e48000), np.uint64(0x0080000000002084)),
            (np.uint64(0x10000000003c1010), np.uint64(0x90100000001c9410)),
            (np.uint64(0x0200000000009000), np.uint64(0x1002000000041080)),
            (np.uint64(0x4040000000001000), np.uint64(0x0240400000009010)),
            ]
        # ====
        # From:  Farzaneh Abed, Eik List, Stefan Lucks, Jakob Wenzel: Differential Cryptanalysis of Round-Reduced Simon and Speck. FSE 2014
        # Trail_base 6r_2012032228080120_202002002a280000_0000000000000000_0000000000000080
        # ====
        if DIFF == 'ALLW14':
            trail_base = [
            (np.uint64(0x0000000000000000), np.uint64(0x0000000000000080)),
            (np.uint64(0x0000000000001000), np.uint64(0x0000000000000010)), # 2^-1
            (np.uint64(0x0000000000120200), np.uint64(0x0000000000000202)), # 2^-3
            (np.uint64(0x0000000010420040), np.uint64(0x4000000000024000)), # 2^-5
            (np.uint64(0x000000124a084808), np.uint64(0x0800000002080808)), # 2^-9
            (np.uint64(0x0000100318400801), np.uint64(0x0100000249000800)), # 2^-12
            (np.uint64(0x2012032228080120), np.uint64(0x202002002a280000)), # 2^-16?
            ]
        # 1-round, Pr = 2^-1
        #input_diff = ((one << versions[VER][5](12)), (one << versions[VER][5](4)))
        # 2-round, Pr = 2^-4 = 2^-{3 + 1}
        #input_diff = ((one << versions[VER][5](9)) ^ (one << versions[VER][5](17)) ^ (one << versions[VER][5](20)), (one << versions[VER][5](1)) ^ (one << versions[VER][5](9)))
        # 3-round, Pr = 2^-9 = 2^-{5 + 3 + 1}
        #input_diff =  ((one << versions[VER][5](6)) ^ (one << versions[VER][5](17)) ^ (one << versions[VER][5](22)) ^ (one << versions[VER][5](28)), (one << versions[VER][5](14)) ^ (one << versions[VER][5](17)) ^ (one << versions[VER][5](62)))
        # 4-round, Pr = 2^-18 = 2^-{9 + 5 + 3 + 1}
        #input_diff  =  ((one << versions[VER][5](3)) ^ (one << versions[VER][5](11)) ^ (one << versions[VER][5](14)) ^ (one << versions[VER][5](19)) ^ (one << versions[VER][5](25)) ^ (one << versions[VER][5](27)) ^ (one << versions[VER][5](30)) ^ (one << versions[VER][5](33)) ^ (one << versions[VER][5](36)), (one << versions[VER][5](3)) ^ (one << versions[VER][5](11)) ^ (one << versions[VER][5](19)) ^ (one << versions[VER][5](25)) ^ (one << versions[VER][5](59)))
        #output_diff = (zeo, (one << versions[VER][5](7)))
        # 5-round, Pr = 2^-30 = 2^-{12 + 9 + 5 + 3 + 1}
        #input_diff  = ((one << versions[VER][5](0)) ^ (one << versions[VER][5](11)) ^ (one << versions[VER][5](22)) ^ (one << versions[VER][5](27)) ^ (one << versions[VER][5](28)) ^ (one << versions[VER][5](32)) ^ (one << versions[VER][5](33)) ^ (one << versions[VER][5](44)), (one << versions[VER][5](11)) ^ (one << versions[VER][5](24)) ^ (one << versions[VER][5](27)) ^ (one << versions[VER][5](30)) ^ (one << versions[VER][5](33)) ^ (one << versions[VER][5](56)));
        #output_diff = ((one << versions[VER][5](3)) ^ (one << versions[VER][5](11)) ^ (one << versions[VER][5](14)) ^ (one << versions[VER][5](19)) ^ (one << versions[VER][5](25)) ^ (one << versions[VER][5](27)) ^ (one << versions[VER][5](30)) ^ (one << versions[VER][5](33)) ^ (one << versions[VER][5](36)), (one << versions[VER][5](3)) ^ (one << versions[VER][5](11)) ^ (one << versions[VER][5](19)) ^ (one << versions[VER][5](25)) ^ (one << versions[VER][5](59))); #(zeo, (one << versions[VER][5](7)))
        #output_diff = (zeo, (one << versions[VER][5](7)))
        # 6-round, Pr = 2^-4? = 2^-{? + 12 + 9 + 5 + 3 + 1}
        #input_diff = ((one << versions[VER][5](5)) ^ (one << versions[VER][5](8)) ^ (one << versions[VER][5](19)) ^ (one << versions[VER][5](27)) ^ (one << versions[VER][5](29)) ^ (one << versions[VER][5](37)) ^ (one << versions[VER][5](40)) ^ (one << versions[VER][5](41)) ^ (one << versions[VER][5](49)) ^ (one << versions[VER][5](52)) ^ (one << versions[VER][5](61)) ^ (one << versions[VER][5](33)), (one << versions[VER][5](19)) ^ (one << versions[VER][5](21)) ^ (one << versions[VER][5](27)) ^ (one << versions[VER][5](29)) ^ (one << versions[VER][5](41)) ^ (one << versions[VER][5](53)) ^ (one << versions[VER][5](61)) ^ (one << versions[VER][5](25)));
        #output_diff = ((one << versions[VER][5](3)) ^ (one << versions[VER][5](11)) ^ (one << versions[VER][5](14)) ^ (one << versions[VER][5](19)) ^ (one << versions[VER][5](25)) ^ (one << versions[VER][5](27)) ^ (one << versions[VER][5](30)) ^ (one << versions[VER][5](33)) ^ (one << versions[VER][5](36)), (one << versions[VER][5](3)) ^ (one << versions[VER][5](11)) ^ (one << versions[VER][5](19)) ^ (one << versions[VER][5](25)) ^ (one << versions[VER][5](59)))
        #output_diff  = ((one << versions[VER][5](0)) ^ (one << versions[VER][5](11)) ^ (one << versions[VER][5](22)) ^ (one << versions[VER][5](27)) ^ (one << versions[VER][5](28)) ^ (one << versions[VER][5](32)) ^ (one << versions[VER][5](33)) ^ (one << versions[VER][5](44)), (one << versions[VER][5](11)) ^ (one << versions[VER][5](24)) ^ (one << versions[VER][5](27)) ^ (one << versions[VER][5](30)) ^ (one << versions[VER][5](33)) ^ (one << versions[VER][5](56)));
        #output_diff = (zeo, (one << versions[VER][5](7)))
        # ====
        # From:  Zhengbin Liu, Yongqiang Li, Lin Jiao, Mingsheng Wang: A New Method for Searching Optimal Differential and Linear Trail_bases in ARX Ciphers. IEEE Trans. Inf. Theory 67(2): 1054-1068 (2021)
        # ====
        if DIFF == 'LLJW21':
            #trail_base = [
            #(np.uint64(0x0000000000000080), np.uint64(0x0000000000000000)),
            #(np.uint64(0x0000000000009000), np.uint64(0x0000000000000010)), # 2^-2
            #(np.uint64(0x0000000000820200), np.uint64(0x0000000000001202)), # 2^-4
            #(np.uint64(0x0000000092400040), np.uint64(0x4000000000104200)), # 2^-6
            #(np.uint64(0x000000800a080808), np.uint64(0x08000000124a0848)), # 2^-10
            #(np.uint64(0x0000900900480001), np.uint64(0x0100001003084008)), # 2^-10
            #(np.uint64(0x20820a2020080020), np.uint64(0x2020120320680801)), # 2^-14
            #(np.uint64(0x9649244004012400), np.uint64(0x20144304600c0104)), # 2^-18
            #]
            #trail_rol = 57
            #for ri in range(len(trail_base)):
            #    trail_base[ri] = (rol(trail_base[ri][0], trail_rol), rol(trail_base[ri][1], trail_rol))
            trail_base = [
            (np.uint64(0x0000000000000001), np.uint64(0x0000000000000000)),
            (np.uint64(0x0000000000000120), np.uint64(0x2000000000000000)), # 2^-2
            (np.uint64(0x0000000000010404), np.uint64(0x0400000000000024)), # 2^-4
            (np.uint64(0x8000000001248000), np.uint64(0x0080000000002084)), # 2^-6
            (np.uint64(0x1000000100141010), np.uint64(0x9010000000249410)), # 2^-10
            (np.uint64(0x0200012012009000), np.uint64(0x1002000020061080)), # 2^-10
            (np.uint64(0x4041041440401000), np.uint64(0x024040240640d010)), # 2^-14
            ]
        # 1-round, Pr = 2^-2
        #input_diff = (np.uint64(0x0000000000009000), np.uint64(0x0000000000000010))
        # 2-round, Pr = 2^-6 = 2^-{4 + 2}  
        #input_diff = (np.uint64(0x0000000000820200), np.uint64(0x0000000000001202))
        # 3-round, Pr = 2^-12 = 2^-{6 + 4 + 2}
        #input_diff = (np.uint64(0x0000000092400040), np.uint64(0x4000000000104200))
        # 4-round, Pr = 2^-22 = 2^-{10 + 6 + 4 + 2}
        #input_diff = (np.uint64(0x000000800a080808), np.uint64(0x08000000124a0848))
        # 5-round, Pr = 2^-32 = 2^-{10 + 10 + 6 + 4 + 2}
        #input_diff = (np.uint64(0x0000900900480001), np.uint64(0x0100001003084008))
        # 6-round, Pr = 2^-46 = 2^-{14 + 10 + 10 + 6 + 4 + 2}
        #input_diff = (np.uint64(0x20820a2020080020), np.uint64(0x2020120320680801))
        # 7-round, Pr = 2^-64 = 2^-{18 + 14 + 10 + 10 + 6 + 4 + 2}
        #input_diff = (np.uint64(0x9649244004012400), np.uint64(0x20144304600c0104))
        #output_diff = (np.uint64(0x20820a2020080020), np.uint64(0x2020120320680801))
        #output_diff = (np.uint64(0x0000900900480001), np.uint64(0x0100001003084008))
        #output_diff = (np.uint64(0x000000800a080808), np.uint64(0x08000000124a0848))
        #output_diff = (np.uint64(0x0000000092400040), np.uint64(0x4000000000104200))
        #output_diff = (np.uint64(0x0000000000820200), np.uint64(0x0000000000001202))
        #output_diff = (np.uint64(0x0000000000009000), np.uint64(0x0000000000000010))
        #output_diff = (np.uint64(0x0000000000000080), np.uint64(0x0000000000000000))
        # ====
        # From:  Ling Song, Zhangjie Huang, Qianqian Yang: Automatic Differential Analysis of ARX Block Ciphers with Application to SPECK and LEA. IACR Cryptol. ePrint Arch. 2016: 209 (2016)
        # ====
        if DIFF == 'SHY16':
            trail_base = [
            (np.uint64(0x0000000000000080), np.uint64(0x0000000000000000)),
            (np.uint64(0x0000000000009000), np.uint64(0x0000000000000010)), # 2^-2
            (np.uint64(0x0000000000820200), np.uint64(0x0000000000001202)), # 2^-4
            (np.uint64(0x00000000F2400040), np.uint64(0x4000000000104200)), # 2^-8
            (np.uint64(0x000000000E080808), np.uint64(0x080000001E4A0848)), # 2^-15
            (np.uint64(0x0000000000480001), np.uint64(0x0100000002084008)), # 2^-8
            (np.uint64(0x2000000000080020), np.uint64(0x2020000000480801)), # 2^-6
            (np.uint64(0x0400000000032400), np.uint64(0x2004000000080104)), # 2^-7
            ]
        # 1-round, Pr = 2^-2
        #input_diff = (np.uint64(0x0000000000009000), np.uint64(0x0000000000000010))
        # 2-round, Pr = 2^-6 = 2^-{4 + 2}  
        #input_diff = (np.uint64(0x0000000000820200), np.uint64(0x0000000000001202))
        # 3-round, Pr = 2^-14 = 2^-{8 + 4 + 2}
        #input_diff =  (np.uint64(0x00000000F2400040), np.uint64(0x4000000000104200))
        # 4-round, Pr = 2^-29 = 2^-{15 + 8 + 4 + 2}
        #input_diff  = (np.uint64(0x000000000E080808), np.uint64(0x080000001E4A0848))
        # 5-round, Pr = 2^-37 = 2^-{8 + 15 + 8 + 4 + 2}
        #input_diff  = (np.uint64(0x0000000000480001), np.uint64(0x0100000002084008))
        # 6-round, Pr = 2^-43 = 2^-{6 + 8 + 15 + 8 + 4 + 2}
        #input_diff = (np.uint64(0x2000000000080020), np.uint64(0x2020000000480801))
        # 7-round, Pr = 2^-50 = 2^-{7 + 6 + 8 + 15 + 8 + 4 + 2}
        #input_diff = (np.uint64(0x0400000000032400), np.uint64(0x2004000000080104))
        #output_diff = (np.uint64(0x2000000000080020), np.uint64(0x2020000000480801))
        #output_diff = (np.uint64(0x0000000000480001), np.uint64(0x0100000002084008))
        #output_diff = (np.uint64(0x000000000E080808), np.uint64(0x080000001E4A0848))
        #output_diff = (np.uint64(0x00000000F2400040), np.uint64(0x4000000000104200))
        #output_diff = (np.uint64(0x0000000000820200), np.uint64(0x0000000000001202))
        #output_diff = (np.uint64(0x0000000000009000), np.uint64(0x0000000000000010))
        #output_diff = (np.uint64(0x0000000000000080), np.uint64(0x0000000000000000))
        # ====
        # From:  Ashutosh Dhar Dwivedi, Pawel Morawiecki, Gautam Srivastava: Differential Cryptanalysis of Round-Reduced SPECK Suitable for Internet of Things Devices. IEEE Access 7: 16476-16486 (2019)
        # ====
        if DIFF == 'DMS19':
            trail_base = [
            (np.uint64(0x8000000000000000), np.uint64(0x0000000000000000)),
            (np.uint64(0x0000000000000090), np.uint64(0x1000000000000000)), # 2^-2
            (np.uint64(0x0000000000008202), np.uint64(0x0200000000000012)), # 2^-4
            (np.uint64(0x4000000000924000), np.uint64(0x0040000000001042)), # 2^-6
            (np.uint64(0x08000000800a0808), np.uint64(0x4808000000124a08)), # 2^-10
            (np.uint64(0x0100009009004800), np.uint64(0x0801000010030840)), # 2^-10
            (np.uint64(0x2020820a20200800), np.uint64(0x0120201203206808)), # 2^-14
            (np.uint64(0x0096492440040124), np.uint64(0x0420144304600c01)), # 2^-18
            ]
        # 1-round, Pr = 2^-2
        #input_diff = (np.uint64(0x0000000000000090), np.uint64(0x1000000000000000))
        # 2-round, Pr = 2^-6 = 2^-{4 + 2}  
        #input_diff = (np.uint64(0x0000000000008202), np.uint64(0x0200000000000012))
        # 3-round, Pr = 2^-12 = 2^-{6 + 4 + 2}
        #input_diff =  (np.uint64(0x4000000000924000), np.uint64(0x0040000000001042))
        # 4-round, Pr = 2^-22 = 2^-{10 + 6 + 4 + 2}
        #input_diff  = (np.uint64(0x08000000800a0808), np.uint64(0x4808000000124a08))
        # 5-round, Pr = 2^-32 = 2^-{10 + 10 + 6 + 4 + 2}
        #input_diff  = (np.uint64(0x0100009009004800), np.uint64(0x0801000010030840))
        # 6-round, Pr = 2^-46 = 2^-{14 + 10 + 10 + 6 + 4 + 2}
        #input_diff = (np.uint64(0x2020820a20200800), np.uint64(0x0120201203206808))
        #output_diff = (np.uint64(0x0100009009004800), np.uint64(0x0801000010030840))
        #output_diff = (np.uint64(0x08000000800a0808), np.uint64(0x4808000000124a08))
        #output_diff = (np.uint64(0x4000000000924000), np.uint64(0x0040000000001042))
        #output_diff = (np.uint64(0x0000000000008202), np.uint64(0x0200000000000012))
        #output_diff = (np.uint64(0x0000000000000090), np.uint64(0x1000000000000000))
        #output_diff = (np.uint64(0x8000000000000000), np.uint64(0x0000000000000000))
    else:
        trail_base = [
            (np.uint16(0x0040), np.uint16(0x0000)),
            (np.uint16(0x1800), np.uint16(0x0010)),
            (np.uint16(0x0201), np.uint16(0x0604)),
            (np.uint16(0x8020), np.uint16(0x4101)),
            (np.uint16(0x3448), np.uint16(0x7048)),
            ]
        #input_diff =  (0x8020, 0x4101)
        #input_diff2 =  (0x8020, 0x4101)
        #output_diff = (0x0040, 0x0000)
    return trail_base

# 定义rol函数，用于循环左移x的k位
def rol(x,k):
    return(((x << versions[VER][5](k)) & MASK_VAL) | (x >> versions[VER][5](WORD_SIZE - k)));
# 定义ror函数，用于循环右移x的k位
def ror(x,k):
    return((x >> versions[VER][5](k)) | ((x << versions[VER][5](WORD_SIZE - k)) & MASK_VAL));