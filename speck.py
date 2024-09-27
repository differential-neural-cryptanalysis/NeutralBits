import numpy as np
from os import urandom
from math import *
from itertools import combinations, chain
from scipy.special import comb
from config import *

# 定义一个加密一轮的函数，用于使用轮密钥k加密p
def one_round_encrypt(p, k):
    c0, c1 = p[0], p[1];
    c0 = ror(c0, ALPHA);
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA);
    c1 = c1 ^ c0;
    return(c0,c1);

# 定义一个解密一轮的函数，用于使用轮密钥k解密c
def one_round_decrypt(c,k):
    c0, c1 = c[0], c[1];
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA);
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA);
    return(c0, c1);

# 定义一个加密一轮的函数，用于使用轮密钥k加密p
def enc_one_round(p, k):
    c0, c1 = p[0], p[1];
    c0 = ror(c0, ALPHA);
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA);
    c1 = c1 ^ c0;
    return(c0,c1);

# 定义一个解密一轮的函数，用于使用轮密钥k解密c
def dec_one_round(c,k):
    c0, c1 = c[0], c[1];
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA);
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA);
    return(c0, c1);

# 定义一个函数，用于扩展密钥
def expand_key(k, t):
    ks = [0 for i in range(t)];
    ks[0] = k[len(k)-1];
    l = list(reversed(k[:len(k)-1]));
    for i in range(t-1):
        l[i%(versions[VER][3]-1)], ks[i+1] = enc_one_round((l[i%(versions[VER][3]-1)], ks[i]), i);
    return(ks);

# 定义一个函数，用于扩展子密钥
def expand_key_subkey(ksa, t):
    ks = [0 for i in range(t)]
    if len(ksa) != versions[VER][3]:
      print("Only support m intermediate subkeys.")
    else:
      l = [0 for i in range(versions[VER][3]-1)]
      for i in range(versions[VER][3] - 2, -1, -1):
          l[i%(versions[VER][3]-1)] = ksa[i + 1] ^ rol(ksa[i], BETA);
          l[i%(versions[VER][3]-1)], tmp = dec_one_round((l[i%(versions[VER][3]-1)], ksa[i + 1]), i);
      ks[0] = ksa[0]
      for i in range(t-1):
          l[i%(versions[VER][3]-1)], ks[i+1] = enc_one_round((l[i%(versions[VER][3]-1)], ks[i]), i);
      return(ks);

# 定义一个函数，用于加密
def encrypt(p, ks):
    x, y = p[0], p[1];
    for k in ks:
        x,y = enc_one_round((x,y), k);
    return(x, y);

# 定义一个函数，用于解密
def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k);
    return(x,y);

def comb_index(n, k):
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)), 
                        int, count=count*k)
    return index.reshape(-1, k)

def GET1(a, i):
    return ((a & (1 << i)) >> i)

def replace_1bit(a, b, i):
    mask = 0xffff ^ (1 << i)
    a = a & mask
    a = a | (b << i)
    return a


# 定义一个函数check_testvector，用于检查测试向量
TEST_VECTORS = [
[(0x1918, 0x1110, 0x0908, 0x0100), (0x6574, 0x694c), (0xa868, 0x42f2)],
[(0x121110, 0x0a0908, 0x020100), (0x20796c, 0x6c6172), (0xc049a5, 0x385adc)],
[(0x1a1918, 0x121110, 0x0a0908, 0x020100), (0x6d2073, 0x696874), (0x735e10, 0xb6445d)],
[(0x13121110, 0x0b0a0908, 0x03020100), (0x74614620, 0x736e6165), (0x9f7952ec, 0x4175946c)],
[(0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100), (0x3b726574, 0x7475432d), (0x8c6fa548, 0x454e028b)],
[(0x0d0c0b0a0908, 0x050403020100), (0x65776f68202c, 0x656761737520), (0x9e4d09ab7178, 0x62bdde8f79aa)],
[(0x151413121110, 0x0d0c0b0a0908, 0x050403020100), (0x656d6974206e, 0x69202c726576), (0x2bf31072228a, 0x7ae440252ee6)],
[(0x0f0e0d0c0b0a0908, 0x0706050403020100), (0x6c61766975716520, 0x7469206564616d20), (0xa65d985179783265, 0x7860fedf5c570d18)],
[(0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100), (0x7261482066656968, 0x43206f7420746e65), (0x1be4cf3a13135566, 0xf9bc185de03c1886)],
[(0x1f1e1d1c1b1a1918, 0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100), (0x65736f6874206e49, 0x202e72656e6f6f70), (0x4109010405c0f53e, 0x4eeeb48d9c188f43)]
]

def check_testvector():
  # 获取测试向量中的key
  key = TEST_VECTORS[VER][0]
  # 获取测试向量中的pt
  pt = TEST_VECTORS[VER][1]
  # 获取key的ks
  ks = expand_key(key, versions[VER][4])
  # 加密pt
  ct = encrypt(pt, ks)
  # 比较加密后的ct和测试向量中的ct
  if (ct == TEST_VECTORS[VER][2]):
    # 如果相等，则返回True
    print("Testvector verified.")
    return(True);
  else:
    # 如果不相等，则返回False
    print("Testvector not verified.")
    return(False);
if __name__ == '__main__':
  check_testvector();
