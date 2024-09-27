#!/bin/bash
from config import *
import speck
from speck import VER
from speck import versions
from os import urandom
import sys, os
import gc
import os.path
from time import time
import numpy as np
import PyPDF2
import pandas as pd
pd.options.display.float_format = '${:,.3f}'.format
from sage.all import *
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy import special, stats
import seaborn as sns
sns.set_style("white")
plt.rc('font', family='serif')
FntSize = 16
FntSize2 = 12
FntSize3 = 11
params = {'axes.labelsize': FntSize,
          'axes.titlesize': FntSize,
          'legend.loc': 'upper left'}
plt.rcParams.update(params)
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FixedLocator, MaxNLocator)
import matplotlib.font_manager as font_manager

font = font_manager.FontProperties(family='serif',
                                   weight='medium',
                                   size=FntSize)
                        
fonttext = {'family': 'serif',
'color':  'black',
'weight': 'medium',
'size': FntSize2,
}
import warnings
warnings.filterwarnings("ignore")

import math
from math import *
import multiprocessing as mp

# 根据Speck的版本选择相应的数据类型
one = (versions[VER][5])(1)
zeo = (versions[VER][5])(0)

keyschedule = 'free'

width4 = versions[VER][0] // 4

# 支持对所有Speck成员的差分搜索条件中性比特，结果写在文件中
# CONDITIONAL = 1 将Speck32的条件中性比特被显示在柱状图中
CONDITIONAL = 0

# 在 BLOCK_SIZE 个比特位置中选取 i 个的组合数，即考虑同步中性比特时，同步 i 个比特，总共有多少个候选的同步i比特集合
XORcN = [special.comb(BLOCK_SIZE, i, exact=True) for i in range(1, XORN+1)]

global DIFF
global end_r
global start_r
global rounds
global tn
global real_tn
global input_diff
global input_diff2
global output_diff
global filepath

def addlabels(x,y,z):
    for i in range(len(x)):
        if y[i] != 0.0:
            plt.text(x[i], y[i], '{:.2f}'.format(y[i]), ha = 'center', fontdict=fonttext, zorder=z)

def addmarks(x,y,m):
    for i in range(len(x)):
        #plt.text(x[i], y[i], m[i], ha = 'center')
        plt.plot([x[i]], [y[i]], color='r', marker=m[i], markersize=15, zorder=XORN+1)

def addmark(x,y,m,c):
    plt.plot(x, y, color=c, marker=m, markersize=15, zorder=XORN+1)

def PDFmerge(pdfs, output):
    pdfMerger = PyPDF2.PdfFileMerger()
    for pdf in pdfs:
        pdfMerger.append(pdf)
    with open(output, 'wb') as f:
        pdfMerger.write(f)

# 将a扩展为比特的列表
def to_bits(a):
    #b = np.unpackbits(np.array([a], dtype='>i8').view(np.uint8), bitorder='big')
    #return b[-WORD_SIZE:]
    b = [0 for i in range(WORD_SIZE)] # 初始化为0
    for i in range(WORD_SIZE):
        bi = (a >> versions[VER][5](i)) & one
        b[WORD_SIZE - 1 - i] = bi
    return b

def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a,b)
        c[c == np.inf] = 0
        c = np.nan_to_num(c)
    return c

def GET1(a, i):
    return ((a & (one << versions[VER][5](i))) >> versions[VER][5](i))

def GET2_XOR(a, i, j):
    return (GET1(a, i) ^ GET1(a, j))

def PUT(x, i, j):
    return ((x << versions[VER][5](i)) | (x << versions[VER][5](j)))

def SWAP(a, i, j):
    return (a ^ PUT(GET2_XOR(a, i, j), i, j))

def replace_1bit(a, b, i):
    mask = MASK_VAL ^ (one << versions[VER][5](i))
    a = a & mask
    a = a | (b << versions[VER][5](i))
    return a

def get_key(rounds):
    key = (np.frombuffer(urandom(versions[VER][7]), dtype=versions[VER][5]).reshape(versions[VER][3], -1)) & MASK_VAL;
    keys = speck.expand_key(key, rounds);
    return keys

def get_plain_pairs(n):
    plain_1 = (np.frombuffer(urandom(versions[VER][6] * n), dtype=versions[VER][5])) & MASK_VAL;
    plain_2 = (np.frombuffer(urandom(versions[VER][6] * n), dtype=versions[VER][5])) & MASK_VAL;
    return plain_1, plain_2

# make a plaintext structure
# takes as input a sequence of plaintexts, a desired plaintext input difference, and a set of neutral bits
# 构造明文结构，对每个输入明文翻转对应的比特，保留原始明文和翻转比特后的明文，同时生成符合输入差分的明文对
def make_plain_structure(
    pt0,
    pt1,
    diff,
    neutral_bits = [[0, 1, 2, 3]]):
    p0 = np.copy(pt0)
    p1 = np.copy(pt1)
    p0 = p0.reshape(-1, 1)
    p1 = p1.reshape(-1, 1)
    for ni in neutral_bits:
        p0_tmp = p0
        p1_tmp = p1
        for bj in range(len(ni)):
            if ni[bj] >= WORD_SIZE:
                d0 = one << versions[VER][5](ni[bj] - WORD_SIZE)
                p0_tmp = p0_tmp ^ d0
            else:
                d1 = one << versions[VER][5](ni[bj]) 
                p1_tmp = p1_tmp ^ d1
        p0 = np.concatenate([p0, p0_tmp], axis = 1)
        p1 = np.concatenate([p1, p1_tmp], axis = 1)
    p0b = p0 ^ diff[0]
    p1b = p1 ^ diff[1]
    return p0, p1, p0b, p1b

# 考虑两条邻接差分路径的切换比特
def make_plain_structure_twodiff(
    pt0,
    pt1,
    diff1,
    diff2,
    neutral_bits = [[0, 1, 2, 3]]):
    p0 = np.copy(pt0)
    p1 = np.copy(pt1)
    p0 = p0.reshape(-1, 1)
    p1 = p1.reshape(-1, 1)
    for ni in neutral_bits:
        p0_tmp = p0
        p1_tmp = p1
        for bj in range(len(ni)):
            if ni[bj] >= WORD_SIZE:
                d0 = one << versions[VER][5](ni[bj] - WORD_SIZE)
                p0_tmp = p0_tmp ^ d0
            else:
                d1 = one << versions[VER][5](ni[bj])
                p1_tmp = p1_tmp ^ d1
        p0 = np.concatenate([p0, p0_tmp], axis = 1)
        p1 = np.concatenate([p1, p1_tmp], axis = 1)
    p0b = np.copy(p0)
    p1b = np.copy(p1)
    sts = 1<<(len(neutral_bits) - 1)
    p0b[:,:sts] = p0b[:,:sts] ^ diff1[0]
    p1b[:,:sts] = p1b[:,:sts] ^ diff1[1]
    p0b[:,sts:] = p0b[:,sts:] ^ diff2[0]
    p1b[:,sts:] = p1b[:,sts:] ^ diff2[1]
    return p0, p1, p0b, p1b

# 考虑两条邻接差分路径的切换比特
def make_plain_structure_twodiff_adjust(
    keys0ci,
    pt0,
    pt1,
    diff1,
    diff2,
    dz,
    neutral_bits = [[0, 1, 2, 3]]):
    dx = ror(diff1[0], ALPHA)
    dy = diff1[1]
    xy_ordered_conds1 = {}
    eq = (dx & dy & dz) | ((dx | dy | dz) ^ MASK_VAL)
    dxy  = dx ^ dy
    dxyz = dxy ^ dz
    dxc1 = dx ^ ror(dxyz, 1)
    cands_xy = []
    all_cands_xy = []
    for bi in range(WORD_SIZE - 2, -1, -1):
        eqi = GET1(eq, bi)
        if eqi == zeo:
            dxyi  = GET1(dxy,  bi)
            if dxyi == zeo: # dx_i ^ dy_i == 0
                # (dx_i = dy_i) => (dz_i = dc_i) because dz_i = dx_i xor dy_i xor dc_i
                # eqi == zeo => dx_i != dz_i => dx_i != dc_i
                # dx_i ^ dc_i = 1 => x_i ^ y_i = dci1 ^ dx_i
                dxc1i = GET1(dxc1, bi)
                cand = ((bi + ALPHA) % WORD_SIZE, bi, dxc1i) # (ix, iy, cnt)
                cands_xy.append(cand)
                all_cands_xy.append(cand)
    cands_xy_vars = list()
    xy_vars_candn = [0 for i in range(WORD_SIZE)]
    xy_vars_cands = [[] for i in range(WORD_SIZE)]
    for ci in range(len(cands_xy)):
        ix = cands_xy[ci][0]
        iy = cands_xy[ci][1]
        xy_vars_candn[ix] += 1; xy_vars_cands[ix].append(ci)
        xy_vars_candn[iy] += 1; xy_vars_cands[iy].append(ci)
        if xy_vars_candn[ix] == 1:
            cands_xy_vars.append(ix)
        if xy_vars_candn[iy] == 1:
            cands_xy_vars.append(iy)
    # detect a circlic constraint, k_i variables in a circlic constraints of x ^ y types 
    # are impossible to appear in carry-type constraints
    # removing any one of the constraints in the circle is OKAY
    cands_xy_vars_tmp = cands_xy_vars.copy()
    for bi in range(len(cands_xy_vars_tmp)):
        old_di = cands_xy_vars_tmp[bi]
        di = old_di
        #print("check circle: di", di)
        intwo = False
        old_ci = xy_vars_cands[di][0]
        while (xy_vars_candn[di] == 2):
            intwo = True
            ci = xy_vars_cands[di][0] + xy_vars_cands[di][1] - old_ci
            old_ci = ci
            #print("check circle: conds_xy ", cands_xy[ci])
            ix = cands_xy[ci][0]
            iy = cands_xy[ci][1]
            si = ix + iy - di
            #print("check circle: si", si)
            di = si
            if di == old_di:
                break
        if (di == old_di) and intwo: # exist circlic constraint, break it
            ci = xy_vars_cands[di][0]
            ix = cands_xy[ci][0]
            iy = cands_xy[ci][1]
            si = ix + iy - di
            #print("circlic constraint, remove ", cands_xy[ci])
            xy_vars_cands[di].remove(ci)
            xy_vars_candn[di] -= 1
            xy_vars_cands[si].remove(ci)
            xy_vars_candn[si] -= 1
    # after breaking circlic constraints, for each groups of constraints, 
    # there must exist one variable appears only in one constraint 
    # the following loop could terminate as long as there is no circlic constraint
    cands_xy_vars_tmp = cands_xy_vars.copy()
    while len(cands_xy_vars_tmp) != 0:
        di = cands_xy_vars_tmp[0]
        cands_xy_vars_tmp.remove(di)
        while xy_vars_candn[di] == 1: # k_di only appears in one condition
            ci = xy_vars_cands[di][0]
            ix = cands_xy[ci][0]
            iy = cands_xy[ci][1]
            cnt = cands_xy[ci][2]
            si = ix + iy - di
            cond = (ix, iy, si, di, cnt)
            xy_ordered_conds1[(ix, iy, si, di)] = cnt
            xy_vars_candn[si] -= 1; xy_vars_cands[si].remove(ci)
            xy_vars_candn[di] -= 1; xy_vars_cands[di].remove(ci)
            if xy_vars_candn[si] == 0:
                cands_xy_vars_tmp.remove(si)
                break
            di = si
        if xy_vars_candn[di] == 2:
            cands_xy_vars_tmp.append(di)
    cands_xy.clear()
    cands_xy_vars.clear()
    xy_vars_candn.clear()
    xy_vars_cands.clear()

    dx = ror(diff2[0], ALPHA)
    dy = diff2[1]
    xy_ordered_conds2 = {}
    eq = (dx & dy & dz) | ((dx | dy | dz) ^ MASK_VAL)
    dxy  = dx ^ dy
    dxyz = dxy ^ dz
    dxc1 = dx ^ ror(dxyz, 1)
    cands_xy = []
    all_cands_xy = []
    for bi in range(WORD_SIZE - 2, -1, -1):
        eqi = GET1(eq, bi)
        if eqi == zeo:
            dxyi  = GET1(dxy,  bi)
            if dxyi == zeo: # dx_i ^ dy_i == 0
                # (dx_i = dy_i) => (dz_i = dc_i) because dz_i = dx_i xor dy_i xor dc_i
                # eqi == zeo => dx_i != dz_i => dx_i != dc_i
                # dx_i ^ dc_i = 1 => x_i ^ y_i = dci1 ^ dx_i
                dxc1i = GET1(dxc1, bi)
                cand = ((bi + ALPHA) % WORD_SIZE, bi, dxc1i) # (ix, iy, cnt)
                cands_xy.append(cand)
                all_cands_xy.append(cand)
    cands_xy_vars = list()
    xy_vars_candn = [0 for i in range(WORD_SIZE)]
    xy_vars_cands = [[] for i in range(WORD_SIZE)]
    for ci in range(len(cands_xy)):
        ix = cands_xy[ci][0]
        iy = cands_xy[ci][1]
        xy_vars_candn[ix] += 1; xy_vars_cands[ix].append(ci)
        xy_vars_candn[iy] += 1; xy_vars_cands[iy].append(ci)
        if xy_vars_candn[ix] == 1:
            cands_xy_vars.append(ix)
        if xy_vars_candn[iy] == 1:
            cands_xy_vars.append(iy)
    # detect a circlic constraint, k_i variables in a circlic constraints of x ^ y types 
    # are impossible to appear in carry-type constraints
    # removing any one of the constraints in the circle is OKAY
    cands_xy_vars_tmp = cands_xy_vars.copy()
    for bi in range(len(cands_xy_vars_tmp)):
        old_di = cands_xy_vars_tmp[bi]
        di = old_di
        #print("check circle: di", di)
        intwo = False
        old_ci = xy_vars_cands[di][0]
        while (xy_vars_candn[di] == 2):
            intwo = True
            ci = xy_vars_cands[di][0] + xy_vars_cands[di][1] - old_ci
            old_ci = ci
            #print("check circle: conds_xy ", cands_xy[ci])
            ix = cands_xy[ci][0]
            iy = cands_xy[ci][1]
            si = ix + iy - di
            #print("check circle: si", si)
            di = si
            if di == old_di:
                break
        if (di == old_di) and intwo: # exist circlic constraint, break it
            ci = xy_vars_cands[di][0]
            ix = cands_xy[ci][0]
            iy = cands_xy[ci][1]
            si = ix + iy - di
            #print("circlic constraint, remove ", cands_xy[ci])
            xy_vars_cands[di].remove(ci)
            xy_vars_candn[di] -= 1
            xy_vars_cands[si].remove(ci)
            xy_vars_candn[si] -= 1
    # after breaking circlic constraints, for each groups of constraints, 
    # there must exist one variable appears only in one constraint 
    # the following loop could terminate as long as there is no circlic constraint
    cands_xy_vars_tmp = cands_xy_vars.copy()
    while len(cands_xy_vars_tmp) != 0:
        di = cands_xy_vars_tmp[0]
        cands_xy_vars_tmp.remove(di)
        while xy_vars_candn[di] == 1: # k_di only appears in one condition
            ci = xy_vars_cands[di][0]
            ix = cands_xy[ci][0]
            iy = cands_xy[ci][1]
            cnt = cands_xy[ci][2]
            si = ix + iy - di
            cond = (ix, iy, si, di, cnt)
            xy_ordered_conds2[(ix, iy, si, di)] = cnt
            xy_vars_candn[si] -= 1; xy_vars_cands[si].remove(ci)
            xy_vars_candn[di] -= 1; xy_vars_cands[di].remove(ci)
            if xy_vars_candn[si] == 0:
                cands_xy_vars_tmp.remove(si)
                break
            di = si
        if xy_vars_candn[di] == 2:
            cands_xy_vars_tmp.append(di)
    cands_xy.clear()
    cands_xy_vars.clear()
    xy_vars_candn.clear()
    xy_vars_cands.clear()
    for ci in xy_ordered_conds1:
        if ci in xy_ordered_conds2:
            if xy_ordered_conds1[ci] != xy_ordered_conds2[ci]:
                set_bit_ix = GET1(pt0, ci[0]) ^ one;
                pt0 = replace_1bit(pt0, set_bit_ix, ci[0])
                #set_bit_iy = GET1(pt1, ci[1]) ^ one;
                #pt1 = replace_1bit(pt1, set_bit_iy, ci[1])

    p0 = np.copy(pt0)
    p1 = np.copy(pt1)
    p0 = p0.reshape(-1, 1)
    p1 = p1.reshape(-1, 1)
    for ni in neutral_bits:
        p0_tmp = p0
        p1_tmp = p1
        for bj in range(len(ni)):
            if ni[bj] >= WORD_SIZE:
                d0 = one << versions[VER][5](ni[bj] - WORD_SIZE)
                p0_tmp = p0_tmp ^ d0
            else:
                d1 = one << versions[VER][5](ni[bj])
                p1_tmp = p1_tmp ^ d1
        p0 = np.concatenate([p0, p0_tmp], axis = 1)
        p1 = np.concatenate([p1, p1_tmp], axis = 1)
    p0b = p0 ^ diff2[0]
    p1b = p1 ^ diff2[1]
    return p0, p1, p0b, p1b

def print_trails(trail_rol):
    global DIFF
    global end_r
    global start_r
    global rounds
    global tn
    global input_diff
    global output_diff
    global filepath
    filename_prefix = filepath + 'print_trails'
    logfile_cur = open(filename_prefix + '.txt', 'a')
    print("Rot: ", trail_rol, file=logfile_cur)
    TotalW = 0
    for this_r in range(end_r-1, start_r-1, -1):
        dx = ror(trail[this_r+1][0], ALPHA)
        dy = trail[this_r+1][1]
        dz = trail[this_r][0]
        eq = (dx & dy & dz) | ((dx | dy | dz) ^ MASK_VAL)
        rnd_weight = 0
        for bi in range(WORD_SIZE - 2, -1, -1):
            eqi = GET1(eq, bi)
            if eqi == zeo:
                rnd_weight += 1
        TotalW += rnd_weight
        if this_r == (end_r - 1):
            print("{2:<2}    {0:0{width}x}    {1:0{width}x}".format(trail[this_r+1][0], trail[this_r+1][1], this_r+1, width=WORD_SIZE//4), file=logfile_cur)
        print("{2:<2}    {0:0{width}x}    {1:0{width}x}    2^-{3:<2}".format(trail[this_r][0], trail[this_r][1], this_r, rnd_weight, width=WORD_SIZE//4), file=logfile_cur)

    print("Total Weight ", TotalW, file=logfile_cur)
    print("\n", file=logfile_cur)
    logfile_cur.close()

# 分析旋转移位之后的差分路径
def analysis_trails(TWeight, trail_rol):
    global DIFF
    global end_r
    global start_r
    global rounds
    global tn
    global input_diff
    global output_diff
    global filepath
    filename_prefix = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_analysis'
    logfile_cur = open(filename_prefix + '.txt', 'w')
    total_weight = 0
    total_cond = 0
    xy_ordered_conds = [[] for i in range(end_r, start_r - 1, -1)]
    xc_ordered_conds = [[None for i in range(WORD_SIZE)] for i in range(end_r, start_r - 1, -1)]
    yc_ordered_conds = [[None for i in range(WORD_SIZE)] for i in range(end_r, start_r - 1, -1)]
    for this_r in range(end_r, start_r, -1):
        if True: #(end_r >= this_r) and (start_r < this_r) and ((keyschedule == 'free') or (end_r - this_r + 1) <= versions[VER][3]):
            print('Round ', this_r, file=logfile_cur)
            dx = ror(trail[this_r][0], ALPHA)
            dy = trail[this_r][1]
            dz = trail[this_r - 1][0]
            print("{0:0{width}b}".format(dx, width=WORD_SIZE), file=logfile_cur)
            print("{0:0{width}b}".format(dy, width=WORD_SIZE), file=logfile_cur)
            print("{0:0{width}b}".format(dz, width=WORD_SIZE), file=logfile_cur)
            eq = (dx & dy & dz) | ((dx | dy | dz) ^ MASK_VAL)
            dxy  = dx ^ dy
            dxyz = dxy ^ dz
            dxc  = dx ^ dxyz     # dz = dx ^ dy ^ dc => dc = dxyz
            dxc1 = dx ^ ror(dxyz, 1)
            dyc1 = dxy ^ dxc1
            cands_xy = []
            cands_xc = []
            cands_yc = []
            all_cands_xy = []
            all_cands_xc = []
            all_cands_yc = []
            # cand (i_x, i_y, i_c, const)
            rnd_weight = 0
            for bi in range(WORD_SIZE - 2, -1, -1):
                eqi = GET1(eq, bi)
                if eqi == zeo:
                    rnd_weight += 1
                    dxyi  = GET1(dxy,  bi)
                    if dxyi == zeo: # dx_i ^ dy_i == 0
                        # (dx_i = dy_i) => (dz_i = dc_i) because dz_i = dx_i xor dy_i xor dc_i
                        # eqi == zeo => dx_i != dz_i => dx_i != dc_i
                        # dx_i ^ dc_i = 1 => x_i ^ y_i = dci1 ^ dx_i
                        dxc1i = GET1(dxc1, bi)
                        cand = ((bi + ALPHA) % WORD_SIZE, bi, dxc1i) # (ix, iy, cnt)
                        cands_xy.append(cand)
                        all_cands_xy.append(cand)
                    else: # dx_i ^ dy_i == 1
                        dxci  = GET1(dxc,  bi) 
                        if dxci == 0: # dx_i ^ dc_i == 0 => x_i ^ c_i = dxc1_i
                            bi_modify = (bi + ALPHA) % WORD_SIZE
                            bi_side_affected = bi_modify
                            dxc1i = GET1(dxc1, bi)
                            cand = (bi_modify, bi, dxc1i) # (ix, ic, cnt)
                            all_cands_xc.append(cand)
                            if bi_side_affected > bi:
                                cands_xc.append(cand)
                        else:  # dx_i ^ dc_i == 1 => y_i ^ c_i = dyc1_i
                            bi_modify = bi
                            bi_side_affected = (bi + WORD_SIZE - ALPHA) % WORD_SIZE
                            dyc1i = GET1(dyc1, bi)
                            cand = (bi_modify, bi, dyc1i) # (iy, ic, cnt)
                            all_cands_yc.append(cand)
                            if bi_side_affected > bi:
                                cands_yc.append(cand)
            print("#all_cands_xy", len(all_cands_xy), file=logfile_cur, flush=True)
            print("#all_cands_xc", len(all_cands_xc), file=logfile_cur, flush=True)
            print("#all_cands_yc", len(all_cands_yc), file=logfile_cur, flush=True)
            for i in range(len(all_cands_xy)):
                print("all_cands_xy", all_cands_xy[i], file=logfile_cur)
            for i in range(len(all_cands_xc)):
                print("all_cands_xc", all_cands_xc[i], file=logfile_cur)
            for i in range(len(all_cands_yc)):
                print("all_cands_yc", all_cands_yc[i], file=logfile_cur)
            #
            cands_xy_vars = list()
            xy_vars_candn = [0 for i in range(WORD_SIZE)]
            xy_vars_cands = [[] for i in range(WORD_SIZE)]
            for ci in range(len(cands_xy)):
                ix = cands_xy[ci][0]
                iy = cands_xy[ci][1]
                xy_vars_candn[ix] += 1; xy_vars_cands[ix].append(ci)
                xy_vars_candn[iy] += 1; xy_vars_cands[iy].append(ci)
                if xy_vars_candn[ix] == 1:
                    cands_xy_vars.append(ix)
                if xy_vars_candn[iy] == 1:
                    cands_xy_vars.append(iy)
            # detect a circlic constraint, k_i variables in a circlic constraints of x ^ y types 
            # are impossible to appear in carry-type constraints
            # removing any one of the constraints in the circle is OKAY
            cands_xy_vars_tmp = cands_xy_vars.copy()
            for bi in range(len(cands_xy_vars_tmp)):
                old_di = cands_xy_vars_tmp[bi]
                di = old_di
                #print("check circle: di", di)
                intwo = False
                old_ci = xy_vars_cands[di][0]
                while (xy_vars_candn[di] == 2):
                    intwo = True
                    ci = xy_vars_cands[di][0] + xy_vars_cands[di][1] - old_ci
                    old_ci = ci
                    #print("check circle: conds_xy ", cands_xy[ci])
                    ix = cands_xy[ci][0]
                    iy = cands_xy[ci][1]
                    si = ix + iy - di
                    #print("check circle: si", si)
                    di = si
                    if di == old_di:
                        break
                if (di == old_di) and intwo: # exist circlic constraint, break it
                    ci = xy_vars_cands[di][0]
                    ix = cands_xy[ci][0]
                    iy = cands_xy[ci][1]
                    si = ix + iy - di
                    #print("circlic constraint, remove ", cands_xy[ci])
                    xy_vars_cands[di].remove(ci)
                    xy_vars_candn[di] -= 1
                    xy_vars_cands[si].remove(ci)
                    xy_vars_candn[si] -= 1
            # after breaking circlic constraints, for each groups of constraints, 
            # there must exist one variable appears only in one constraint 
            # the following loop could terminate as long as there is no circlic constraint
            rnd_cond = 0
            cands_xy_vars_tmp = cands_xy_vars.copy()
            while len(cands_xy_vars_tmp) != 0:
                di = cands_xy_vars_tmp[0]
                cands_xy_vars_tmp.remove(di)
                while xy_vars_candn[di] == 1: # k_di only appears in one condition
                    ci = xy_vars_cands[di][0]
                    ix = cands_xy[ci][0]
                    iy = cands_xy[ci][1]
                    cnt = cands_xy[ci][2]
                    si = ix + iy - di
                    cond = (ix, iy, si, di, cnt)
                    xy_ordered_conds[this_r].append(cond)
                    xy_vars_candn[si] -= 1; xy_vars_cands[si].remove(ci)
                    xy_vars_candn[di] -= 1; xy_vars_cands[di].remove(ci)
                    if xy_vars_candn[si] == 0:
                        cands_xy_vars_tmp.remove(si)
                        break
                    di = si
                if xy_vars_candn[di] == 2:
                    cands_xy_vars_tmp.append(di)
            for ci in range(len(xy_ordered_conds[this_r]) - 1, -1, -1):
                cond = xy_ordered_conds[this_r][ci]
                print("cond xy", cond, file=logfile_cur)
                rnd_cond += 1
            #
            cands_xc_vars = list()
            for ci in range(len(cands_xc)):
                ix = cands_xc[ci][0]
                if (ix in cands_xy_vars) == False:
                    cands_xc_vars.append(ix)
                    xc_ordered_conds[this_r][ix] = cands_xc[ci]
            for ci in range(len(cands_yc)):
                iy = cands_yc[ci][0]
                if ((iy in cands_xy_vars) or (iy in cands_xc_vars)) == False:
                    yc_ordered_conds[this_r][iy] = cands_yc[ci]
            for bi in range(WORD_SIZE - 1):
                if yc_ordered_conds[this_r][bi] != None:
                    print("cond yc", yc_ordered_conds[this_r][bi], file=logfile_cur)
                    rnd_cond += 1
                if xc_ordered_conds[this_r][bi] != None:
                    print("cond xc", xc_ordered_conds[this_r][bi], file=logfile_cur)
                    rnd_cond += 1
            cands_xy.clear()
            cands_xc.clear()
            cands_yc.clear()
            cands_xy_vars.clear()
            xy_vars_candn.clear()
            xy_vars_cands.clear()
            cands_xc_vars.clear()
            total_weight += rnd_weight
            total_cond += rnd_cond
            print("Wt = " + "{:<6}".format(rnd_weight) + "Cd = " + "{:<6}".format(rnd_cond) + "Pr = 2^-" + "{:<6}".format(rnd_weight-rnd_cond), file=logfile_cur, flush=True)
    TWeight[trail_rol] = total_weight
    print("TWt = " + "{:<6}".format(total_weight) + "TCd = " + "{:<6}".format(total_cond) + "TPr = 2^-" + "{:<6}".format(total_weight-total_cond), file=logfile_cur, flush=True)

    logfile_cur.close()

# 生成正确明文和对应的轮密钥
def gen_correctpairs(idx):
    global DIFF
    global end_r
    global start_r
    global rounds
    global tn
    global input_diff
    global output_diff
    global filepath

    # 输出文件名的设置
    filename_prefix = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_proc' + str(idx)
    #logfile_cur = open(filename_prefix + '.txt', 'w')

    # 
    if PN < 16:
        batch_size = 1<<14
    else:
        batch_size = 1<<14 #1<<22
    cnt_correct = 0.0
    prob = 0.0
    correct_plain_l = np.array([], dtype=versions[VER][5])
    correct_plain_r = np.array([], dtype=versions[VER][5])
    correct_key     = np.array([], dtype=versions[VER][5])
    if keyschedule == 'free':
        correct_key     = np.reshape(correct_key, (rounds, -1))
    else:
        correct_key     = np.reshape(correct_key, (versions[VER][3], -1))
    total_weight = 0
    total_cond = 0
    xy_ordered_conds = [[] for i in range(end_r, start_r - 1, -1)]
    xc_ordered_conds = [[None for i in range(WORD_SIZE)] for i in range(end_r, start_r - 1, -1)]
    yc_ordered_conds = [[None for i in range(WORD_SIZE)] for i in range(end_r, start_r - 1, -1)]
    for this_r in range(end_r, start_r, -1):
        if True: #(end_r >= this_r) and (start_r < this_r) and ((keyschedule == 'free') or (end_r - this_r + 1) <= versions[VER][3]):
            #print('Round ', this_r, file=logfile_cur)
            dx = ror(trail[this_r][0], ALPHA)
            dy = trail[this_r][1]
            dz = trail[this_r - 1][0]
            #print("{0:0{width}b}".format(dx, width=WORD_SIZE), file=logfile_cur)
            #print("{0:0{width}b}".format(dy, width=WORD_SIZE), file=logfile_cur)
            #print("{0:0{width}b}".format(dz, width=WORD_SIZE), file=logfile_cur)
            eq = (dx & dy & dz) | ((dx | dy | dz) ^ MASK_VAL)
            dxy  = dx ^ dy
            dxyz = dxy ^ dz
            dxc  = dx ^ dxyz     # dz = dx ^ dy ^ dc => dc = dxyz
            dxc1 = dx ^ ror(dxyz, 1)
            dyc1 = dxy ^ dxc1
            cands_xy = []
            cands_xc = []
            cands_yc = []
            # cand (i_x, i_y, i_c, const)
            rnd_weight = 0
            for bi in range(WORD_SIZE - 2, -1, -1):
                eqi = GET1(eq, bi)
                if eqi == zeo:
                    rnd_weight += 1
                    dxyi  = GET1(dxy,  bi)
                    if dxyi == zeo: # dx_i ^ dy_i == 0
                        # (dx_i = dy_i) => (dz_i = dc_i) because dz_i = dx_i xor dy_i xor dc_i
                        # eqi == zeo => dx_i != dz_i => dx_i != dc_i
                        # dx_i ^ dc_i = 1 => x_i ^ y_i = dci1 ^ dx_i
                        dxc1i = GET1(dxc1, bi)
                        cand = ((bi + ALPHA) % WORD_SIZE, bi, dxc1i) # (ix, iy, cnt)
                        cands_xy.append(cand)
                    else: # dx_i ^ dy_i == 1
                        dxci  = GET1(dxc,  bi) 
                        if dxci == 0: # dx_i ^ dc_i == 0 => x_i ^ c_i = dxc1_i
                            bi_modify = (bi + ALPHA) % WORD_SIZE
                            bi_side_affected = bi_modify
                            if bi_side_affected > bi:
                                dxc1i = GET1(dxc1, bi)
                                cand = (bi_modify, bi, dxc1i) # (ix, ic, cnt)
                                cands_xc.append(cand)
                        else:  # dx_i ^ dc_i == 1 => y_i ^ c_i = dyc1_i
                            bi_modify = bi
                            bi_side_affected = (bi + WORD_SIZE - ALPHA) % WORD_SIZE
                            if bi_side_affected > bi:
                                dyc1i = GET1(dyc1, bi)
                                cand = (bi_modify, bi, dyc1i) # (iy, ic, cnt)
                                cands_yc.append(cand)
            ##print("#cands_xy", len(cands_xy))
            ##print("#cands_xc", len(cands_xc))
            ##print("#cands_yc", len(cands_yc))
            #
            cands_xy_vars = list()
            xy_vars_candn = [0 for i in range(WORD_SIZE)]
            xy_vars_cands = [[] for i in range(WORD_SIZE)]
            for ci in range(len(cands_xy)):
                ix = cands_xy[ci][0]
                iy = cands_xy[ci][1]
                xy_vars_candn[ix] += 1; xy_vars_cands[ix].append(ci)
                xy_vars_candn[iy] += 1; xy_vars_cands[iy].append(ci)
                if xy_vars_candn[ix] == 1:
                    cands_xy_vars.append(ix)
                if xy_vars_candn[iy] == 1:
                    cands_xy_vars.append(iy)
            # detect a circlic constraint, k_i variables in a circlic constraints of x ^ y types 
            # are impossible to appear in carry-type constraints
            # removing any one of the constraints in the circle is OKAY
            cands_xy_vars_tmp = cands_xy_vars.copy()
            for bi in range(len(cands_xy_vars_tmp)):
                old_di = cands_xy_vars_tmp[bi]
                di = old_di
                ##print("check circle: di", di)
                intwo = False
                old_ci = xy_vars_cands[di][0]
                while (xy_vars_candn[di] == 2):
                    intwo = True
                    ci = xy_vars_cands[di][0] + xy_vars_cands[di][1] - old_ci
                    old_ci = ci
                    ##print("check circle: conds_xy ", cands_xy[ci])
                    ix = cands_xy[ci][0]
                    iy = cands_xy[ci][1]
                    si = ix + iy - di
                    ##print("check circle: si", si)
                    di = si
                    if di == old_di:
                        break
                if (di == old_di) and intwo: # exist circlic constraint, break it
                    ci = xy_vars_cands[di][0]
                    ix = cands_xy[ci][0]
                    iy = cands_xy[ci][1]
                    si = ix + iy - di
                    ##print("circlic constraint, remove ", cands_xy[ci])
                    xy_vars_cands[di].remove(ci)
                    xy_vars_candn[di] -= 1
                    xy_vars_cands[si].remove(ci)
                    xy_vars_candn[si] -= 1
            # after breaking circlic constraints, for each groups of constraints, 
            # there must exist one variable appears only in one constraint 
            # the following loop could terminate as long as there is no circlic constraint
            rnd_cond = 0
            cands_xy_vars_tmp = cands_xy_vars.copy()
            while len(cands_xy_vars_tmp) != 0:
                di = cands_xy_vars_tmp[0]
                cands_xy_vars_tmp.remove(di)
                while xy_vars_candn[di] == 1: # k_di only appears in one condition
                    ci = xy_vars_cands[di][0]
                    ix = cands_xy[ci][0]
                    iy = cands_xy[ci][1]
                    cnt = cands_xy[ci][2]
                    si = ix + iy - di
                    cond = (ix, iy, si, di, cnt)
                    xy_ordered_conds[this_r].append(cond)
                    xy_vars_candn[si] -= 1; xy_vars_cands[si].remove(ci)
                    xy_vars_candn[di] -= 1; xy_vars_cands[di].remove(ci)
                    if xy_vars_candn[si] == 0:
                        cands_xy_vars_tmp.remove(si)
                        break
                    di = si
                if xy_vars_candn[di] == 2:
                    cands_xy_vars_tmp.append(di)
            for ci in range(len(xy_ordered_conds[this_r]) - 1, -1, -1):
                cond = xy_ordered_conds[this_r][ci]
                #print("cond xy", cond, file=logfile_cur)
                rnd_cond += 1
            #
            cands_xc_vars = list()
            for ci in range(len(cands_xc)):
                ix = cands_xc[ci][0]
                if (ix in cands_xy_vars) == False:
                    cands_xc_vars.append(ix)
                    xc_ordered_conds[this_r][ix] = cands_xc[ci]
            for ci in range(len(cands_yc)):
                iy = cands_yc[ci][0]
                if ((iy in cands_xy_vars) or (iy in cands_xc_vars)) == False:
                    yc_ordered_conds[this_r][iy] = cands_yc[ci]
            for bi in range(WORD_SIZE - 1):
                if yc_ordered_conds[this_r][bi] != None:
                    #print("cond yc", yc_ordered_conds[this_r][bi], file=logfile_cur)
                    rnd_cond += 1
                if xc_ordered_conds[this_r][bi] != None:
                    #print("cond xc", xc_ordered_conds[this_r][bi], file=logfile_cur)
                    rnd_cond += 1
            cands_xy.clear()
            cands_xc.clear()
            cands_yc.clear()
            cands_xy_vars.clear()
            xy_vars_candn.clear()
            xy_vars_cands.clear()
            cands_xc_vars.clear()
            total_weight += rnd_weight
            total_cond += rnd_cond
            #print("Wt = " + "{:<6}".format(rnd_weight) + "Cd = " + "{:<6}".format(rnd_cond) + "Pr = 2^-" + "{:<6}".format(rnd_weight-rnd_cond), file=logfile_cur, flush=True)
    #print("TWt = " + "{:<6}".format(total_weight) + "TCd = " + "{:<6}".format(total_cond) + "TPr = 2^-" + "{:<6}".format(total_weight-total_cond), file=logfile_cur, flush=True)
    while cnt_correct < tn:
        if (keyschedule == 'free'):
            key = (np.frombuffer(urandom(versions[VER][6] * rounds * batch_size), dtype=versions[VER][5]).reshape(rounds, batch_size)) & MASK_VAL;
        else:
            key = (np.frombuffer(urandom(versions[VER][7] * batch_size), dtype=versions[VER][5]).reshape(versions[VER][3], batch_size)) & MASK_VAL;
        plain_1, plain_2 = get_plain_pairs(batch_size)
        #
        if True:
            cr = 0
            plain_1_cd = np.copy(plain_1)
            plain_2_cd = np.copy(plain_2)
            for this_r in range(end_r, start_r, -1):
                if (end_r >= this_r) and (start_r < this_r) and ((keyschedule == 'free') or (end_r - this_r + 1) <= versions[VER][3]):
                    if end_r != this_r:
                        plain_1_cd, plain_2_cd = plain_1_cd ^ key[cr-1], plain_2_cd ^ key[cr-1]
                        plain_1_cd, plain_2_cd = speck.one_round_encrypt((plain_1_cd, plain_2_cd), zeo)
                    for ci in range(len(xy_ordered_conds[this_r]) - 1, -1, -1):
                        cond = xy_ordered_conds[this_r][ci]
                        ix, iy, si, di, cnt = cond[0], cond[1], cond[2], cond[3], cond[4]
                        set_bit_ix_iy = GET1(plain_1_cd, ix) ^ GET1(plain_2_cd, iy) ^ GET1(key[cr], si) ^ cnt; key[cr] = replace_1bit(key[cr], set_bit_ix_iy, di);
                    #
                    for bi in range(WORD_SIZE - 1):
                        if yc_ordered_conds[this_r][bi] != None:
                            iy, ic, cnt = yc_ordered_conds[this_r][bi][0], yc_ordered_conds[this_r][bi][1], yc_ordered_conds[this_r][bi][2]
                            plain_x, plain_y = plain_1_cd ^ key[cr], plain_2_cd ^ key[cr]
                            plain_x = ror(plain_x, ALPHA)
                            plain_1s2 = (plain_x + plain_y) & MASK_VAL
                            plain_c = plain_1s2 ^ plain_x ^ plain_y
                            set_bit_iy_ic = GET1(plain_2_cd, iy) ^ GET1(plain_c, ic) ^ cnt; key[cr] = replace_1bit(key[cr], set_bit_iy_ic, iy);
                        if xc_ordered_conds[this_r][bi] != None:
                            ix, ic, cnt = xc_ordered_conds[this_r][bi][0], xc_ordered_conds[this_r][bi][1], xc_ordered_conds[this_r][bi][2]
                            plain_x, plain_y = plain_1_cd ^ key[cr], plain_2_cd ^ key[cr]
                            plain_x = ror(plain_x, ALPHA)
                            plain_1s2 = (plain_x + plain_y) & MASK_VAL
                            plain_c = plain_1s2 ^ plain_x ^ plain_y
                            set_bit_ix_ic = GET1(plain_1_cd, ix) ^ GET1(plain_c, ic) ^ cnt; key[cr] = replace_1bit(key[cr], set_bit_ix_ic, ix);
                    cr += 1
            del plain_1_cd
            del plain_2_cd
        if (keyschedule == 'free'):
            keys = key;
        else:
            keys = speck.expand_key_subkey(key, rounds);
            #keys = speck.expand_key(key, rounds)
        #
        plain_11 = plain_1 ^ input_diff[0]
        plain_12 = plain_2 ^ input_diff[1]
        #
        plain_01, plain_02 = speck.one_round_decrypt((plain_1, plain_2), zeo)
        plain_11, plain_12 = speck.one_round_decrypt((plain_11, plain_12), zeo)
        #
        t_01, t_02 = plain_01, plain_02
        t_11, t_12 = plain_11, plain_12
        #
        if TEST_TRAIL:
            diffs0 = []
            diffs1 = []
        for ii in range(rounds):
            t_01, t_02 = speck.one_round_encrypt((t_01, t_02), keys[ii])
            t_11, t_12 = speck.one_round_encrypt((t_11, t_12), keys[ii])
            if TEST_TRAIL:
                diff0 = t_01 ^ t_11
                diff1 = t_02 ^ t_12
                diffs0.append(diff0)
                diffs1.append(diff1)
        diff0 = t_01 ^ t_11
        diff1 = t_02 ^ t_12
        #
        for ci in range(batch_size):
            if TEST_TRAIL:
                correct_flag = True
                for ii in range(rounds):
                    if not (diffs0[ii][ci] == trail[end_r - ii][0] and (diffs1[ii][ci] == trail[end_r - ii][1])):
                        correct_flag = False
                if correct_flag:
                    cnt_correct = cnt_correct + 1.0
                    correct_plain_l = np.concatenate((correct_plain_l, [plain_1[ci]]), axis=0)
                    correct_plain_r = np.concatenate((correct_plain_r, [plain_2[ci]]), axis=0)
                    if keyschedule == 'free':
                        correct_key     = np.concatenate((correct_key    , np.reshape(key[:, ci], (rounds, -1))), axis=1)
                    else:
                        correct_key     = np.concatenate((correct_key    , np.reshape(key[:, ci], (versions[VER][3], -1))), axis=1)
            if (TEST_TRAIL == False) and (diff0[ci] == output_diff[0] and (diff1[ci] == output_diff[1])):
                cnt_correct = cnt_correct + 1.0
                correct_plain_l = np.concatenate((correct_plain_l, [plain_1[ci]]), axis=0)
                correct_plain_r = np.concatenate((correct_plain_r, [plain_2[ci]]), axis=0)
                if keyschedule == 'free':
                    correct_key     = np.concatenate((correct_key    , np.reshape(key[:, ci], (rounds, -1))), axis=1)
                else:
                    correct_key     = np.concatenate((correct_key    , np.reshape(key[:, ci], (versions[VER][3], -1))), axis=1)
        prob = prob + batch_size
        del plain_1
        del plain_2
        del key
        del keys
        gc.collect()
    # 存储正确明文和轮密钥
    np.save(filename_prefix + '_CorPl.npy', correct_plain_l[:tn])
    np.save(filename_prefix + '_CorPr.npy', correct_plain_r[:tn])
    np.save(filename_prefix + '_CorK.npy' , correct_key[:,:tn])
    #print("Find correct pairs: ", len(correct_plain_l[:tn]), file=logfile_cur, flush=True)
    np.set_printoptions(formatter=dict(int=lambda x: '{:{fill}3d}'.format(x, fill=' ')))
    #print("Find right pair after about 2^", '{:0.6f}'.format(np.log2(prob/cnt_correct)), " trails ", file=logfile_cur, flush=True)
    #logfile_cur.close()
    del correct_plain_l
    del correct_plain_r
    del correct_key
    gc.collect()

# 搜索中性比特及其概率
def searching_neutralbits(idx):
    global DIFF
    global end_r
    global start_r
    global rounds
    global tn
    global real_tn
    global input_diff
    global input_diff2
    global output_diff
    global filepath
    global XORN
    global threshold_freq
    global threshold_freq_low
    #
    cnt_correct = 0.0 # 正确明文的数量
    filename_prefix_all = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_proc'
    # 读取生成的符合明文
    plain_1 = np.load(filename_prefix_all + '_CorPl.npy')
    plain_2 = np.load(filename_prefix_all + '_CorPr.npy')
    key = np.load(filename_prefix_all + '_CorK.npy')
    cnt_correct = len(plain_1)
    if keyschedule == 'free':
        keys = key
    else:
        keys = speck.expand_key_subkey(key, rounds)
    keys_st = np.copy(keys)
    keys_st = np.reshape(keys_st, (rounds, cnt_correct, -1))
    keys_st = np.repeat(keys_st, 2, axis=2)

    logfile_allc = open(filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_allc_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc' + str(idx) + '.txt', 'a+')
    logfile_alls = open(filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_alls_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc' + str(idx) + '.txt', 'a+')
    logfile_cnds = open(filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_cnds_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc' + str(idx) + '.txt', 'a+')
    logfile_1xor = open(filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_1xor_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc' + str(idx) + '.txt', 'a+')

    # 输出对齐标志位
    np.set_printoptions(formatter={'float': '{:8.3f}'.format, 'int': '{:8d}'.format}, linewidth=np.inf)
    print(
    "  Pr_x_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), 
    "  Pr_x_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), 
    "  Pr_y_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), 
    "  Pr_y_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), 
    "  Pr_xn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), 
    "  Pr_yn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_yn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), 
    "  Pr_xa_p_y_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), 
    "  Pr_xa_p_y_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), 
    "  Pr_xa_x_c_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xa_x_c_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), 
    "  Pr_y_x_c_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_y_x_c_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), 
    "  Pr_c_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_c_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), 
    file=logfile_allc, flush=True)
    print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array([-1]), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(0.0), 
    "  Pr_x_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_x_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_x_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_x_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_y_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_y_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xn1_nt_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xn1_nt_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_yn1_nt_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_yn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_yn1_nt_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_yn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xa_p_y_nt_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_p_y_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xa_p_y_nt_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_p_y_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xa_x_c_nt_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_x_c_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xa_x_c_nt_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_x_c_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_y_x_c_nt_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_x_c_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_y_x_c_nt_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_x_c_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_c_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_c_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_c_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_c_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    file=logfile_alls, flush=True)
    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array([-1]), separator=', ')), " neutral freq: "                                      ,'{:0.3f}'.format(0.0), ' from ', '{:0.3f}'.format(0.0), 
    "                  " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    file=logfile_cnds, flush=True)

    global AND_plain_x_all        # AND_plain_x_all[i]      是 x                               的第 i 比特是 1 的正确对的个数
    global AND_plain_y_all        # AND_plain_y_all[i]      是 y                               的第 i 比特是 1 的正确对的个数
    global AND_plain_xn1_all      # AND_plain_xn1_all[i]    是 (x>>>alpha) ⊕ ((x>>>alpha)<<1) 的第 i 比特是 1 的正确对的个数
    global AND_plain_yn1_all      # AND_plain_yn1_all[i]    是 y ⊕ (y<<1)                     的第 i 比特是 1 的正确对的个数
    global AND_plain_x7_p_y_all   # AND_plain_x7_p_y_all[i] 是 (x>>>alpha) ⊕ y                的第 i 比特是 1 的正确对的个数
    global AND_xa_x_c_all         # AND_xa_x_c_all[i]       是 (x>>>alpha) ⊕ carry            的第 i 比特是 1 的正确对的个数
    global AND_y_x_c_all          # AND_y_x_c_all[i]        是 y ⊕ carry                      的第 i 比特是 1 的正确对的个数
    global AND_c_all              # AND_c_all[i]            是 carry                           的第 i 比特是 1 的正确对的个数
    global Full_1xor              # Full_1xor[候选比特集索引 NBi_1xor]/cnt_correct = 第 NBi_1xor 个候选比特集是中性的次数
    global AND_plain_x_1xor       # AND_plain_x_1xor      [NBi_1xor, i] 是第 NBi_1xor 个候选比特集对其体现中性且其 x                               的第 i 比特是 1 的正确对的个数
    global AND_plain_y_1xor       # AND_plain_y_1xor      [NBi_1xor, i] 是第 NBi_1xor 个候选比特集对其体现中性且其 y                               的第 i 比特是 1 的正确对的个数
    global AND_plain_xn1_1xor     # AND_plain_xn1_1xor    [NBi_1xor, i] 是第 NBi_1xor 个候选比特集对其体现中性且其 (x>>>alpha) ⊕ ((x>>>alpha)<<1) 的第 i 比特是 1 的正确对的个数
    global AND_plain_yn1_1xor     # AND_plain_yn1_1xor    [NBi_1xor, i] 是第 NBi_1xor 个候选比特集对其体现中性且其 y ⊕ (y<<1)                     的第 i 比特是 1 的正确对的个数
    global AND_plain_x7_p_y_1xor  # AND_plain_x7_p_y_1xor [NBi_1xor, i] 是第 NBi_1xor 个候选比特集对其体现中性且其 (x>>>alpha) ⊕ y                的第 i 比特是 1 的正确对的个数
    global AND_xa_x_c_1xor        # AND_xa_x_c_1xor       [NBi_1xor, i] 是第 NBi_1xor 个候选比特集对其体现中性且其 (x>>>alpha) ⊕ carry            的第 i 比特是 1 的正确对的个数
    global AND_y_x_c_1xor         # AND_y_x_c_1xor        [NBi_1xor, i] 是第 NBi_1xor 个候选比特集对其体现中性且其 y ⊕ carry                      的第 i 比特是 1 的正确对的个数
    global AND_c_1xor             # AND_c_1xor            [NBi_1xor, i] 是第 NBi_1xor 个候选比特集对其体现中性且其 carry                           的第 i 比特是 1 的正确对的个数
    AND_plain_x_all          .fill(0.0)
    AND_plain_y_all          .fill(0.0)
    AND_plain_xn1_all        .fill(0.0)
    AND_plain_yn1_all        .fill(0.0)
    AND_plain_x7_p_y_all     .fill(0.0)
    AND_xa_x_c_all           .fill(0.0)
    AND_y_x_c_all            .fill(0.0)
    AND_c_all                .fill(0.0)
    Full_1xor                .fill(0.0)
    AND_plain_x_1xor         .fill(0.0)
    AND_plain_y_1xor         .fill(0.0)
    AND_plain_xn1_1xor       .fill(0.0)
    AND_plain_yn1_1xor       .fill(0.0)
    AND_plain_x7_p_y_1xor    .fill(0.0)
    AND_xa_x_c_1xor          .fill(0.0)
    AND_y_x_c_1xor           .fill(0.0)
    AND_c_1xor               .fill(0.0)
    # 根据测试的中性比特个数，分别输出对齐标志位，以及初始化变量
    print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array([-1]), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(0.0), 
    "  Pr_x_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_x_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_x_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_x_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_y_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_y_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xn1_nt_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xn1_nt_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_yn1_nt_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_yn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_yn1_nt_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_yn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xa_p_y_nt_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_p_y_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xa_p_y_nt_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_p_y_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xa_x_c_nt_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_x_c_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_xa_x_c_nt_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_x_c_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_y_x_c_nt_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_x_c_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_y_x_c_nt_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_x_c_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_c_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_c_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    "  Pr_c_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_c_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
    file=logfile_1xor, flush=True)
    if XORN >= 2:
        logfile_2xor             = open(filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_2xor_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc' + str(idx) + '.txt', 'a+')
        global Full_2xor            
        global AND_plain_x_2xor     
        global AND_plain_y_2xor     
        global AND_plain_xn1_2xor   
        global AND_plain_yn1_2xor   
        global AND_plain_x7_p_y_2xor
        global AND_xa_x_c_2xor      
        global AND_y_x_c_2xor       
        global AND_c_2xor           
        Full_2xor                .fill(0.0)
        AND_plain_x_2xor         .fill(0.0)
        AND_plain_y_2xor         .fill(0.0)
        AND_plain_xn1_2xor       .fill(0.0)
        AND_plain_yn1_2xor       .fill(0.0)
        AND_plain_x7_p_y_2xor    .fill(0.0)
        AND_xa_x_c_2xor          .fill(0.0)
        AND_y_x_c_2xor           .fill(0.0)
        AND_c_2xor               .fill(0.0)
        print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array([-1]), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(0.0), 
        "  Pr_x_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_x_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_x_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_x_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xn1_nt_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xn1_nt_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_yn1_nt_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_yn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_yn1_nt_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_yn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_p_y_nt_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_p_y_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_p_y_nt_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_p_y_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_x_c_nt_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_x_c_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_x_c_nt_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_x_c_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_x_c_nt_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_x_c_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_x_c_nt_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_x_c_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_c_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_c_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_c_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_c_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        file=logfile_2xor, flush=True)
    if XORN >= 3:
        logfile_3xor             = open(filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_3xor_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc' + str(idx) + '.txt', 'a+')
        global Full_3xor            
        global AND_plain_x_3xor     
        global AND_plain_y_3xor     
        global AND_plain_xn1_3xor   
        global AND_plain_yn1_3xor   
        global AND_plain_x7_p_y_3xor
        global AND_xa_x_c_3xor      
        global AND_y_x_c_3xor       
        global AND_c_3xor           
        Full_3xor                .fill(0.0)
        AND_plain_x_3xor         .fill(0.0)
        AND_plain_y_3xor         .fill(0.0)
        AND_plain_xn1_3xor       .fill(0.0)
        AND_plain_yn1_3xor       .fill(0.0)
        AND_plain_x7_p_y_3xor    .fill(0.0)
        AND_xa_x_c_3xor          .fill(0.0)
        AND_y_x_c_3xor           .fill(0.0)
        AND_c_3xor               .fill(0.0)
        print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array([-1]), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(0.0), 
        "  Pr_x_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_x_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_x_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_x_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xn1_nt_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xn1_nt_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_yn1_nt_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_yn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_yn1_nt_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_yn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_p_y_nt_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_p_y_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_p_y_nt_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_p_y_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_x_c_nt_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_x_c_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_x_c_nt_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_x_c_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_x_c_nt_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_x_c_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_x_c_nt_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_x_c_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_c_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_c_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_c_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_c_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        file=logfile_3xor, flush=True)
    if XORN >= 4:
        logfile_4xor             = open(filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_4xor_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc' + str(idx) + '.txt', 'a+')
        global Full_4xor            
        global AND_plain_x_4xor     
        global AND_plain_y_4xor     
        global AND_plain_xn1_4xor   
        global AND_plain_yn1_4xor   
        global AND_plain_x7_p_y_4xor
        global AND_xa_x_c_4xor      
        global AND_y_x_c_4xor       
        global AND_c_4xor           
        Full_4xor                .fill(0.0)
        AND_plain_x_4xor         .fill(0.0)
        AND_plain_y_4xor         .fill(0.0)
        AND_plain_xn1_4xor       .fill(0.0)
        AND_plain_yn1_4xor       .fill(0.0)
        AND_plain_x7_p_y_4xor    .fill(0.0)
        AND_xa_x_c_4xor          .fill(0.0)
        AND_y_x_c_4xor           .fill(0.0)
        AND_c_4xor               .fill(0.0)
        print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array([-1]), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(0.0), 
        "  Pr_x_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_x_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_x_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_x_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xn1_nt_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xn1_nt_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_yn1_nt_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_yn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_yn1_nt_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_yn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_p_y_nt_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_p_y_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_p_y_nt_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_p_y_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_x_c_nt_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_x_c_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_x_c_nt_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_x_c_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_x_c_nt_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_x_c_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_x_c_nt_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_x_c_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_c_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_c_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_c_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_c_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        file=logfile_4xor, flush=True)
    if XORN >= 5:
        logfile_5xor             = open(filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_5xor_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc' + str(idx) + '.txt', 'a+')
        global Full_5xor            
        global AND_plain_x_5xor     
        global AND_plain_y_5xor     
        global AND_plain_xn1_5xor   
        global AND_plain_yn1_5xor   
        global AND_plain_x7_p_y_5xor
        global AND_xa_x_c_5xor      
        global AND_y_x_c_5xor       
        global AND_c_5xor           
        Full_5xor                .fill(0.0)
        AND_plain_x_5xor         .fill(0.0)
        AND_plain_y_5xor         .fill(0.0)
        AND_plain_xn1_5xor       .fill(0.0)
        AND_plain_yn1_5xor       .fill(0.0)
        AND_plain_x7_p_y_5xor    .fill(0.0)
        AND_xa_x_c_5xor          .fill(0.0)
        AND_y_x_c_5xor           .fill(0.0)
        AND_c_5xor               .fill(0.0)
        print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array([-1]), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(0.0), 
        "  Pr_x_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_x_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_x_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_x_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xn1_nt_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xn1_nt_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_yn1_nt_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_yn1_eq1 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_yn1_nt_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_yn1_eq0 "   , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_p_y_nt_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_p_y_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_p_y_nt_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_p_y_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_x_c_nt_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_x_c_eq1 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_xa_x_c_nt_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_xa_x_c_eq0 ", np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_x_c_nt_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_x_c_eq1 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_y_x_c_nt_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_y_x_c_eq0 " , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_c_nt_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_c_eq1 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        "  Pr_c_nt_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int), "  Pr_nt_c_eq0 "     , np.arange(WORD_SIZE - 1, -1, -1, dtype=int),
        file=logfile_5xor, flush=True)
    # 中性比特的索引
    NBi_1xor = 0
    NBi_2xor = 0
    NBi_3xor = 0
    NBi_4xor = 0
    NBi_5xor = 0
    for ci in range(cnt_correct):
        x0              = plain_1[ci] ^ keys[0][ci]     # 实际攻击中，首轮是free round，密钥异或下移至第一轮左右两个分支，plain_1是free round的输出左分支，异或首轮密钥，得到第一轮的真实输入
        y0              = plain_2[ci] ^ keys[0][ci]     # 实际攻击中，首轮是free round，密钥异或下移至第一轮左右两个分支，plain_2是free round的输出右分支，异或首轮密钥，得到第一轮的真实输入
        xrol7           = ror(x0, ALPHA)  # x >>> alpha, 第一轮模加的上输入
        xn1             = xrol7 ^ ((xrol7 << versions[VER][5](1) ) & MASK_VAL)  # (x >>> alpha) ⊕ (x >>> alpha)<<1, 考虑 x_i ⊕ x_{i-1} for 0 < i < WORD_SIZE
        yn1             = y0 ^ ((y0 << versions[VER][5](1)) & MASK_VAL)         # y ⊕ (y<<1), 考虑 y_i ⊕ y_{i-1} for 0 < i < WORD_SIZE
        plain_x7_p_y    = xrol7 ^ (y0)                  # (x>>>alpha) ⊕ y  考虑模加两个输入的异或
        xa_s_y          = (xrol7 + y0) & MASK_VAL # x + y mod n
        xa_x_c          = xa_s_y ^ (y0)                 # x ⊕ carry(x, y)
        y_x_c           = xa_s_y ^ (xrol7)              # y ⊕ carry(x, y)
        carry           = xa_s_y ^ plain_x7_p_y         # x + y mod n = x ⊕ y ⊕ carry(x, y) => carry(x, y) = (x + y mod n) ⊕ (x ⊕ y)
        # 统计明文中的各项频次
        AND_plain_x_all         = AND_plain_x_all + to_bits(x0)                    # 统计正确对的 x                                 各位置比特为1的次数
        AND_plain_y_all         = AND_plain_y_all + to_bits(y0)                    # 统计正确对的 y                                 各位置比特为1的次数
        AND_plain_xn1_all       = AND_plain_xn1_all + to_bits(xn1)                 # 统计正确对的 (x >>> alpha) ⊕ (x >>> alpha)<<1 各位置比特为1的次
        AND_plain_yn1_all       = AND_plain_yn1_all + to_bits(yn1)                 # 统计正确对的 y ⊕ (y<<1)                       各位置比特为1的次
        AND_plain_x7_p_y_all    = AND_plain_x7_p_y_all + to_bits(plain_x7_p_y)     # 统计正确对的 (x>>>alpha) ⊕ y                  各位置比特为1的次数
        AND_xa_x_c_all          = AND_xa_x_c_all + to_bits(xa_x_c)                 # 统计正确对的 (x>>>alpha) ⊕ carry              各位置比特为1的次数
        AND_y_x_c_all           = AND_y_x_c_all  + to_bits(y_x_c)                  # 统计正确对的 y ⊕ carry                        各位置比特为1的次数
        AND_c_all               = AND_c_all      + to_bits(carry)                  # 统计正确对的 carry                             各位置比特为1的次数
    # 对指定比特位置
    for xi in range(idx, (idx+1)):
        cur_neutral_bits_1xor = [[xi]]
        # 根据给定比特位置和输入差分，构造符合输入差分的翻转比特的明文结构
        # plain_01[.][0]为原始明文，plain_01[.][1]为翻转比特后的明文，plain_11为异或上输入差分的明文
        if input_diff2 == None:
            plain_01, plain_02, plain_11, plain_12 = make_plain_structure(plain_1, plain_2, input_diff, cur_neutral_bits_1xor)
        else:
            #plain_01, plain_02, plain_11, plain_12 = make_plain_structure_twodiff(plain_1, plain_2, input_diff, input_diff2, cur_neutral_bits_1xor)
            plain_01, plain_02, plain_11, plain_12 = make_plain_structure_twodiff_adjust(keys[0][ci], plain_1, plain_2, input_diff, input_diff2, trail[end_r - 1][0], cur_neutral_bits_1xor)
        # 先使用全0密钥经过一轮解密
        t_01, t_02 = speck.one_round_decrypt((plain_01, plain_02), zeo)
        t_11, t_12 = speck.one_round_decrypt((plain_11, plain_12), zeo)
        # 进行正常的加密
        t_01, t_02 = speck.encrypt((t_01, t_02), keys_st)
        t_11, t_12 = speck.encrypt((t_11, t_12), keys_st)
        # 计算输出差分，diff0[1]是翻转比特后的输出差分
        diff0 = t_01 ^ t_11
        diff1 = t_02 ^ t_12
        # 记录输出差分与给定差分相等的明文的索引值
        right_pairs = []
        for cci in range(len(diff0)):
            if ((diff0[cci][1] == output_diff[0]) and (diff1[cci][1] == output_diff[1])):
                right_pairs.append(cci)
        # 对仍然符合差分路径的明文统计各约束条件相应的频次
        for nci in range(len(right_pairs)):
            x0              = plain_01[right_pairs[nci]][0] ^ keys_st[0][right_pairs[nci]][0]
            y0              = plain_02[right_pairs[nci]][0] ^ keys_st[0][right_pairs[nci]][0]
            xrol7           = ror(x0, ALPHA)
            xn1             = xrol7 ^ ((xrol7 << versions[VER][5](1)) & MASK_VAL)
            yn1             = y0 ^ ((y0 << versions[VER][5](1)) & MASK_VAL)
            plain_x7_p_y    = ror(x0, ALPHA) ^ (y0)      
            xa_s_y          = (xrol7 + y0) & MASK_VAL            
            xa_x_c          = xa_s_y ^ (y0)
            y_x_c           = xa_s_y ^ (xrol7)
            carry           = xa_s_y ^ plain_x7_p_y
            # 统计频次
            Full_1xor[NBi_1xor]                = Full_1xor[NBi_1xor] + 1.0                               # Full_1xor[NBi_1xor]/cnt_correct = 第 NBi_1xor 个候选比特集是中性的频率
            AND_plain_x_1xor[NBi_1xor]         = AND_plain_x_1xor[NBi_1xor] + to_bits(x0)                # AND_plain_x_1xor[NBi_1xor, i]     : 统计第 NBi_1xor 个候选比特集对这个正确对体现中性，且这个正确对的 x                               的第 i 比特等于1 的次数
            AND_plain_y_1xor[NBi_1xor]         = AND_plain_y_1xor[NBi_1xor] + to_bits(y0)                # AND_plain_y_1xor[NBi_1xor, i]     : 统计第 NBi_1xor 个候选比特集对这个正确对体现中性，且这个正确对的 y                               的第 i 比特等于1 的次数
            AND_plain_xn1_1xor[NBi_1xor]       = AND_plain_xn1_1xor[NBi_1xor] + to_bits(xn1)             # AND_plain_xn1_1xor[NBi_1xor, i]   : 统计第 NBi_1xor 个候选比特集对这个正确对体现中性，且这个正确对的 (x>>>alpha) ⊕ ((x>>>alpha)<<1) 的第 i 比特等于1 的次数
            AND_plain_yn1_1xor[NBi_1xor]       = AND_plain_yn1_1xor[NBi_1xor] + to_bits(yn1)             # AND_plain_yn1_1xor[NBi_1xor, i]   : 统计第 NBi_1xor 个候选比特集对这个正确对体现中性，且这个正确对的 y ⊕ (y<<1)                     的第 i 比特等于1 的次数
            AND_plain_x7_p_y_1xor[NBi_1xor]    = AND_plain_x7_p_y_1xor[NBi_1xor] + to_bits(plain_x7_p_y) # AND_plain_x7_p_y_1xor[NBi_1xor, i]: 统计第 NBi_1xor 个候选比特集对这个正确对体现中性，且这个正确对的 (x>>>alpha) ⊕ y                的第 i 比特等于1 的次数
            AND_xa_x_c_1xor[NBi_1xor]          = AND_xa_x_c_1xor[NBi_1xor] + to_bits(xa_x_c)             # AND_xa_x_c_1xor[NBi_1xor, i]      : 统计第 NBi_1xor 个候选比特集对这个正确对体现中性，且这个正确对的 (x>>>alpha) ⊕ carry            的第 i 比特等于1 的次数
            AND_y_x_c_1xor[NBi_1xor]           = AND_y_x_c_1xor[NBi_1xor]  + to_bits(y_x_c)              # AND_y_x_c_1xor[NBi_1xor, i]       : 统计第 NBi_1xor 个候选比特集对这个正确对体现中性，且这个正确对的 y ⊕ carry                      的第 i 比特等于1 的次数
            AND_c_1xor[NBi_1xor]               = AND_c_1xor[NBi_1xor]      + to_bits(carry)              # AND_c_1xor[NBi_1xor, i]           : 统计第 NBi_1xor 个候选比特集对这个正确对体现中性，且这个正确对的 carry                           的第 i 比特等于1 的次数
        NBi_1xor = NBi_1xor + 1
        #
        if XORN >= 2:
            for xi2 in range(xi+1, BLOCK_SIZE):
                cur_neutral_bits_2xor = [[xi, xi2]]
                #
                if input_diff2 == None:
                    plain_01, plain_02, plain_11, plain_12 = make_plain_structure(plain_1, plain_2, input_diff, cur_neutral_bits_2xor)
                else:
                    #plain_01, plain_02, plain_11, plain_12 = make_plain_structure_twodiff(plain_1, plain_2, input_diff, input_diff2, cur_neutral_bits_2xor)
                    plain_01, plain_02, plain_11, plain_12 = make_plain_structure_twodiff_adjust(keys[0][ci], plain_1, plain_2, input_diff, input_diff2, trail[end_r - 1][0], cur_neutral_bits_2xor)
                #
                t_01, t_02 = speck.one_round_decrypt((plain_01, plain_02), zeo)
                t_11, t_12 = speck.one_round_decrypt((plain_11, plain_12), zeo)
                #
                t_01, t_02 = speck.encrypt((t_01, t_02), keys_st)
                t_11, t_12 = speck.encrypt((t_11, t_12), keys_st)
                #
                diff0 = t_01 ^ t_11
                diff1 = t_02 ^ t_12
                #
                right_pairs = []
                for cci in range(len(diff0)):
                    if ((diff0[cci][1] == output_diff[0]) and (diff1[cci][1] == output_diff[1])):
                        right_pairs.append(cci)
                #
                for nci in range(len(right_pairs)):
                    x0              = plain_01[right_pairs[nci]][0] ^ keys_st[0][right_pairs[nci]][0]
                    y0              = plain_02[right_pairs[nci]][0] ^ keys_st[0][right_pairs[nci]][0]
                    xrol7           = ror(x0, ALPHA)
                    xn1             = xrol7 ^ ((xrol7 << versions[VER][5](1)) & MASK_VAL)
                    yn1             = y0 ^ ((y0 << versions[VER][5](1)) & MASK_VAL)
                    plain_x7_p_y    = ror(x0, ALPHA) ^ (y0)
                    xa_s_y          = (xrol7 + y0) & MASK_VAL
                    xa_x_c          = xa_s_y ^ (y0)
                    y_x_c           = xa_s_y ^ (xrol7)
                    carry           = xa_s_y ^ plain_x7_p_y
                    #
                    Full_2xor[NBi_2xor]                = Full_2xor[NBi_2xor] + 1.0
                    AND_plain_x_2xor[NBi_2xor]         = AND_plain_x_2xor[NBi_2xor] + to_bits(x0)
                    AND_plain_y_2xor[NBi_2xor]         = AND_plain_y_2xor[NBi_2xor] + to_bits(y0)
                    AND_plain_xn1_2xor[NBi_2xor]       = AND_plain_xn1_2xor[NBi_2xor] + to_bits(xn1)
                    AND_plain_yn1_2xor[NBi_2xor]       = AND_plain_yn1_2xor[NBi_2xor] + to_bits(yn1)
                    AND_plain_x7_p_y_2xor[NBi_2xor]    = AND_plain_x7_p_y_2xor[NBi_2xor] + to_bits(plain_x7_p_y)
                    AND_xa_x_c_2xor[NBi_2xor]          = AND_xa_x_c_2xor[NBi_2xor] + to_bits(xa_x_c)
                    AND_y_x_c_2xor[NBi_2xor]           = AND_y_x_c_2xor[NBi_2xor]  + to_bits(y_x_c)
                    AND_c_2xor[NBi_2xor]               = AND_c_2xor[NBi_2xor]      + to_bits(carry)
                NBi_2xor = NBi_2xor + 1
                #
                if XORN >= 3:
                    for xi3 in range(xi2+1, BLOCK_SIZE):
                        cur_neutral_bits_3xor = [[xi, xi2, xi3]]
                        #
                        if input_diff2 == None:
                            plain_01, plain_02, plain_11, plain_12 = make_plain_structure(plain_1, plain_2, input_diff, cur_neutral_bits_3xor)
                        else:
                            #plain_01, plain_02, plain_11, plain_12 = make_plain_structure_twodiff(plain_1, plain_2, input_diff, input_diff2, cur_neutral_bits_3xor)
                            plain_01, plain_02, plain_11, plain_12 = make_plain_structure_twodiff_adjust(keys[0][ci], plain_1, plain_2, input_diff, input_diff2, trail[end_r - 1][0], cur_neutral_bits_3xor)
                        #
                        t_01, t_02 = speck.one_round_decrypt((plain_01, plain_02), zeo)
                        t_11, t_12 = speck.one_round_decrypt((plain_11, plain_12), zeo)
                        #
                        t_01, t_02 = speck.encrypt((t_01, t_02), keys_st)
                        t_11, t_12 = speck.encrypt((t_11, t_12), keys_st)
                        #
                        diff0 = t_01 ^ t_11
                        diff1 = t_02 ^ t_12
                        #
                        right_pairs = []
                        for cci in range(len(diff0)):
                            if ((diff0[cci][1] == output_diff[0]) and (diff1[cci][1] == output_diff[1])):
                                right_pairs.append(cci)
                        #
                        for nci in range(len(right_pairs)):
                            x0              = plain_01[right_pairs[nci]][0] ^ keys_st[0][right_pairs[nci]][0]
                            y0              = plain_02[right_pairs[nci]][0] ^ keys_st[0][right_pairs[nci]][0]
                            xrol7           = ror(x0, ALPHA)
                            xn1             = xrol7 ^ ((xrol7 << versions[VER][5](1)) & MASK_VAL)
                            yn1             = y0 ^ ((y0 << versions[VER][5](1)) & MASK_VAL)
                            plain_x7_p_y    = ror(x0, ALPHA) ^ (y0)
                            xa_s_y          = (xrol7 + y0) & MASK_VAL
                            xa_x_c          = xa_s_y ^ (y0)
                            y_x_c           = xa_s_y ^ (xrol7)
                            carry           = xa_s_y ^ plain_x7_p_y
                            #
                            Full_3xor[NBi_3xor]                = Full_3xor[NBi_3xor] + 1.0
                            AND_plain_x_3xor[NBi_3xor]         = AND_plain_x_3xor[NBi_3xor] + to_bits(x0)
                            AND_plain_y_3xor[NBi_3xor]         = AND_plain_y_3xor[NBi_3xor] + to_bits(y0)
                            AND_plain_xn1_3xor[NBi_3xor]       = AND_plain_xn1_3xor[NBi_3xor] + to_bits(xn1)
                            AND_plain_yn1_3xor[NBi_3xor]       = AND_plain_yn1_3xor[NBi_3xor] + to_bits(yn1)
                            AND_plain_x7_p_y_3xor[NBi_3xor]    = AND_plain_x7_p_y_3xor[NBi_3xor] + to_bits(plain_x7_p_y)
                            AND_xa_x_c_3xor[NBi_3xor]          = AND_xa_x_c_3xor[NBi_3xor] + to_bits(xa_x_c)
                            AND_y_x_c_3xor[NBi_3xor]           = AND_y_x_c_3xor[NBi_3xor]  + to_bits(y_x_c)
                            AND_c_3xor[NBi_3xor]               = AND_c_3xor[NBi_3xor]      + to_bits(carry)
                        NBi_3xor = NBi_3xor + 1
                        #
                        if XORN >= 4:
                            for xi4 in range(xi3+1, BLOCK_SIZE):
                                cur_neutral_bits_4xor = [[xi, xi2, xi3, xi4]]
                                #
                                if input_diff2 == None:
                                    plain_01, plain_02, plain_11, plain_12 = make_plain_structure(plain_1, plain_2, input_diff, cur_neutral_bits_4xor)
                                else:
                                    #plain_01, plain_02, plain_11, plain_12 = make_plain_structure_twodiff(plain_1, plain_2, input_diff, input_diff2, cur_neutral_bits_4xor)
                                    plain_01, plain_02, plain_11, plain_12 = make_plain_structure_twodiff_adjust(keys[0][ci], plain_1, plain_2, input_diff, input_diff2, trail[end_r - 1][0], cur_neutral_bits_4xor)
                                #
                                t_01, t_02 = speck.one_round_decrypt((plain_01, plain_02), zeo)
                                t_11, t_12 = speck.one_round_decrypt((plain_11, plain_12), zeo)
                                #
                                t_01, t_02 = speck.encrypt((t_01, t_02), keys_st)
                                t_11, t_12 = speck.encrypt((t_11, t_12), keys_st)
                                #
                                diff0 = t_01 ^ t_11
                                diff1 = t_02 ^ t_12
                                #
                                right_pairs = []
                                for cci in range(len(diff0)):
                                    if ((diff0[cci][1] == output_diff[0]) and (diff1[cci][1] == output_diff[1])):
                                        right_pairs.append(cci)
                                #
                                for nci in range(len(right_pairs)):
                                    x0              = plain_01[right_pairs[nci]][0] ^ keys_st[0][right_pairs[nci]][0]
                                    y0              = plain_02[right_pairs[nci]][0] ^ keys_st[0][right_pairs[nci]][0]
                                    xrol7           = ror(x0, ALPHA)
                                    xn1             = xrol7 ^ ((xrol7 << versions[VER][5](1)) & MASK_VAL)
                                    yn1             = y0 ^ ((y0 << versions[VER][5](1)) & MASK_VAL)
                                    plain_x7_p_y    = ror(x0, ALPHA) ^ (y0)
                                    xa_s_y          = (xrol7 + y0) & MASK_VAL
                                    xa_x_c          = xa_s_y ^ (y0)
                                    y_x_c           = xa_s_y ^ (xrol7)
                                    carry           = xa_s_y ^ plain_x7_p_y
                                    #
                                    Full_4xor[NBi_4xor]                = Full_4xor[NBi_4xor] + 1.0
                                    AND_plain_x_4xor[NBi_4xor]         = AND_plain_x_4xor[NBi_4xor] + to_bits(x0)
                                    AND_plain_y_4xor[NBi_4xor]         = AND_plain_y_4xor[NBi_4xor] + to_bits(y0)
                                    AND_plain_xn1_4xor[NBi_4xor]       = AND_plain_xn1_4xor[NBi_4xor] + to_bits(xn1)
                                    AND_plain_yn1_4xor[NBi_4xor]       = AND_plain_yn1_4xor[NBi_4xor] + to_bits(yn1)
                                    AND_plain_x7_p_y_4xor[NBi_4xor]    = AND_plain_x7_p_y_4xor[NBi_4xor] + to_bits(plain_x7_p_y)
                                    AND_xa_x_c_4xor[NBi_4xor]          = AND_xa_x_c_4xor[NBi_4xor] + to_bits(xa_x_c)
                                    AND_y_x_c_4xor[NBi_4xor]           = AND_y_x_c_4xor[NBi_4xor]  + to_bits(y_x_c)
                                    AND_c_4xor[NBi_4xor]               = AND_c_4xor[NBi_4xor]      + to_bits(carry)
                                NBi_4xor = NBi_4xor + 1
                                #
                                if XORN >= 5:
                                    for xi5 in range(xi4+1, BLOCK_SIZE):
                                        cur_neutral_bits_5xor = [[xi, xi2, xi3, xi4, xi5]]
                                        #
                                        if input_diff2 == None:
                                            plain_01, plain_02, plain_11, plain_12 = make_plain_structure(plain_1, plain_2, input_diff, cur_neutral_bits_5xor)
                                        else:
                                            #plain_01, plain_02, plain_11, plain_12 = make_plain_structure_twodiff(plain_1, plain_2, input_diff, input_diff2, cur_neutral_bits_5xor)
                                            plain_01, plain_02, plain_11, plain_12 = make_plain_structure_twodiff_adjust(keys[0][ci], plain_1, plain_2, input_diff, input_diff2, trail[end_r - 1][0], cur_neutral_bits_5xor)
                                        #
                                        t_01, t_02 = speck.one_round_decrypt((plain_01, plain_02), zeo)
                                        t_11, t_12 = speck.one_round_decrypt((plain_11, plain_12), zeo)
                                        #
                                        t_01, t_02 = speck.encrypt((t_01, t_02), keys_st)
                                        t_11, t_12 = speck.encrypt((t_11, t_12), keys_st)
                                        #
                                        diff0 = t_01 ^ t_11
                                        diff1 = t_02 ^ t_12
                                        #
                                        right_pairs = []
                                        for cci in range(len(diff0)):
                                            if ((diff0[cci][1] == output_diff[0]) and (diff1[cci][1] == output_diff[1])):
                                                right_pairs.append(cci)
                                        #
                                        for nci in range(len(right_pairs)):
                                            x0              = plain_01[right_pairs[nci]][0] ^ keys_st[0][right_pairs[nci]][0]
                                            y0              = plain_02[right_pairs[nci]][0] ^ keys_st[0][right_pairs[nci]][0]
                                            xrol7           = ror(x0, ALPHA)
                                            xn1             = xrol7 ^ ((xrol7 << versions[VER][5](1)) & MASK_VAL)
                                            yn1             = y0 ^ ((y0 << versions[VER][5](1)) & MASK_VAL)
                                            plain_x7_p_y    = ror(x0, ALPHA) ^ (y0)
                                            xa_s_y          = (xrol7 + y0) & MASK_VAL
                                            xa_x_c          = xa_s_y ^ (y0)
                                            y_x_c           = xa_s_y ^ (xrol7)
                                            carry           = xa_s_y ^ plain_x7_p_y
                                            #
                                            Full_5xor[NBi_5xor]                = Full_5xor[NBi_5xor] + 1.0
                                            AND_plain_x_5xor[NBi_5xor]         = AND_plain_x_5xor[NBi_5xor] + to_bits(x0)
                                            AND_plain_y_5xor[NBi_5xor]         = AND_plain_y_5xor[NBi_5xor] + to_bits(y0)
                                            AND_plain_xn1_5xor[NBi_5xor]       = AND_plain_xn1_5xor[NBi_5xor] + to_bits(xn1)
                                            AND_plain_yn1_5xor[NBi_5xor]       = AND_plain_yn1_5xor[NBi_5xor] + to_bits(yn1)
                                            AND_plain_x7_p_y_5xor[NBi_5xor]    = AND_plain_x7_p_y_5xor[NBi_5xor] + to_bits(plain_x7_p_y)
                                            AND_xa_x_c_5xor[NBi_5xor]          = AND_xa_x_c_5xor[NBi_5xor] + to_bits(xa_x_c)
                                            AND_y_x_c_5xor[NBi_5xor]           = AND_y_x_c_5xor[NBi_5xor]  + to_bits(y_x_c)
                                            AND_c_5xor[NBi_5xor]               = AND_c_5xor[NBi_5xor]      + to_bits(carry)
                                        NBi_5xor = NBi_5xor + 1
    # 中性比特的索引
    NBi_1xor = 0
    NBi_2xor = 0
    NBi_3xor = 0
    NBi_4xor = 0
    NBi_5xor = 0

    print(
    "  Pr_x_eq1 "     , AND_plain_x_all/cnt_correct           ,  # x 各比特是1的频率，为了检测 x_i = 1 对中性概率的影响, for 0 <= i < WORD_SIZE
    "  Pr_x_eq0 "     , 1.0 - AND_plain_x_all/cnt_correct     ,  # x 各比特是0的频率，为了检测 x_i = 0 对中性概率的影响, for 0 <= i < WORD_SIZE
    "  Pr_y_eq1 "     , AND_plain_y_all/cnt_correct           ,  # y 各比特是1的频率，为了检测 y_i = 1 对中性概率的影响, for 0 <= i < WORD_SIZE
    "  Pr_y_eq0 "     , 1.0 - AND_plain_y_all/cnt_correct     ,  # y 各比特是0的频率，为了检测 y_i = 0 对中性概率的影响, for 0 <= i < WORD_SIZE
    "  Pr_xn1_eq1 "   , AND_plain_xn1_all/cnt_correct         ,  # (x>>>alpha) ⊕ ((x>>>alpha)<<1) 各比特是1的频率，为了检测 x_i ⊕ x_{i-1} = 1 对中性概率的影响, for 0 < i < WORD_SIZE
    "  Pr_xn1_eq0 "   , 1.0 - AND_plain_xn1_all/cnt_correct   ,  # (x>>>alpha) ⊕ ((x>>>alpha)<<1) 各比特是0的频率，为了检测 x_i ⊕ x_{i-1} = 0 对中性概率的影响, for 0 < i < WORD_SIZE
    "  Pr_yn1_eq1 "   , AND_plain_yn1_all/cnt_correct         ,  # y ⊕ (y<<1) 各比特是1的频率，为了检测 y_i ⊕ y_{i-1} = 1 对中性概率的影响, for 0 < i < WORD_SIZE
    "  Pr_yn1_eq0 "   , 1.0 - AND_plain_yn1_all/cnt_correct   ,  # y ⊕ (y<<1) 各比特是0的频率，为了检测 y_i ⊕ y_{i-1} = 0 对中性概率的影响, for 0 < i < WORD_SIZE
    "  Pr_xa_p_y_eq1 ", AND_plain_x7_p_y_all/cnt_correct      ,  # (x>>>alpha) ⊕ y 各比特是1的频率，为了检测 ((x>>>alpha) ⊕ y)_i = 1 对中性概率的影响, for 0 <= i < WORD_SIZE
    "  Pr_xa_p_y_eq0 ", 1.0 - AND_plain_x7_p_y_all/cnt_correct,  # (x>>>alpha) ⊕ y 各比特是0的频率，为了检测 ((x>>>alpha) ⊕ y)_i = 0 对中性概率的影响, for 0 <= i < WORD_SIZE
    "  Pr_xa_x_c_eq1 ", AND_xa_x_c_all/cnt_correct            ,  # (x>>>alpha) ⊕ carry 各比特是1的频率，为了检测 ((x>>>alpha) ⊕ carry)_i = 1 对中性概率的影响, for 0 <= i < WORD_SIZE
    "  Pr_xa_x_c_eq0 ", 1.0 - AND_xa_x_c_all/cnt_correct      ,  # (x>>>alpha) ⊕ carry 各比特是0的频率，为了检测 ((x>>>alpha) ⊕ carry)_i = 0 对中性概率的影响, for 0 <= i < WORD_SIZE
    "  Pr_y_x_c_eq1 " , AND_y_x_c_all/cnt_correct             ,  # y ⊕ carry 各比特是1的频率，为了检测 (y ⊕ carry)_i = 1 对中性概率的影响, for 0 <= i < WORD_SIZE
    "  Pr_y_x_c_eq0 " , 1.0 - AND_y_x_c_all/cnt_correct       ,  # y ⊕ carry 各比特是0的频率，为了检测 (y ⊕ carry)_i = 0 对中性概率的影响, for 0 <= i < WORD_SIZE
    "  Pr_c_eq1 "     , AND_c_all/cnt_correct                 ,  # carry 各比特是1的频率，为了检测 carry_i = 1 对中性概率的影响, for 0 <= i < WORD_SIZE
    "  Pr_c_eq0 "     , 1.0 - AND_c_all/cnt_correct           ,  # carry 各比特是0的频率，为了检测 carry_i = 0 对中性概率的影响, for 0 <= i < WORD_SIZE
    file=logfile_allc, flush=True)
    for xi in range(idx, idx+1):
        cur_neutral_bits_1xor = [xi]
        # 统计当前比特满足各约束条件下是中性比特的概率
        Pr_nt_x_eq1       = div0(AND_plain_x_1xor[NBi_1xor]                           , AND_plain_x_all)                     # Pr_nt_x_eq1     [i] = x                                 第 i 位置比特 = 1 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_x_eq0       = div0(Full_1xor[NBi_1xor] - AND_plain_x_1xor[NBi_1xor]     , cnt_correct - AND_plain_x_all)       # Pr_nt_x_eq0     [i] = x                                 第 i 位置比特 = 0 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_y_eq1       = div0(AND_plain_y_1xor[NBi_1xor]                           , AND_plain_y_all)                     # Pr_nt_y_eq1     [i] = y                                 第 i 位置比特 = 1 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_y_eq0       = div0(Full_1xor[NBi_1xor] - AND_plain_y_1xor[NBi_1xor]     , cnt_correct - AND_plain_y_all)       # Pr_nt_y_eq0     [i] = y                                 第 i 位置比特 = 0 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_xn1_eq1     = div0(AND_plain_xn1_1xor[NBi_1xor]                         , AND_plain_xn1_all)                   # Pr_nt_xn1_eq1   [i] = (x >>> alpha) ⊕ (x >>> alpha)<<1 第 i 位置比特 = 1 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_xn1_eq0     = div0(Full_1xor[NBi_1xor] - AND_plain_xn1_1xor[NBi_1xor]   , cnt_correct - AND_plain_xn1_all)     # Pr_nt_xn1_eq0   [i] = (x >>> alpha) ⊕ (x >>> alpha)<<1 第 i 位置比特 = 0 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_yn1_eq1     = div0(AND_plain_yn1_1xor[NBi_1xor]                         , AND_plain_yn1_all)                   # Pr_nt_yn1_eq1   [i] = y ⊕ (y<<1)                       第 i 位置比特 = 1 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_yn1_eq0     = div0(Full_1xor[NBi_1xor] - AND_plain_yn1_1xor[NBi_1xor]   , cnt_correct - AND_plain_yn1_all)     # Pr_nt_yn1_eq0   [i] = y ⊕ (y<<1)                       第 i 位置比特 = 0 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_xa_p_y_eq1  = div0(AND_plain_x7_p_y_1xor[NBi_1xor]                      , AND_plain_x7_p_y_all)                # Pr_nt_xa_p_y_eq1[i] = (x>>>alpha) ⊕ y                  第 i 位置比特 = 1 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_xa_p_y_eq0  = div0(Full_1xor[NBi_1xor] - AND_plain_x7_p_y_1xor[NBi_1xor], cnt_correct - AND_plain_x7_p_y_all)  # Pr_nt_xa_p_y_eq0[i] = (x>>>alpha) ⊕ y                  第 i 位置比特 = 0 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_xa_x_c_eq1  = div0(AND_xa_x_c_1xor[NBi_1xor]                            , AND_xa_x_c_all)                      # Pr_nt_xa_x_c_eq1[i] = (x>>>alpha) ⊕ carry              第 i 位置比特 = 1 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_xa_x_c_eq0  = div0(Full_1xor[NBi_1xor] - AND_xa_x_c_1xor[NBi_1xor]      , cnt_correct - AND_xa_x_c_all)        # Pr_nt_xa_x_c_eq0[i] = (x>>>alpha) ⊕ carry              第 i 位置比特 = 0 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_y_x_c_eq1   = div0(AND_y_x_c_1xor[NBi_1xor]                             , AND_y_x_c_all)                       # Pr_nt_y_x_c_eq1 [i] = y ⊕ carry                        第 i 位置比特 = 1 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_y_x_c_eq0   = div0(Full_1xor[NBi_1xor] - AND_y_x_c_1xor[NBi_1xor]       , cnt_correct - AND_y_x_c_all)         # Pr_nt_y_x_c_eq0 [i] = y ⊕ carry                        第 i 位置比特 = 0 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_c_eq1       = div0(AND_c_1xor[NBi_1xor]                                 , AND_c_all)                           # Pr_nt_c_eq1     [i] = carry                             第 i 位置比特 = 1 的条件下第NBi_1xor个候选比特集是中性的概率
        Pr_nt_c_eq0       = div0(Full_1xor[NBi_1xor] - AND_c_1xor[NBi_1xor]           , cnt_correct - AND_c_all)             # Pr_nt_c_eq0     [i] = carry                             第 i 位置比特 = 0 的条件下第NBi_1xor个候选比特集是中性的概率
        # 当前比特是中性比特的概率为Full_1xor[NBi_1xor]/cnt_correct
        if Full_1xor[NBi_1xor]/cnt_correct > threshold_freq_low: # 中性率大于阈值
            print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), 
            "  Pr_x_nt_eq1 "     , div0(AND_plain_x_1xor[NBi_1xor]                                  , Full_1xor[NBi_1xor]), "  Pr_nt_x_eq1 "     , Pr_nt_x_eq1           ,
            "  Pr_x_nt_eq0 "     , div0(Full_1xor[NBi_1xor] - AND_plain_x_1xor[NBi_1xor]            , Full_1xor[NBi_1xor]), "  Pr_nt_x_eq0 "     , Pr_nt_x_eq0           ,
            "  Pr_y_nt_eq1 "     , div0(AND_plain_y_1xor[NBi_1xor]                                  , Full_1xor[NBi_1xor]), "  Pr_nt_y_eq1 "     , Pr_nt_y_eq1           ,
            "  Pr_y_nt_eq0 "     , div0(Full_1xor[NBi_1xor] - AND_plain_y_1xor[NBi_1xor]            , Full_1xor[NBi_1xor]), "  Pr_nt_y_eq0 "     , Pr_nt_y_eq0           ,
            "  Pr_xn1_nt_eq1 "   , div0(AND_plain_xn1_1xor[NBi_1xor]                                , Full_1xor[NBi_1xor]), "  Pr_nt_xn1_eq1 "   , Pr_nt_xn1_eq1         ,
            "  Pr_xn1_nt_eq0 "   , div0(Full_1xor[NBi_1xor] - AND_plain_xn1_1xor[NBi_1xor]          , Full_1xor[NBi_1xor]), "  Pr_nt_xn1_eq0 "   , Pr_nt_xn1_eq0         ,
            "  Pr_yn1_nt_eq1 "   , div0(AND_plain_yn1_1xor[NBi_1xor]                                , Full_1xor[NBi_1xor]), "  Pr_nt_yn1_eq1 "   , Pr_nt_yn1_eq1         ,
            "  Pr_yn1_nt_eq0 "   , div0(Full_1xor[NBi_1xor] - AND_plain_yn1_1xor[NBi_1xor]          , Full_1xor[NBi_1xor]), "  Pr_nt_yn1_eq0 "   , Pr_nt_yn1_eq0         ,
            "  Pr_xa_p_y_nt_eq1 ", div0(AND_plain_x7_p_y_1xor[NBi_1xor]                             , Full_1xor[NBi_1xor]), "  Pr_nt_xa_p_y_eq1 ", Pr_nt_xa_p_y_eq1      ,
            "  Pr_xa_p_y_nt_eq0 ", div0(Full_1xor[NBi_1xor] - AND_plain_x7_p_y_1xor[NBi_1xor]       , Full_1xor[NBi_1xor]), "  Pr_nt_xa_p_y_eq0 ", Pr_nt_xa_p_y_eq0      ,
            "  Pr_xa_x_c_nt_eq1 ", div0(AND_xa_x_c_1xor[NBi_1xor]                                   , Full_1xor[NBi_1xor]), "  Pr_nt_xa_x_c_eq1 ", Pr_nt_xa_x_c_eq1,
            "  Pr_xa_x_c_nt_eq0 ", div0(Full_1xor[NBi_1xor] - AND_xa_x_c_1xor[NBi_1xor]             , Full_1xor[NBi_1xor]), "  Pr_nt_xa_x_c_eq0 ", Pr_nt_xa_x_c_eq0,
            "  Pr_y_x_c_nt_eq1 " , div0(AND_y_x_c_1xor[NBi_1xor]                                    , Full_1xor[NBi_1xor]), "  Pr_nt_y_x_c_eq1 " , Pr_nt_y_x_c_eq1 ,
            "  Pr_y_x_c_nt_eq0 " , div0(Full_1xor[NBi_1xor] - AND_y_x_c_1xor[NBi_1xor]              , Full_1xor[NBi_1xor]), "  Pr_nt_y_x_c_eq0 " , Pr_nt_y_x_c_eq0 ,
            "  Pr_c_nt_eq1 "     , div0(AND_c_1xor[NBi_1xor]                                        , Full_1xor[NBi_1xor]), "  Pr_nt_c_eq1 "     , Pr_nt_c_eq1     ,
            "  Pr_c_nt_eq0 "     , div0(Full_1xor[NBi_1xor] - AND_c_1xor[NBi_1xor]                  , Full_1xor[NBi_1xor]), "  Pr_nt_c_eq0 "     , Pr_nt_c_eq0     ,
            file=logfile_1xor, flush=True)
        if Full_1xor[NBi_1xor]/cnt_correct > threshold_freq:
            print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), 
            "  Pr_x_nt_eq1 "     , div0(AND_plain_x_1xor[NBi_1xor]                                  , Full_1xor[NBi_1xor]), "  Pr_nt_x_eq1 "     , Pr_nt_x_eq1           ,
            "  Pr_x_nt_eq0 "     , div0(Full_1xor[NBi_1xor] - AND_plain_x_1xor[NBi_1xor]            , Full_1xor[NBi_1xor]), "  Pr_nt_x_eq0 "     , Pr_nt_x_eq0           ,
            "  Pr_y_nt_eq1 "     , div0(AND_plain_y_1xor[NBi_1xor]                                  , Full_1xor[NBi_1xor]), "  Pr_nt_y_eq1 "     , Pr_nt_y_eq1           ,
            "  Pr_y_nt_eq0 "     , div0(Full_1xor[NBi_1xor] - AND_plain_y_1xor[NBi_1xor]            , Full_1xor[NBi_1xor]), "  Pr_nt_y_eq0 "     , Pr_nt_y_eq0           ,
            "  Pr_xn1_nt_eq1 "   , div0(AND_plain_xn1_1xor[NBi_1xor]                                , Full_1xor[NBi_1xor]), "  Pr_nt_xn1_eq1 "   , Pr_nt_xn1_eq1         ,
            "  Pr_xn1_nt_eq0 "   , div0(Full_1xor[NBi_1xor] - AND_plain_xn1_1xor[NBi_1xor]          , Full_1xor[NBi_1xor]), "  Pr_nt_xn1_eq0 "   , Pr_nt_xn1_eq0         ,
            "  Pr_yn1_nt_eq1 "   , div0(AND_plain_yn1_1xor[NBi_1xor]                                , Full_1xor[NBi_1xor]), "  Pr_nt_yn1_eq1 "   , Pr_nt_yn1_eq1         ,
            "  Pr_yn1_nt_eq0 "   , div0(Full_1xor[NBi_1xor] - AND_plain_yn1_1xor[NBi_1xor]          , Full_1xor[NBi_1xor]), "  Pr_nt_yn1_eq0 "   , Pr_nt_yn1_eq0         ,
            "  Pr_xa_p_y_nt_eq1 ", div0(AND_plain_x7_p_y_1xor[NBi_1xor]                             , Full_1xor[NBi_1xor]), "  Pr_nt_xa_p_y_eq1 ", Pr_nt_xa_p_y_eq1      ,
            "  Pr_xa_p_y_nt_eq0 ", div0(Full_1xor[NBi_1xor] - AND_plain_x7_p_y_1xor[NBi_1xor]       , Full_1xor[NBi_1xor]), "  Pr_nt_xa_p_y_eq0 ", Pr_nt_xa_p_y_eq0      ,
            "  Pr_xa_x_c_nt_eq1 ", div0(AND_xa_x_c_1xor[NBi_1xor]                                   , Full_1xor[NBi_1xor]), "  Pr_nt_xa_x_c_eq1 ", Pr_nt_xa_x_c_eq1,
            "  Pr_xa_x_c_nt_eq0 ", div0(Full_1xor[NBi_1xor] - AND_xa_x_c_1xor[NBi_1xor]             , Full_1xor[NBi_1xor]), "  Pr_nt_xa_x_c_eq0 ", Pr_nt_xa_x_c_eq0,
            "  Pr_y_x_c_nt_eq1 " , div0(AND_y_x_c_1xor[NBi_1xor]                                    , Full_1xor[NBi_1xor]), "  Pr_nt_y_x_c_eq1 " , Pr_nt_y_x_c_eq1 ,
            "  Pr_y_x_c_nt_eq0 " , div0(Full_1xor[NBi_1xor] - AND_y_x_c_1xor[NBi_1xor]              , Full_1xor[NBi_1xor]), "  Pr_nt_y_x_c_eq0 " , Pr_nt_y_x_c_eq0 ,
            "  Pr_c_nt_eq1 "     , div0(AND_c_1xor[NBi_1xor]                                        , Full_1xor[NBi_1xor]), "  Pr_nt_c_eq1 "     , Pr_nt_c_eq1     ,
            "  Pr_c_nt_eq0 "     , div0(Full_1xor[NBi_1xor] - AND_c_1xor[NBi_1xor]                  , Full_1xor[NBi_1xor]), "  Pr_nt_c_eq0 "     , Pr_nt_c_eq0     ,
            file=logfile_alls, flush=True)
        # 计算各个频率与理论概率的偏差，偏差大于阈值才输出结果，观察约束条件对中性率的影响
        diff_Pr_nt_x_eq1       = Pr_nt_x_eq1       - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_x_eq1       = np.max(Pr_nt_x_eq1     );
        diff_Pr_nt_x_eq0       = Pr_nt_x_eq0       - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_x_eq0       = np.max(Pr_nt_x_eq0     );
        diff_Pr_nt_y_eq1       = Pr_nt_y_eq1       - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_y_eq1       = np.max(Pr_nt_y_eq1     );
        diff_Pr_nt_y_eq0       = Pr_nt_y_eq0       - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_y_eq0       = np.max(Pr_nt_y_eq0     );
        diff_Pr_nt_xn1_eq1     = Pr_nt_xn1_eq1     - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_xn1_eq1     = np.max(Pr_nt_xn1_eq1   );
        diff_Pr_nt_xn1_eq0     = Pr_nt_xn1_eq0     - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_xn1_eq0     = np.max(Pr_nt_xn1_eq0   );
        diff_Pr_nt_yn1_eq1     = Pr_nt_yn1_eq1     - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_yn1_eq1     = np.max(Pr_nt_yn1_eq1   );
        diff_Pr_nt_yn1_eq0     = Pr_nt_yn1_eq0     - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_yn1_eq0     = np.max(Pr_nt_yn1_eq0   );
        diff_Pr_nt_xa_p_y_eq1  = Pr_nt_xa_p_y_eq1  - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_xa_p_y_eq1  = np.max(Pr_nt_xa_p_y_eq1);
        diff_Pr_nt_xa_p_y_eq0  = Pr_nt_xa_p_y_eq0  - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_xa_p_y_eq0  = np.max(Pr_nt_xa_p_y_eq0);
        diff_Pr_nt_xa_x_c_eq1  = Pr_nt_xa_x_c_eq1  - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_xa_x_c_eq1  = np.max(Pr_nt_xa_x_c_eq1);
        diff_Pr_nt_xa_x_c_eq0  = Pr_nt_xa_x_c_eq0  - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_xa_x_c_eq0  = np.max(Pr_nt_xa_x_c_eq0);
        diff_Pr_nt_y_x_c_eq1   = Pr_nt_y_x_c_eq1   - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_y_x_c_eq1   = np.max(Pr_nt_y_x_c_eq1 );
        diff_Pr_nt_y_x_c_eq0   = Pr_nt_y_x_c_eq0   - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_y_x_c_eq0   = np.max(Pr_nt_y_x_c_eq0 );
        diff_Pr_nt_c_eq1       = Pr_nt_c_eq1       - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_c_eq1       = np.max(Pr_nt_c_eq1     );
        diff_Pr_nt_c_eq0       = Pr_nt_c_eq0       - Full_1xor[NBi_1xor]/cnt_correct;  max_Pr_nt_c_eq0       = np.max(Pr_nt_c_eq0     );
        if len(diff_Pr_nt_x_eq1     [diff_Pr_nt_x_eq1      >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_x_eq1     ), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_x_eq1      " , Pr_nt_x_eq1     , file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_x_eq0     [diff_Pr_nt_x_eq0      >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_x_eq0     ), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_x_eq0      " , Pr_nt_x_eq0     , file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_y_eq1     [diff_Pr_nt_y_eq1      >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_eq1     ), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_y_eq1      " , Pr_nt_y_eq1     , file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_y_eq0     [diff_Pr_nt_y_eq0      >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_eq0     ), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_y_eq0      " , Pr_nt_y_eq0     , file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_xn1_eq1   [diff_Pr_nt_xn1_eq1    >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xn1_eq1   ), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_xn1_eq1    " , Pr_nt_xn1_eq1   , file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_xn1_eq0   [diff_Pr_nt_xn1_eq0    >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xn1_eq0   ), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_xn1_eq0    " , Pr_nt_xn1_eq0   , file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_yn1_eq1   [diff_Pr_nt_yn1_eq1    >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_yn1_eq1   ), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_yn1_eq1    " , Pr_nt_yn1_eq1   , file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_yn1_eq0   [diff_Pr_nt_yn1_eq0    >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_yn1_eq0   ), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_yn1_eq0    " , Pr_nt_yn1_eq0   , file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_xa_p_y_eq1[diff_Pr_nt_xa_p_y_eq1 >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_p_y_eq1), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_xa_p_y_eq1 " , Pr_nt_xa_p_y_eq1, file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_xa_p_y_eq0[diff_Pr_nt_xa_p_y_eq0 >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_p_y_eq0), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_xa_p_y_eq0 " , Pr_nt_xa_p_y_eq0, file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_xa_x_c_eq1[diff_Pr_nt_xa_x_c_eq1 >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_x_c_eq1), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_xa_x_c_eq1 " , Pr_nt_xa_x_c_eq1, file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_xa_x_c_eq0[diff_Pr_nt_xa_x_c_eq0 >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_x_c_eq0), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_xa_x_c_eq0 " , Pr_nt_xa_x_c_eq0, file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_y_x_c_eq1 [diff_Pr_nt_y_x_c_eq1  >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_x_c_eq1 ), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_y_x_c_eq1  " , Pr_nt_y_x_c_eq1 , file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_y_x_c_eq0 [diff_Pr_nt_y_x_c_eq0  >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_x_c_eq0 ), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_y_x_c_eq0  " , Pr_nt_y_x_c_eq0 , file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_c_eq1     [diff_Pr_nt_c_eq1      >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_c_eq1     ), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_c_eq1      " , Pr_nt_c_eq1     , file=logfile_cnds, flush=True)
        if len(diff_Pr_nt_c_eq0     [diff_Pr_nt_c_eq0      >= GPT]) != 0:
            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_1xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_c_eq0     ), ' from ', '{:0.3f}'.format(Full_1xor[NBi_1xor]/cnt_correct), " Pr_nt_c_eq0      " , Pr_nt_c_eq0     , file=logfile_cnds, flush=True)
        NBi_1xor = NBi_1xor + 1
        #
        del Pr_nt_x_eq1       
        del Pr_nt_x_eq0       
        del Pr_nt_y_eq1       
        del Pr_nt_y_eq0       
        del Pr_nt_xn1_eq1     
        del Pr_nt_xn1_eq0     
        del Pr_nt_yn1_eq1     
        del Pr_nt_yn1_eq0     
        del Pr_nt_xa_p_y_eq1  
        del Pr_nt_xa_p_y_eq0  
        del Pr_nt_xa_x_c_eq1  
        del Pr_nt_xa_x_c_eq0  
        del Pr_nt_y_x_c_eq1   
        del Pr_nt_y_x_c_eq0   
        del Pr_nt_c_eq1       
        del Pr_nt_c_eq0       
        del diff_Pr_nt_x_eq1     
        del diff_Pr_nt_x_eq0     
        del diff_Pr_nt_y_eq1     
        del diff_Pr_nt_y_eq0     
        del diff_Pr_nt_xn1_eq1   
        del diff_Pr_nt_xn1_eq0   
        del diff_Pr_nt_yn1_eq1   
        del diff_Pr_nt_yn1_eq0   
        del diff_Pr_nt_xa_p_y_eq1
        del diff_Pr_nt_xa_p_y_eq0
        del diff_Pr_nt_xa_x_c_eq1
        del diff_Pr_nt_xa_x_c_eq0
        del diff_Pr_nt_y_x_c_eq1 
        del diff_Pr_nt_y_x_c_eq0 
        del diff_Pr_nt_c_eq1     
        del diff_Pr_nt_c_eq0     
    logfile_1xor.close()
    if XORN >= 2:
        for xi in range(idx, idx+1):
            for xi2 in range(xi+1, BLOCK_SIZE):
                cur_neutral_bits_2xor = [xi, xi2]
                Pr_nt_x_eq1       = div0(AND_plain_x_2xor[NBi_2xor]                           , AND_plain_x_all)                   
                Pr_nt_x_eq0       = div0(Full_2xor[NBi_2xor] - AND_plain_x_2xor[NBi_2xor]     , cnt_correct - AND_plain_x_all)     
                Pr_nt_y_eq1       = div0(AND_plain_y_2xor[NBi_2xor]                           , AND_plain_y_all)                   
                Pr_nt_y_eq0       = div0(Full_2xor[NBi_2xor] - AND_plain_y_2xor[NBi_2xor]     , cnt_correct - AND_plain_y_all)     
                Pr_nt_xn1_eq1     = div0(AND_plain_xn1_2xor[NBi_2xor]                         , AND_plain_xn1_all)                 
                Pr_nt_xn1_eq0     = div0(Full_2xor[NBi_2xor] - AND_plain_xn1_2xor[NBi_2xor]   , cnt_correct - AND_plain_xn1_all)   
                Pr_nt_yn1_eq1     = div0(AND_plain_yn1_2xor[NBi_2xor]                         , AND_plain_yn1_all)                 
                Pr_nt_yn1_eq0     = div0(Full_2xor[NBi_2xor] - AND_plain_yn1_2xor[NBi_2xor]   , cnt_correct - AND_plain_yn1_all)   
                Pr_nt_xa_p_y_eq1  = div0(AND_plain_x7_p_y_2xor[NBi_2xor]                      , AND_plain_x7_p_y_all)              
                Pr_nt_xa_p_y_eq0  = div0(Full_2xor[NBi_2xor] - AND_plain_x7_p_y_2xor[NBi_2xor], cnt_correct - AND_plain_x7_p_y_all)
                Pr_nt_xa_x_c_eq1  = div0(AND_xa_x_c_2xor[NBi_2xor]                            , AND_xa_x_c_all)                    
                Pr_nt_xa_x_c_eq0  = div0(Full_2xor[NBi_2xor] - AND_xa_x_c_2xor[NBi_2xor]      , cnt_correct - AND_xa_x_c_all)      
                Pr_nt_y_x_c_eq1   = div0(AND_y_x_c_2xor[NBi_2xor]                             , AND_y_x_c_all)                     
                Pr_nt_y_x_c_eq0   = div0(Full_2xor[NBi_2xor] - AND_y_x_c_2xor[NBi_2xor]       , cnt_correct - AND_y_x_c_all)       
                Pr_nt_c_eq1       = div0(AND_c_2xor[NBi_2xor]                                 , AND_c_all)                         
                Pr_nt_c_eq0       = div0(Full_2xor[NBi_2xor] - AND_c_2xor[NBi_2xor]           , cnt_correct - AND_c_all)           
                if Full_2xor[NBi_2xor]/cnt_correct > threshold_freq_low:
                    print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), 
                    "  Pr_x_nt_eq1 "     , div0(AND_plain_x_2xor[NBi_2xor]                                  , Full_2xor[NBi_2xor]), "  Pr_nt_x_eq1 "     , Pr_nt_x_eq1           ,
                    "  Pr_x_nt_eq0 "     , div0(Full_2xor[NBi_2xor] - AND_plain_x_2xor[NBi_2xor]            , Full_2xor[NBi_2xor]), "  Pr_nt_x_eq0 "     , Pr_nt_x_eq0           ,
                    "  Pr_y_nt_eq1 "     , div0(AND_plain_y_2xor[NBi_2xor]                                  , Full_2xor[NBi_2xor]), "  Pr_nt_y_eq1 "     , Pr_nt_y_eq1           ,
                    "  Pr_y_nt_eq0 "     , div0(Full_2xor[NBi_2xor] - AND_plain_y_2xor[NBi_2xor]            , Full_2xor[NBi_2xor]), "  Pr_nt_y_eq0 "     , Pr_nt_y_eq0           ,
                    "  Pr_xn1_nt_eq1 "   , div0(AND_plain_xn1_2xor[NBi_2xor]                                , Full_2xor[NBi_2xor]), "  Pr_nt_xn1_eq1 "   , Pr_nt_xn1_eq1         ,
                    "  Pr_xn1_nt_eq0 "   , div0(Full_2xor[NBi_2xor] - AND_plain_xn1_2xor[NBi_2xor]          , Full_2xor[NBi_2xor]), "  Pr_nt_xn1_eq0 "   , Pr_nt_xn1_eq0         ,
                    "  Pr_yn1_nt_eq1 "   , div0(AND_plain_yn1_2xor[NBi_2xor]                                , Full_2xor[NBi_2xor]), "  Pr_nt_yn1_eq1 "   , Pr_nt_yn1_eq1         ,
                    "  Pr_yn1_nt_eq0 "   , div0(Full_2xor[NBi_2xor] - AND_plain_yn1_2xor[NBi_2xor]          , Full_2xor[NBi_2xor]), "  Pr_nt_yn1_eq0 "   , Pr_nt_yn1_eq0         ,
                    "  Pr_xa_p_y_nt_eq1 ", div0(AND_plain_x7_p_y_2xor[NBi_2xor]                             , Full_2xor[NBi_2xor]), "  Pr_nt_xa_p_y_eq1 ", Pr_nt_xa_p_y_eq1      ,
                    "  Pr_xa_p_y_nt_eq0 ", div0(Full_2xor[NBi_2xor] - AND_plain_x7_p_y_2xor[NBi_2xor]       , Full_2xor[NBi_2xor]), "  Pr_nt_xa_p_y_eq0 ", Pr_nt_xa_p_y_eq0      ,
                    "  Pr_xa_x_c_nt_eq1 ", div0(AND_xa_x_c_2xor[NBi_2xor]                                   , Full_2xor[NBi_2xor]), "  Pr_nt_xa_x_c_eq1 ", Pr_nt_xa_x_c_eq1,
                    "  Pr_xa_x_c_nt_eq0 ", div0(Full_2xor[NBi_2xor] - AND_xa_x_c_2xor[NBi_2xor]             , Full_2xor[NBi_2xor]), "  Pr_nt_xa_x_c_eq0 ", Pr_nt_xa_x_c_eq0,
                    "  Pr_y_x_c_nt_eq1 " , div0(AND_y_x_c_2xor[NBi_2xor]                                    , Full_2xor[NBi_2xor]), "  Pr_nt_y_x_c_eq1 " , Pr_nt_y_x_c_eq1 ,
                    "  Pr_y_x_c_nt_eq0 " , div0(Full_2xor[NBi_2xor] - AND_y_x_c_2xor[NBi_2xor]              , Full_2xor[NBi_2xor]), "  Pr_nt_y_x_c_eq0 " , Pr_nt_y_x_c_eq0 ,
                    "  Pr_c_nt_eq1 "     , div0(AND_c_2xor[NBi_2xor]                                        , Full_2xor[NBi_2xor]), "  Pr_nt_c_eq1 "     , Pr_nt_c_eq1     ,
                    "  Pr_c_nt_eq0 "     , div0(Full_2xor[NBi_2xor] - AND_c_2xor[NBi_2xor]                  , Full_2xor[NBi_2xor]), "  Pr_nt_c_eq0 "     , Pr_nt_c_eq0     ,
                    file=logfile_2xor, flush=True)
                if Full_2xor[NBi_2xor]/cnt_correct > threshold_freq:
                    print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), 
                    "  Pr_x_nt_eq1 "     , div0(AND_plain_x_2xor[NBi_2xor]                                  , Full_2xor[NBi_2xor]), "  Pr_nt_x_eq1 "     , Pr_nt_x_eq1           ,
                    "  Pr_x_nt_eq0 "     , div0(Full_2xor[NBi_2xor] - AND_plain_x_2xor[NBi_2xor]            , Full_2xor[NBi_2xor]), "  Pr_nt_x_eq0 "     , Pr_nt_x_eq0           ,
                    "  Pr_y_nt_eq1 "     , div0(AND_plain_y_2xor[NBi_2xor]                                  , Full_2xor[NBi_2xor]), "  Pr_nt_y_eq1 "     , Pr_nt_y_eq1           ,
                    "  Pr_y_nt_eq0 "     , div0(Full_2xor[NBi_2xor] - AND_plain_y_2xor[NBi_2xor]            , Full_2xor[NBi_2xor]), "  Pr_nt_y_eq0 "     , Pr_nt_y_eq0           ,
                    "  Pr_xn1_nt_eq1 "   , div0(AND_plain_xn1_2xor[NBi_2xor]                                , Full_2xor[NBi_2xor]), "  Pr_nt_xn1_eq1 "   , Pr_nt_xn1_eq1         ,
                    "  Pr_xn1_nt_eq0 "   , div0(Full_2xor[NBi_2xor] - AND_plain_xn1_2xor[NBi_2xor]          , Full_2xor[NBi_2xor]), "  Pr_nt_xn1_eq0 "   , Pr_nt_xn1_eq0         ,
                    "  Pr_yn1_nt_eq1 "   , div0(AND_plain_yn1_2xor[NBi_2xor]                                , Full_2xor[NBi_2xor]), "  Pr_nt_yn1_eq1 "   , Pr_nt_yn1_eq1         ,
                    "  Pr_yn1_nt_eq0 "   , div0(Full_2xor[NBi_2xor] - AND_plain_yn1_2xor[NBi_2xor]          , Full_2xor[NBi_2xor]), "  Pr_nt_yn1_eq0 "   , Pr_nt_yn1_eq0         ,
                    "  Pr_xa_p_y_nt_eq1 ", div0(AND_plain_x7_p_y_2xor[NBi_2xor]                             , Full_2xor[NBi_2xor]), "  Pr_nt_xa_p_y_eq1 ", Pr_nt_xa_p_y_eq1      ,
                    "  Pr_xa_p_y_nt_eq0 ", div0(Full_2xor[NBi_2xor] - AND_plain_x7_p_y_2xor[NBi_2xor]       , Full_2xor[NBi_2xor]), "  Pr_nt_xa_p_y_eq0 ", Pr_nt_xa_p_y_eq0      ,
                    "  Pr_xa_x_c_nt_eq1 ", div0(AND_xa_x_c_2xor[NBi_2xor]                                   , Full_2xor[NBi_2xor]), "  Pr_nt_xa_x_c_eq1 ", Pr_nt_xa_x_c_eq1,
                    "  Pr_xa_x_c_nt_eq0 ", div0(Full_2xor[NBi_2xor] - AND_xa_x_c_2xor[NBi_2xor]             , Full_2xor[NBi_2xor]), "  Pr_nt_xa_x_c_eq0 ", Pr_nt_xa_x_c_eq0,
                    "  Pr_y_x_c_nt_eq1 " , div0(AND_y_x_c_2xor[NBi_2xor]                                    , Full_2xor[NBi_2xor]), "  Pr_nt_y_x_c_eq1 " , Pr_nt_y_x_c_eq1 ,
                    "  Pr_y_x_c_nt_eq0 " , div0(Full_2xor[NBi_2xor] - AND_y_x_c_2xor[NBi_2xor]              , Full_2xor[NBi_2xor]), "  Pr_nt_y_x_c_eq0 " , Pr_nt_y_x_c_eq0 ,
                    "  Pr_c_nt_eq1 "     , div0(AND_c_2xor[NBi_2xor]                                        , Full_2xor[NBi_2xor]), "  Pr_nt_c_eq1 "     , Pr_nt_c_eq1     ,
                    "  Pr_c_nt_eq0 "     , div0(Full_2xor[NBi_2xor] - AND_c_2xor[NBi_2xor]                  , Full_2xor[NBi_2xor]), "  Pr_nt_c_eq0 "     , Pr_nt_c_eq0     ,
                    file=logfile_alls, flush=True)
                diff_Pr_nt_x_eq1       = Pr_nt_x_eq1       - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_x_eq1       = np.max(Pr_nt_x_eq1     );
                diff_Pr_nt_x_eq0       = Pr_nt_x_eq0       - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_x_eq0       = np.max(Pr_nt_x_eq0     );
                diff_Pr_nt_y_eq1       = Pr_nt_y_eq1       - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_y_eq1       = np.max(Pr_nt_y_eq1     );
                diff_Pr_nt_y_eq0       = Pr_nt_y_eq0       - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_y_eq0       = np.max(Pr_nt_y_eq0     );
                diff_Pr_nt_xn1_eq1     = Pr_nt_xn1_eq1     - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_xn1_eq1     = np.max(Pr_nt_xn1_eq1   );
                diff_Pr_nt_xn1_eq0     = Pr_nt_xn1_eq0     - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_xn1_eq0     = np.max(Pr_nt_xn1_eq0   );
                diff_Pr_nt_yn1_eq1     = Pr_nt_yn1_eq1     - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_yn1_eq1     = np.max(Pr_nt_yn1_eq1   );
                diff_Pr_nt_yn1_eq0     = Pr_nt_yn1_eq0     - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_yn1_eq0     = np.max(Pr_nt_yn1_eq0   );
                diff_Pr_nt_xa_p_y_eq1  = Pr_nt_xa_p_y_eq1  - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_xa_p_y_eq1  = np.max(Pr_nt_xa_p_y_eq1);
                diff_Pr_nt_xa_p_y_eq0  = Pr_nt_xa_p_y_eq0  - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_xa_p_y_eq0  = np.max(Pr_nt_xa_p_y_eq0);
                diff_Pr_nt_xa_x_c_eq1  = Pr_nt_xa_x_c_eq1  - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_xa_x_c_eq1  = np.max(Pr_nt_xa_x_c_eq1);
                diff_Pr_nt_xa_x_c_eq0  = Pr_nt_xa_x_c_eq0  - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_xa_x_c_eq0  = np.max(Pr_nt_xa_x_c_eq0);
                diff_Pr_nt_y_x_c_eq1   = Pr_nt_y_x_c_eq1   - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_y_x_c_eq1   = np.max(Pr_nt_y_x_c_eq1 );
                diff_Pr_nt_y_x_c_eq0   = Pr_nt_y_x_c_eq0   - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_y_x_c_eq0   = np.max(Pr_nt_y_x_c_eq0 );
                diff_Pr_nt_c_eq1       = Pr_nt_c_eq1       - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_c_eq1       = np.max(Pr_nt_c_eq1     );
                diff_Pr_nt_c_eq0       = Pr_nt_c_eq0       - Full_2xor[NBi_2xor]/cnt_correct;  max_Pr_nt_c_eq0       = np.max(Pr_nt_c_eq0     );
                if len(diff_Pr_nt_x_eq1     [diff_Pr_nt_x_eq1      >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_x_eq1     ), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_x_eq1      " , Pr_nt_x_eq1     , file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_x_eq0     [diff_Pr_nt_x_eq0      >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_x_eq0     ), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_x_eq0      " , Pr_nt_x_eq0     , file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_y_eq1     [diff_Pr_nt_y_eq1      >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_eq1     ), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_y_eq1      " , Pr_nt_y_eq1     , file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_y_eq0     [diff_Pr_nt_y_eq0      >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_eq0     ), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_y_eq0      " , Pr_nt_y_eq0     , file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_xn1_eq1   [diff_Pr_nt_xn1_eq1    >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xn1_eq1   ), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_xn1_eq1    " , Pr_nt_xn1_eq1   , file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_xn1_eq0   [diff_Pr_nt_xn1_eq0    >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xn1_eq0   ), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_xn1_eq0    " , Pr_nt_xn1_eq0   , file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_yn1_eq1   [diff_Pr_nt_yn1_eq1    >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_yn1_eq1   ), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_yn1_eq1    " , Pr_nt_yn1_eq1   , file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_yn1_eq0   [diff_Pr_nt_yn1_eq0    >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_yn1_eq0   ), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_yn1_eq0    " , Pr_nt_yn1_eq0   , file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_xa_p_y_eq1[diff_Pr_nt_xa_p_y_eq1 >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_p_y_eq1), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_xa_p_y_eq1 " , Pr_nt_xa_p_y_eq1, file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_xa_p_y_eq0[diff_Pr_nt_xa_p_y_eq0 >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_p_y_eq0), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_xa_p_y_eq0 " , Pr_nt_xa_p_y_eq0, file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_xa_x_c_eq1[diff_Pr_nt_xa_x_c_eq1 >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_x_c_eq1), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_xa_x_c_eq1 " , Pr_nt_xa_x_c_eq1, file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_xa_x_c_eq0[diff_Pr_nt_xa_x_c_eq0 >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_x_c_eq0), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_xa_x_c_eq0 " , Pr_nt_xa_x_c_eq0, file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_y_x_c_eq1 [diff_Pr_nt_y_x_c_eq1  >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_x_c_eq1 ), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_y_x_c_eq1  " , Pr_nt_y_x_c_eq1 , file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_y_x_c_eq0 [diff_Pr_nt_y_x_c_eq0  >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_x_c_eq0 ), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_y_x_c_eq0  " , Pr_nt_y_x_c_eq0 , file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_c_eq1     [diff_Pr_nt_c_eq1      >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_c_eq1     ), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_c_eq1      " , Pr_nt_c_eq1     , file=logfile_cnds, flush=True)
                if len(diff_Pr_nt_c_eq0     [diff_Pr_nt_c_eq0      >= GPT]) != 0:
                    print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_2xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_c_eq0     ), ' from ', '{:0.3f}'.format(Full_2xor[NBi_2xor]/cnt_correct), " Pr_nt_c_eq0      " , Pr_nt_c_eq0     , file=logfile_cnds, flush=True)
                NBi_2xor = NBi_2xor + 1
                del Pr_nt_x_eq1       
                del Pr_nt_x_eq0       
                del Pr_nt_y_eq1       
                del Pr_nt_y_eq0       
                del Pr_nt_xn1_eq1     
                del Pr_nt_xn1_eq0     
                del Pr_nt_yn1_eq1     
                del Pr_nt_yn1_eq0     
                del Pr_nt_xa_p_y_eq1  
                del Pr_nt_xa_p_y_eq0  
                del Pr_nt_xa_x_c_eq1  
                del Pr_nt_xa_x_c_eq0  
                del Pr_nt_y_x_c_eq1   
                del Pr_nt_y_x_c_eq0   
                del Pr_nt_c_eq1       
                del Pr_nt_c_eq0       
                del diff_Pr_nt_x_eq1     
                del diff_Pr_nt_x_eq0     
                del diff_Pr_nt_y_eq1     
                del diff_Pr_nt_y_eq0     
                del diff_Pr_nt_xn1_eq1   
                del diff_Pr_nt_xn1_eq0   
                del diff_Pr_nt_yn1_eq1   
                del diff_Pr_nt_yn1_eq0   
                del diff_Pr_nt_xa_p_y_eq1
                del diff_Pr_nt_xa_p_y_eq0
                del diff_Pr_nt_xa_x_c_eq1
                del diff_Pr_nt_xa_x_c_eq0
                del diff_Pr_nt_y_x_c_eq1 
                del diff_Pr_nt_y_x_c_eq0 
                del diff_Pr_nt_c_eq1     
                del diff_Pr_nt_c_eq0     
                #
        logfile_2xor.close()
    if XORN >= 3:
        for xi in range(idx, idx+1):
            for xi2 in range(xi+1, BLOCK_SIZE):
                for xi3 in range(xi2+1, BLOCK_SIZE):
                    cur_neutral_bits_3xor = [xi, xi2, xi3]
                    Pr_nt_x_eq1       = div0(AND_plain_x_3xor[NBi_3xor]                           , AND_plain_x_all)                   
                    Pr_nt_x_eq0       = div0(Full_3xor[NBi_3xor] - AND_plain_x_3xor[NBi_3xor]     , cnt_correct - AND_plain_x_all)     
                    Pr_nt_y_eq1       = div0(AND_plain_y_3xor[NBi_3xor]                           , AND_plain_y_all)                   
                    Pr_nt_y_eq0       = div0(Full_3xor[NBi_3xor] - AND_plain_y_3xor[NBi_3xor]     , cnt_correct - AND_plain_y_all)     
                    Pr_nt_xn1_eq1     = div0(AND_plain_xn1_3xor[NBi_3xor]                         , AND_plain_xn1_all)                 
                    Pr_nt_xn1_eq0     = div0(Full_3xor[NBi_3xor] - AND_plain_xn1_3xor[NBi_3xor]   , cnt_correct - AND_plain_xn1_all)   
                    Pr_nt_yn1_eq1     = div0(AND_plain_yn1_3xor[NBi_3xor]                         , AND_plain_yn1_all)                 
                    Pr_nt_yn1_eq0     = div0(Full_3xor[NBi_3xor] - AND_plain_yn1_3xor[NBi_3xor]   , cnt_correct - AND_plain_yn1_all)   
                    Pr_nt_xa_p_y_eq1  = div0(AND_plain_x7_p_y_3xor[NBi_3xor]                      , AND_plain_x7_p_y_all)              
                    Pr_nt_xa_p_y_eq0  = div0(Full_3xor[NBi_3xor] - AND_plain_x7_p_y_3xor[NBi_3xor], cnt_correct - AND_plain_x7_p_y_all)
                    Pr_nt_xa_x_c_eq1  = div0(AND_xa_x_c_3xor[NBi_3xor]                            , AND_xa_x_c_all)                    
                    Pr_nt_xa_x_c_eq0  = div0(Full_3xor[NBi_3xor] - AND_xa_x_c_3xor[NBi_3xor]      , cnt_correct - AND_xa_x_c_all)      
                    Pr_nt_y_x_c_eq1   = div0(AND_y_x_c_3xor[NBi_3xor]                             , AND_y_x_c_all)                     
                    Pr_nt_y_x_c_eq0   = div0(Full_3xor[NBi_3xor] - AND_y_x_c_3xor[NBi_3xor]       , cnt_correct - AND_y_x_c_all)       
                    Pr_nt_c_eq1       = div0(AND_c_3xor[NBi_3xor]                                 , AND_c_all)                         
                    Pr_nt_c_eq0       = div0(Full_3xor[NBi_3xor] - AND_c_3xor[NBi_3xor]           , cnt_correct - AND_c_all)           
                    if Full_3xor[NBi_3xor]/cnt_correct > threshold_freq_low:
                        print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), 
                        "  Pr_x_nt_eq1 "     , div0(AND_plain_x_3xor[NBi_3xor]                                  , Full_3xor[NBi_3xor]), "  Pr_nt_x_eq1 "     , Pr_nt_x_eq1           ,
                        "  Pr_x_nt_eq0 "     , div0(Full_3xor[NBi_3xor] - AND_plain_x_3xor[NBi_3xor]            , Full_3xor[NBi_3xor]), "  Pr_nt_x_eq0 "     , Pr_nt_x_eq0           ,
                        "  Pr_y_nt_eq1 "     , div0(AND_plain_y_3xor[NBi_3xor]                                  , Full_3xor[NBi_3xor]), "  Pr_nt_y_eq1 "     , Pr_nt_y_eq1           ,
                        "  Pr_y_nt_eq0 "     , div0(Full_3xor[NBi_3xor] - AND_plain_y_3xor[NBi_3xor]            , Full_3xor[NBi_3xor]), "  Pr_nt_y_eq0 "     , Pr_nt_y_eq0           ,
                        "  Pr_xn1_nt_eq1 "   , div0(AND_plain_xn1_3xor[NBi_3xor]                                , Full_3xor[NBi_3xor]), "  Pr_nt_xn1_eq1 "   , Pr_nt_xn1_eq1         ,
                        "  Pr_xn1_nt_eq0 "   , div0(Full_3xor[NBi_3xor] - AND_plain_xn1_3xor[NBi_3xor]          , Full_3xor[NBi_3xor]), "  Pr_nt_xn1_eq0 "   , Pr_nt_xn1_eq0         ,
                        "  Pr_yn1_nt_eq1 "   , div0(AND_plain_yn1_3xor[NBi_3xor]                                , Full_3xor[NBi_3xor]), "  Pr_nt_yn1_eq1 "   , Pr_nt_yn1_eq1         ,
                        "  Pr_yn1_nt_eq0 "   , div0(Full_3xor[NBi_3xor] - AND_plain_yn1_3xor[NBi_3xor]          , Full_3xor[NBi_3xor]), "  Pr_nt_yn1_eq0 "   , Pr_nt_yn1_eq0         ,
                        "  Pr_xa_p_y_nt_eq1 ", div0(AND_plain_x7_p_y_3xor[NBi_3xor]                             , Full_3xor[NBi_3xor]), "  Pr_nt_xa_p_y_eq1 ", Pr_nt_xa_p_y_eq1      ,
                        "  Pr_xa_p_y_nt_eq0 ", div0(Full_3xor[NBi_3xor] - AND_plain_x7_p_y_3xor[NBi_3xor]       , Full_3xor[NBi_3xor]), "  Pr_nt_xa_p_y_eq0 ", Pr_nt_xa_p_y_eq0      ,
                        "  Pr_xa_x_c_nt_eq1 ", div0(AND_xa_x_c_3xor[NBi_3xor]                                   , Full_3xor[NBi_3xor]), "  Pr_nt_xa_x_c_eq1 ", Pr_nt_xa_x_c_eq1,
                        "  Pr_xa_x_c_nt_eq0 ", div0(Full_3xor[NBi_3xor] - AND_xa_x_c_3xor[NBi_3xor]             , Full_3xor[NBi_3xor]), "  Pr_nt_xa_x_c_eq0 ", Pr_nt_xa_x_c_eq0,
                        "  Pr_y_x_c_nt_eq1 " , div0(AND_y_x_c_3xor[NBi_3xor]                                    , Full_3xor[NBi_3xor]), "  Pr_nt_y_x_c_eq1 " , Pr_nt_y_x_c_eq1 ,
                        "  Pr_y_x_c_nt_eq0 " , div0(Full_3xor[NBi_3xor] - AND_y_x_c_3xor[NBi_3xor]              , Full_3xor[NBi_3xor]), "  Pr_nt_y_x_c_eq0 " , Pr_nt_y_x_c_eq0 ,
                        "  Pr_c_nt_eq1 "     , div0(AND_c_3xor[NBi_3xor]                                        , Full_3xor[NBi_3xor]), "  Pr_nt_c_eq1 "     , Pr_nt_c_eq1     ,
                        "  Pr_c_nt_eq0 "     , div0(Full_3xor[NBi_3xor] - AND_c_3xor[NBi_3xor]                  , Full_3xor[NBi_3xor]), "  Pr_nt_c_eq0 "     , Pr_nt_c_eq0     ,
                        file=logfile_3xor, flush=True)
                    if Full_3xor[NBi_3xor]/cnt_correct > threshold_freq:
                        print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), 
                        "  Pr_x_nt_eq1 "     , div0(AND_plain_x_3xor[NBi_3xor]                                  , Full_3xor[NBi_3xor]), "  Pr_nt_x_eq1 "     , Pr_nt_x_eq1           ,
                        "  Pr_x_nt_eq0 "     , div0(Full_3xor[NBi_3xor] - AND_plain_x_3xor[NBi_3xor]            , Full_3xor[NBi_3xor]), "  Pr_nt_x_eq0 "     , Pr_nt_x_eq0           ,
                        "  Pr_y_nt_eq1 "     , div0(AND_plain_y_3xor[NBi_3xor]                                  , Full_3xor[NBi_3xor]), "  Pr_nt_y_eq1 "     , Pr_nt_y_eq1           ,
                        "  Pr_y_nt_eq0 "     , div0(Full_3xor[NBi_3xor] - AND_plain_y_3xor[NBi_3xor]            , Full_3xor[NBi_3xor]), "  Pr_nt_y_eq0 "     , Pr_nt_y_eq0           ,
                        "  Pr_xn1_nt_eq1 "   , div0(AND_plain_xn1_3xor[NBi_3xor]                                , Full_3xor[NBi_3xor]), "  Pr_nt_xn1_eq1 "   , Pr_nt_xn1_eq1         ,
                        "  Pr_xn1_nt_eq0 "   , div0(Full_3xor[NBi_3xor] - AND_plain_xn1_3xor[NBi_3xor]          , Full_3xor[NBi_3xor]), "  Pr_nt_xn1_eq0 "   , Pr_nt_xn1_eq0         ,
                        "  Pr_yn1_nt_eq1 "   , div0(AND_plain_yn1_3xor[NBi_3xor]                                , Full_3xor[NBi_3xor]), "  Pr_nt_yn1_eq1 "   , Pr_nt_yn1_eq1         ,
                        "  Pr_yn1_nt_eq0 "   , div0(Full_3xor[NBi_3xor] - AND_plain_yn1_3xor[NBi_3xor]          , Full_3xor[NBi_3xor]), "  Pr_nt_yn1_eq0 "   , Pr_nt_yn1_eq0         ,
                        "  Pr_xa_p_y_nt_eq1 ", div0(AND_plain_x7_p_y_3xor[NBi_3xor]                             , Full_3xor[NBi_3xor]), "  Pr_nt_xa_p_y_eq1 ", Pr_nt_xa_p_y_eq1      ,
                        "  Pr_xa_p_y_nt_eq0 ", div0(Full_3xor[NBi_3xor] - AND_plain_x7_p_y_3xor[NBi_3xor]       , Full_3xor[NBi_3xor]), "  Pr_nt_xa_p_y_eq0 ", Pr_nt_xa_p_y_eq0      ,
                        "  Pr_xa_x_c_nt_eq1 ", div0(AND_xa_x_c_3xor[NBi_3xor]                                   , Full_3xor[NBi_3xor]), "  Pr_nt_xa_x_c_eq1 ", Pr_nt_xa_x_c_eq1,
                        "  Pr_xa_x_c_nt_eq0 ", div0(Full_3xor[NBi_3xor] - AND_xa_x_c_3xor[NBi_3xor]             , Full_3xor[NBi_3xor]), "  Pr_nt_xa_x_c_eq0 ", Pr_nt_xa_x_c_eq0,
                        "  Pr_y_x_c_nt_eq1 " , div0(AND_y_x_c_3xor[NBi_3xor]                                    , Full_3xor[NBi_3xor]), "  Pr_nt_y_x_c_eq1 " , Pr_nt_y_x_c_eq1 ,
                        "  Pr_y_x_c_nt_eq0 " , div0(Full_3xor[NBi_3xor] - AND_y_x_c_3xor[NBi_3xor]              , Full_3xor[NBi_3xor]), "  Pr_nt_y_x_c_eq0 " , Pr_nt_y_x_c_eq0 ,
                        "  Pr_c_nt_eq1 "     , div0(AND_c_3xor[NBi_3xor]                                        , Full_3xor[NBi_3xor]), "  Pr_nt_c_eq1 "     , Pr_nt_c_eq1     ,
                        "  Pr_c_nt_eq0 "     , div0(Full_3xor[NBi_3xor] - AND_c_3xor[NBi_3xor]                  , Full_3xor[NBi_3xor]), "  Pr_nt_c_eq0 "     , Pr_nt_c_eq0     ,
                        file=logfile_alls, flush=True)
                    diff_Pr_nt_x_eq1       = Pr_nt_x_eq1       - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_x_eq1       = np.max(Pr_nt_x_eq1     );
                    diff_Pr_nt_x_eq0       = Pr_nt_x_eq0       - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_x_eq0       = np.max(Pr_nt_x_eq0     );
                    diff_Pr_nt_y_eq1       = Pr_nt_y_eq1       - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_y_eq1       = np.max(Pr_nt_y_eq1     );
                    diff_Pr_nt_y_eq0       = Pr_nt_y_eq0       - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_y_eq0       = np.max(Pr_nt_y_eq0     );
                    diff_Pr_nt_xn1_eq1     = Pr_nt_xn1_eq1     - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_xn1_eq1     = np.max(Pr_nt_xn1_eq1   );
                    diff_Pr_nt_xn1_eq0     = Pr_nt_xn1_eq0     - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_xn1_eq0     = np.max(Pr_nt_xn1_eq0   );
                    diff_Pr_nt_yn1_eq1     = Pr_nt_yn1_eq1     - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_yn1_eq1     = np.max(Pr_nt_yn1_eq1   );
                    diff_Pr_nt_yn1_eq0     = Pr_nt_yn1_eq0     - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_yn1_eq0     = np.max(Pr_nt_yn1_eq0   );
                    diff_Pr_nt_xa_p_y_eq1  = Pr_nt_xa_p_y_eq1  - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_xa_p_y_eq1  = np.max(Pr_nt_xa_p_y_eq1);
                    diff_Pr_nt_xa_p_y_eq0  = Pr_nt_xa_p_y_eq0  - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_xa_p_y_eq0  = np.max(Pr_nt_xa_p_y_eq0);
                    diff_Pr_nt_xa_x_c_eq1  = Pr_nt_xa_x_c_eq1  - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_xa_x_c_eq1  = np.max(Pr_nt_xa_x_c_eq1);
                    diff_Pr_nt_xa_x_c_eq0  = Pr_nt_xa_x_c_eq0  - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_xa_x_c_eq0  = np.max(Pr_nt_xa_x_c_eq0);
                    diff_Pr_nt_y_x_c_eq1   = Pr_nt_y_x_c_eq1   - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_y_x_c_eq1   = np.max(Pr_nt_y_x_c_eq1 );
                    diff_Pr_nt_y_x_c_eq0   = Pr_nt_y_x_c_eq0   - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_y_x_c_eq0   = np.max(Pr_nt_y_x_c_eq0 );
                    diff_Pr_nt_c_eq1       = Pr_nt_c_eq1       - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_c_eq1       = np.max(Pr_nt_c_eq1     );
                    diff_Pr_nt_c_eq0       = Pr_nt_c_eq0       - Full_3xor[NBi_3xor]/cnt_correct;  max_Pr_nt_c_eq0       = np.max(Pr_nt_c_eq0     );
                    if len(diff_Pr_nt_x_eq1     [diff_Pr_nt_x_eq1      >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_x_eq1     ), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_x_eq1      " , Pr_nt_x_eq1     , file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_x_eq0     [diff_Pr_nt_x_eq0      >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_x_eq0     ), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_x_eq0      " , Pr_nt_x_eq0     , file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_y_eq1     [diff_Pr_nt_y_eq1      >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_eq1     ), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_y_eq1      " , Pr_nt_y_eq1     , file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_y_eq0     [diff_Pr_nt_y_eq0      >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_eq0     ), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_y_eq0      " , Pr_nt_y_eq0     , file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_xn1_eq1   [diff_Pr_nt_xn1_eq1    >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xn1_eq1   ), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_xn1_eq1    " , Pr_nt_xn1_eq1   , file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_xn1_eq0   [diff_Pr_nt_xn1_eq0    >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xn1_eq0   ), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_xn1_eq0    " , Pr_nt_xn1_eq0   , file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_yn1_eq1   [diff_Pr_nt_yn1_eq1    >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_yn1_eq1   ), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_yn1_eq1    " , Pr_nt_yn1_eq1   , file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_yn1_eq0   [diff_Pr_nt_yn1_eq0    >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_yn1_eq0   ), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_yn1_eq0    " , Pr_nt_yn1_eq0   , file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_xa_p_y_eq1[diff_Pr_nt_xa_p_y_eq1 >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_p_y_eq1), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_xa_p_y_eq1 " , Pr_nt_xa_p_y_eq1, file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_xa_p_y_eq0[diff_Pr_nt_xa_p_y_eq0 >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_p_y_eq0), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_xa_p_y_eq0 " , Pr_nt_xa_p_y_eq0, file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_xa_x_c_eq1[diff_Pr_nt_xa_x_c_eq1 >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_x_c_eq1), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_xa_x_c_eq1 " , Pr_nt_xa_x_c_eq1, file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_xa_x_c_eq0[diff_Pr_nt_xa_x_c_eq0 >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_x_c_eq0), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_xa_x_c_eq0 " , Pr_nt_xa_x_c_eq0, file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_y_x_c_eq1 [diff_Pr_nt_y_x_c_eq1  >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_x_c_eq1 ), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_y_x_c_eq1  " , Pr_nt_y_x_c_eq1 , file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_y_x_c_eq0 [diff_Pr_nt_y_x_c_eq0  >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_x_c_eq0 ), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_y_x_c_eq0  " , Pr_nt_y_x_c_eq0 , file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_c_eq1     [diff_Pr_nt_c_eq1      >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_c_eq1     ), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_c_eq1      " , Pr_nt_c_eq1     , file=logfile_cnds, flush=True)
                    if len(diff_Pr_nt_c_eq0     [diff_Pr_nt_c_eq0      >= GPT]) != 0:
                        print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_3xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_c_eq0     ), ' from ', '{:0.3f}'.format(Full_3xor[NBi_3xor]/cnt_correct), " Pr_nt_c_eq0      " , Pr_nt_c_eq0     , file=logfile_cnds, flush=True)
                    NBi_3xor = NBi_3xor + 1
                    del Pr_nt_x_eq1       
                    del Pr_nt_x_eq0       
                    del Pr_nt_y_eq1       
                    del Pr_nt_y_eq0       
                    del Pr_nt_xn1_eq1     
                    del Pr_nt_xn1_eq0     
                    del Pr_nt_yn1_eq1     
                    del Pr_nt_yn1_eq0     
                    del Pr_nt_xa_p_y_eq1  
                    del Pr_nt_xa_p_y_eq0  
                    del Pr_nt_xa_x_c_eq1  
                    del Pr_nt_xa_x_c_eq0  
                    del Pr_nt_y_x_c_eq1   
                    del Pr_nt_y_x_c_eq0   
                    del Pr_nt_c_eq1       
                    del Pr_nt_c_eq0       
                    del diff_Pr_nt_x_eq1     
                    del diff_Pr_nt_x_eq0     
                    del diff_Pr_nt_y_eq1     
                    del diff_Pr_nt_y_eq0     
                    del diff_Pr_nt_xn1_eq1   
                    del diff_Pr_nt_xn1_eq0   
                    del diff_Pr_nt_yn1_eq1   
                    del diff_Pr_nt_yn1_eq0   
                    del diff_Pr_nt_xa_p_y_eq1
                    del diff_Pr_nt_xa_p_y_eq0
                    del diff_Pr_nt_xa_x_c_eq1
                    del diff_Pr_nt_xa_x_c_eq0
                    del diff_Pr_nt_y_x_c_eq1 
                    del diff_Pr_nt_y_x_c_eq0 
                    del diff_Pr_nt_c_eq1     
                    del diff_Pr_nt_c_eq0     
                    #
        logfile_3xor.close()
    if XORN >= 4:
        for xi in range(idx, idx+1):
            for xi2 in range(xi+1, BLOCK_SIZE):
                for xi3 in range(xi2+1, BLOCK_SIZE):
                    for xi4 in range(xi3+1, BLOCK_SIZE):
                        cur_neutral_bits_4xor = [xi, xi2, xi3, xi4]
                        Pr_nt_x_eq1       = div0(AND_plain_x_4xor[NBi_4xor]                           , AND_plain_x_all)                   
                        Pr_nt_x_eq0       = div0(Full_4xor[NBi_4xor] - AND_plain_x_4xor[NBi_4xor]     , cnt_correct - AND_plain_x_all)     
                        Pr_nt_y_eq1       = div0(AND_plain_y_4xor[NBi_4xor]                           , AND_plain_y_all)                   
                        Pr_nt_y_eq0       = div0(Full_4xor[NBi_4xor] - AND_plain_y_4xor[NBi_4xor]     , cnt_correct - AND_plain_y_all)     
                        Pr_nt_xn1_eq1     = div0(AND_plain_xn1_4xor[NBi_4xor]                         , AND_plain_xn1_all)                 
                        Pr_nt_xn1_eq0     = div0(Full_4xor[NBi_4xor] - AND_plain_xn1_4xor[NBi_4xor]   , cnt_correct - AND_plain_xn1_all)   
                        Pr_nt_yn1_eq1     = div0(AND_plain_yn1_4xor[NBi_4xor]                         , AND_plain_yn1_all)                 
                        Pr_nt_yn1_eq0     = div0(Full_4xor[NBi_4xor] - AND_plain_yn1_4xor[NBi_4xor]   , cnt_correct - AND_plain_yn1_all)   
                        Pr_nt_xa_p_y_eq1  = div0(AND_plain_x7_p_y_4xor[NBi_4xor]                      , AND_plain_x7_p_y_all)              
                        Pr_nt_xa_p_y_eq0  = div0(Full_4xor[NBi_4xor] - AND_plain_x7_p_y_4xor[NBi_4xor], cnt_correct - AND_plain_x7_p_y_all)
                        Pr_nt_xa_x_c_eq1  = div0(AND_xa_x_c_4xor[NBi_4xor]                            , AND_xa_x_c_all)                    
                        Pr_nt_xa_x_c_eq0  = div0(Full_4xor[NBi_4xor] - AND_xa_x_c_4xor[NBi_4xor]      , cnt_correct - AND_xa_x_c_all)      
                        Pr_nt_y_x_c_eq1   = div0(AND_y_x_c_4xor[NBi_4xor]                             , AND_y_x_c_all)                     
                        Pr_nt_y_x_c_eq0   = div0(Full_4xor[NBi_4xor] - AND_y_x_c_4xor[NBi_4xor]       , cnt_correct - AND_y_x_c_all)       
                        Pr_nt_c_eq1       = div0(AND_c_4xor[NBi_4xor]                                 , AND_c_all)                         
                        Pr_nt_c_eq0       = div0(Full_4xor[NBi_4xor] - AND_c_4xor[NBi_4xor]           , cnt_correct - AND_c_all)           
                        if Full_4xor[NBi_4xor]/cnt_correct > threshold_freq_low:
                            print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), 
                            "  Pr_x_nt_eq1 "     , div0(AND_plain_x_4xor[NBi_4xor]                                  , Full_4xor[NBi_4xor]), "  Pr_nt_x_eq1 "     , Pr_nt_x_eq1           ,
                            "  Pr_x_nt_eq0 "     , div0(Full_4xor[NBi_4xor] - AND_plain_x_4xor[NBi_4xor]            , Full_4xor[NBi_4xor]), "  Pr_nt_x_eq0 "     , Pr_nt_x_eq0           ,
                            "  Pr_y_nt_eq1 "     , div0(AND_plain_y_4xor[NBi_4xor]                                  , Full_4xor[NBi_4xor]), "  Pr_nt_y_eq1 "     , Pr_nt_y_eq1           ,
                            "  Pr_y_nt_eq0 "     , div0(Full_4xor[NBi_4xor] - AND_plain_y_4xor[NBi_4xor]            , Full_4xor[NBi_4xor]), "  Pr_nt_y_eq0 "     , Pr_nt_y_eq0           ,
                            "  Pr_xn1_nt_eq1 "   , div0(AND_plain_xn1_4xor[NBi_4xor]                                , Full_4xor[NBi_4xor]), "  Pr_nt_xn1_eq1 "   , Pr_nt_xn1_eq1         ,
                            "  Pr_xn1_nt_eq0 "   , div0(Full_4xor[NBi_4xor] - AND_plain_xn1_4xor[NBi_4xor]          , Full_4xor[NBi_4xor]), "  Pr_nt_xn1_eq0 "   , Pr_nt_xn1_eq0         ,
                            "  Pr_yn1_nt_eq1 "   , div0(AND_plain_yn1_4xor[NBi_4xor]                                , Full_4xor[NBi_4xor]), "  Pr_nt_yn1_eq1 "   , Pr_nt_yn1_eq1         ,
                            "  Pr_yn1_nt_eq0 "   , div0(Full_4xor[NBi_4xor] - AND_plain_yn1_4xor[NBi_4xor]          , Full_4xor[NBi_4xor]), "  Pr_nt_yn1_eq0 "   , Pr_nt_yn1_eq0         ,
                            "  Pr_xa_p_y_nt_eq1 ", div0(AND_plain_x7_p_y_4xor[NBi_4xor]                             , Full_4xor[NBi_4xor]), "  Pr_nt_xa_p_y_eq1 ", Pr_nt_xa_p_y_eq1      ,
                            "  Pr_xa_p_y_nt_eq0 ", div0(Full_4xor[NBi_4xor] - AND_plain_x7_p_y_4xor[NBi_4xor]       , Full_4xor[NBi_4xor]), "  Pr_nt_xa_p_y_eq0 ", Pr_nt_xa_p_y_eq0      ,
                            "  Pr_xa_x_c_nt_eq1 ", div0(AND_xa_x_c_4xor[NBi_4xor]                                   , Full_4xor[NBi_4xor]), "  Pr_nt_xa_x_c_eq1 ", Pr_nt_xa_x_c_eq1,
                            "  Pr_xa_x_c_nt_eq0 ", div0(Full_4xor[NBi_4xor] - AND_xa_x_c_4xor[NBi_4xor]             , Full_4xor[NBi_4xor]), "  Pr_nt_xa_x_c_eq0 ", Pr_nt_xa_x_c_eq0,
                            "  Pr_y_x_c_nt_eq1 " , div0(AND_y_x_c_4xor[NBi_4xor]                                    , Full_4xor[NBi_4xor]), "  Pr_nt_y_x_c_eq1 " , Pr_nt_y_x_c_eq1 ,
                            "  Pr_y_x_c_nt_eq0 " , div0(Full_4xor[NBi_4xor] - AND_y_x_c_4xor[NBi_4xor]              , Full_4xor[NBi_4xor]), "  Pr_nt_y_x_c_eq0 " , Pr_nt_y_x_c_eq0 ,
                            "  Pr_c_nt_eq1 "     , div0(AND_c_4xor[NBi_4xor]                                        , Full_4xor[NBi_4xor]), "  Pr_nt_c_eq1 "     , Pr_nt_c_eq1     ,
                            "  Pr_c_nt_eq0 "     , div0(Full_4xor[NBi_4xor] - AND_c_4xor[NBi_4xor]                  , Full_4xor[NBi_4xor]), "  Pr_nt_c_eq0 "     , Pr_nt_c_eq0     ,
                            file=logfile_4xor, flush=True)
                        if Full_4xor[NBi_4xor]/cnt_correct > threshold_freq:
                            print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), 
                            "  Pr_x_nt_eq1 "     , div0(AND_plain_x_4xor[NBi_4xor]                                  , Full_4xor[NBi_4xor]), "  Pr_nt_x_eq1 "     , Pr_nt_x_eq1           ,
                            "  Pr_x_nt_eq0 "     , div0(Full_4xor[NBi_4xor] - AND_plain_x_4xor[NBi_4xor]            , Full_4xor[NBi_4xor]), "  Pr_nt_x_eq0 "     , Pr_nt_x_eq0           ,
                            "  Pr_y_nt_eq1 "     , div0(AND_plain_y_4xor[NBi_4xor]                                  , Full_4xor[NBi_4xor]), "  Pr_nt_y_eq1 "     , Pr_nt_y_eq1           ,
                            "  Pr_y_nt_eq0 "     , div0(Full_4xor[NBi_4xor] - AND_plain_y_4xor[NBi_4xor]            , Full_4xor[NBi_4xor]), "  Pr_nt_y_eq0 "     , Pr_nt_y_eq0           ,
                            "  Pr_xn1_nt_eq1 "   , div0(AND_plain_xn1_4xor[NBi_4xor]                                , Full_4xor[NBi_4xor]), "  Pr_nt_xn1_eq1 "   , Pr_nt_xn1_eq1         ,
                            "  Pr_xn1_nt_eq0 "   , div0(Full_4xor[NBi_4xor] - AND_plain_xn1_4xor[NBi_4xor]          , Full_4xor[NBi_4xor]), "  Pr_nt_xn1_eq0 "   , Pr_nt_xn1_eq0         ,
                            "  Pr_yn1_nt_eq1 "   , div0(AND_plain_yn1_4xor[NBi_4xor]                                , Full_4xor[NBi_4xor]), "  Pr_nt_yn1_eq1 "   , Pr_nt_yn1_eq1         ,
                            "  Pr_yn1_nt_eq0 "   , div0(Full_4xor[NBi_4xor] - AND_plain_yn1_4xor[NBi_4xor]          , Full_4xor[NBi_4xor]), "  Pr_nt_yn1_eq0 "   , Pr_nt_yn1_eq0         ,
                            "  Pr_xa_p_y_nt_eq1 ", div0(AND_plain_x7_p_y_4xor[NBi_4xor]                             , Full_4xor[NBi_4xor]), "  Pr_nt_xa_p_y_eq1 ", Pr_nt_xa_p_y_eq1      ,
                            "  Pr_xa_p_y_nt_eq0 ", div0(Full_4xor[NBi_4xor] - AND_plain_x7_p_y_4xor[NBi_4xor]       , Full_4xor[NBi_4xor]), "  Pr_nt_xa_p_y_eq0 ", Pr_nt_xa_p_y_eq0      ,
                            "  Pr_xa_x_c_nt_eq1 ", div0(AND_xa_x_c_4xor[NBi_4xor]                                   , Full_4xor[NBi_4xor]), "  Pr_nt_xa_x_c_eq1 ", Pr_nt_xa_x_c_eq1,
                            "  Pr_xa_x_c_nt_eq0 ", div0(Full_4xor[NBi_4xor] - AND_xa_x_c_4xor[NBi_4xor]             , Full_4xor[NBi_4xor]), "  Pr_nt_xa_x_c_eq0 ", Pr_nt_xa_x_c_eq0,
                            "  Pr_y_x_c_nt_eq1 " , div0(AND_y_x_c_4xor[NBi_4xor]                                    , Full_4xor[NBi_4xor]), "  Pr_nt_y_x_c_eq1 " , Pr_nt_y_x_c_eq1 ,
                            "  Pr_y_x_c_nt_eq0 " , div0(Full_4xor[NBi_4xor] - AND_y_x_c_4xor[NBi_4xor]              , Full_4xor[NBi_4xor]), "  Pr_nt_y_x_c_eq0 " , Pr_nt_y_x_c_eq0 ,
                            "  Pr_c_nt_eq1 "     , div0(AND_c_4xor[NBi_4xor]                                        , Full_4xor[NBi_4xor]), "  Pr_nt_c_eq1 "     , Pr_nt_c_eq1     ,
                            "  Pr_c_nt_eq0 "     , div0(Full_4xor[NBi_4xor] - AND_c_4xor[NBi_4xor]                  , Full_4xor[NBi_4xor]), "  Pr_nt_c_eq0 "     , Pr_nt_c_eq0     ,
                            file=logfile_alls, flush=True)
                        diff_Pr_nt_x_eq1       = Pr_nt_x_eq1       - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_x_eq1       = np.max(Pr_nt_x_eq1     );
                        diff_Pr_nt_x_eq0       = Pr_nt_x_eq0       - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_x_eq0       = np.max(Pr_nt_x_eq0     );
                        diff_Pr_nt_y_eq1       = Pr_nt_y_eq1       - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_y_eq1       = np.max(Pr_nt_y_eq1     );
                        diff_Pr_nt_y_eq0       = Pr_nt_y_eq0       - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_y_eq0       = np.max(Pr_nt_y_eq0     );
                        diff_Pr_nt_xn1_eq1     = Pr_nt_xn1_eq1     - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_xn1_eq1     = np.max(Pr_nt_xn1_eq1   );
                        diff_Pr_nt_xn1_eq0     = Pr_nt_xn1_eq0     - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_xn1_eq0     = np.max(Pr_nt_xn1_eq0   );
                        diff_Pr_nt_yn1_eq1     = Pr_nt_yn1_eq1     - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_yn1_eq1     = np.max(Pr_nt_yn1_eq1   );
                        diff_Pr_nt_yn1_eq0     = Pr_nt_yn1_eq0     - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_yn1_eq0     = np.max(Pr_nt_yn1_eq0   );
                        diff_Pr_nt_xa_p_y_eq1  = Pr_nt_xa_p_y_eq1  - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_xa_p_y_eq1  = np.max(Pr_nt_xa_p_y_eq1);
                        diff_Pr_nt_xa_p_y_eq0  = Pr_nt_xa_p_y_eq0  - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_xa_p_y_eq0  = np.max(Pr_nt_xa_p_y_eq0);
                        diff_Pr_nt_xa_x_c_eq1  = Pr_nt_xa_x_c_eq1  - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_xa_x_c_eq1  = np.max(Pr_nt_xa_x_c_eq1);
                        diff_Pr_nt_xa_x_c_eq0  = Pr_nt_xa_x_c_eq0  - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_xa_x_c_eq0  = np.max(Pr_nt_xa_x_c_eq0);
                        diff_Pr_nt_y_x_c_eq1   = Pr_nt_y_x_c_eq1   - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_y_x_c_eq1   = np.max(Pr_nt_y_x_c_eq1 );
                        diff_Pr_nt_y_x_c_eq0   = Pr_nt_y_x_c_eq0   - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_y_x_c_eq0   = np.max(Pr_nt_y_x_c_eq0 );
                        diff_Pr_nt_c_eq1       = Pr_nt_c_eq1       - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_c_eq1       = np.max(Pr_nt_c_eq1     );
                        diff_Pr_nt_c_eq0       = Pr_nt_c_eq0       - Full_4xor[NBi_4xor]/cnt_correct;  max_Pr_nt_c_eq0       = np.max(Pr_nt_c_eq0     );
                        if len(diff_Pr_nt_x_eq1     [diff_Pr_nt_x_eq1      >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_x_eq1     ), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_x_eq1      " , Pr_nt_x_eq1     , file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_x_eq0     [diff_Pr_nt_x_eq0      >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_x_eq0     ), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_x_eq0      " , Pr_nt_x_eq0     , file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_y_eq1     [diff_Pr_nt_y_eq1      >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_eq1     ), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_y_eq1      " , Pr_nt_y_eq1     , file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_y_eq0     [diff_Pr_nt_y_eq0      >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_eq0     ), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_y_eq0      " , Pr_nt_y_eq0     , file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_xn1_eq1   [diff_Pr_nt_xn1_eq1    >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xn1_eq1   ), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_xn1_eq1    " , Pr_nt_xn1_eq1   , file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_xn1_eq0   [diff_Pr_nt_xn1_eq0    >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xn1_eq0   ), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_xn1_eq0    " , Pr_nt_xn1_eq0   , file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_yn1_eq1   [diff_Pr_nt_yn1_eq1    >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_yn1_eq1   ), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_yn1_eq1    " , Pr_nt_yn1_eq1   , file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_yn1_eq0   [diff_Pr_nt_yn1_eq0    >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_yn1_eq0   ), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_yn1_eq0    " , Pr_nt_yn1_eq0   , file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_xa_p_y_eq1[diff_Pr_nt_xa_p_y_eq1 >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_p_y_eq1), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_xa_p_y_eq1 " , Pr_nt_xa_p_y_eq1, file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_xa_p_y_eq0[diff_Pr_nt_xa_p_y_eq0 >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_p_y_eq0), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_xa_p_y_eq0 " , Pr_nt_xa_p_y_eq0, file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_xa_x_c_eq1[diff_Pr_nt_xa_x_c_eq1 >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_x_c_eq1), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_xa_x_c_eq1 " , Pr_nt_xa_x_c_eq1, file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_xa_x_c_eq0[diff_Pr_nt_xa_x_c_eq0 >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_x_c_eq0), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_xa_x_c_eq0 " , Pr_nt_xa_x_c_eq0, file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_y_x_c_eq1 [diff_Pr_nt_y_x_c_eq1  >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_x_c_eq1 ), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_y_x_c_eq1  " , Pr_nt_y_x_c_eq1 , file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_y_x_c_eq0 [diff_Pr_nt_y_x_c_eq0  >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_x_c_eq0 ), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_y_x_c_eq0  " , Pr_nt_y_x_c_eq0 , file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_c_eq1     [diff_Pr_nt_c_eq1      >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_c_eq1     ), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_c_eq1      " , Pr_nt_c_eq1     , file=logfile_cnds, flush=True)
                        if len(diff_Pr_nt_c_eq0     [diff_Pr_nt_c_eq0      >= GPT]) != 0:
                            print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_4xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_c_eq0     ), ' from ', '{:0.3f}'.format(Full_4xor[NBi_4xor]/cnt_correct), " Pr_nt_c_eq0      " , Pr_nt_c_eq0     , file=logfile_cnds, flush=True)
                        NBi_4xor = NBi_4xor + 1
                        del Pr_nt_x_eq1       
                        del Pr_nt_x_eq0       
                        del Pr_nt_y_eq1       
                        del Pr_nt_y_eq0       
                        del Pr_nt_xn1_eq1     
                        del Pr_nt_xn1_eq0     
                        del Pr_nt_yn1_eq1     
                        del Pr_nt_yn1_eq0     
                        del Pr_nt_xa_p_y_eq1  
                        del Pr_nt_xa_p_y_eq0  
                        del Pr_nt_xa_x_c_eq1  
                        del Pr_nt_xa_x_c_eq0  
                        del Pr_nt_y_x_c_eq1   
                        del Pr_nt_y_x_c_eq0   
                        del Pr_nt_c_eq1       
                        del Pr_nt_c_eq0       
                        del diff_Pr_nt_x_eq1     
                        del diff_Pr_nt_x_eq0     
                        del diff_Pr_nt_y_eq1     
                        del diff_Pr_nt_y_eq0     
                        del diff_Pr_nt_xn1_eq1   
                        del diff_Pr_nt_xn1_eq0   
                        del diff_Pr_nt_yn1_eq1   
                        del diff_Pr_nt_yn1_eq0   
                        del diff_Pr_nt_xa_p_y_eq1
                        del diff_Pr_nt_xa_p_y_eq0
                        del diff_Pr_nt_xa_x_c_eq1
                        del diff_Pr_nt_xa_x_c_eq0
                        del diff_Pr_nt_y_x_c_eq1 
                        del diff_Pr_nt_y_x_c_eq0 
                        del diff_Pr_nt_c_eq1     
                        del diff_Pr_nt_c_eq0     
        logfile_4xor.close()
    if XORN >= 5:
        for xi in range(idx, idx+1):
            for xi2 in range(xi+1, BLOCK_SIZE):
                for xi3 in range(xi2+1, BLOCK_SIZE):
                    for xi4 in range(xi3+1, BLOCK_SIZE):
                        for xi5 in range(xi4+1, BLOCK_SIZE):
                            cur_neutral_bits_5xor = [xi, xi2, xi3, xi4, xi5]
                            Pr_nt_x_eq1       = div0(AND_plain_x_5xor[NBi_5xor]                           , AND_plain_x_all)                   
                            Pr_nt_x_eq0       = div0(Full_5xor[NBi_5xor] - AND_plain_x_5xor[NBi_5xor]     , cnt_correct - AND_plain_x_all)     
                            Pr_nt_y_eq1       = div0(AND_plain_y_5xor[NBi_5xor]                           , AND_plain_y_all)                   
                            Pr_nt_y_eq0       = div0(Full_5xor[NBi_5xor] - AND_plain_y_5xor[NBi_5xor]     , cnt_correct - AND_plain_y_all)     
                            Pr_nt_xn1_eq1     = div0(AND_plain_xn1_5xor[NBi_5xor]                         , AND_plain_xn1_all)                 
                            Pr_nt_xn1_eq0     = div0(Full_5xor[NBi_5xor] - AND_plain_xn1_5xor[NBi_5xor]   , cnt_correct - AND_plain_xn1_all)   
                            Pr_nt_yn1_eq1     = div0(AND_plain_yn1_5xor[NBi_5xor]                         , AND_plain_yn1_all)                 
                            Pr_nt_yn1_eq0     = div0(Full_5xor[NBi_5xor] - AND_plain_yn1_5xor[NBi_5xor]   , cnt_correct - AND_plain_yn1_all)   
                            Pr_nt_xa_p_y_eq1  = div0(AND_plain_x7_p_y_5xor[NBi_5xor]                      , AND_plain_x7_p_y_all)              
                            Pr_nt_xa_p_y_eq0  = div0(Full_5xor[NBi_5xor] - AND_plain_x7_p_y_5xor[NBi_5xor], cnt_correct - AND_plain_x7_p_y_all)
                            Pr_nt_xa_x_c_eq1  = div0(AND_xa_x_c_5xor[NBi_5xor]                            , AND_xa_x_c_all)                    
                            Pr_nt_xa_x_c_eq0  = div0(Full_5xor[NBi_5xor] - AND_xa_x_c_5xor[NBi_5xor]      , cnt_correct - AND_xa_x_c_all)      
                            Pr_nt_y_x_c_eq1   = div0(AND_y_x_c_5xor[NBi_5xor]                             , AND_y_x_c_all)                     
                            Pr_nt_y_x_c_eq0   = div0(Full_5xor[NBi_5xor] - AND_y_x_c_5xor[NBi_5xor]       , cnt_correct - AND_y_x_c_all)       
                            Pr_nt_c_eq1       = div0(AND_c_5xor[NBi_5xor]                                 , AND_c_all)                         
                            Pr_nt_c_eq0       = div0(Full_5xor[NBi_5xor] - AND_c_5xor[NBi_5xor]           , cnt_correct - AND_c_all)           
                            if Full_5xor[NBi_5xor]/cnt_correct > threshold_freq_low:
                                print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), 
                                "  Pr_x_nt_eq1 "     , div0(AND_plain_x_5xor[NBi_5xor]                                  , Full_5xor[NBi_5xor]), "  Pr_nt_x_eq1 "     , Pr_nt_x_eq1           ,
                                "  Pr_x_nt_eq0 "     , div0(Full_5xor[NBi_5xor] - AND_plain_x_5xor[NBi_5xor]            , Full_5xor[NBi_5xor]), "  Pr_nt_x_eq0 "     , Pr_nt_x_eq0           ,
                                "  Pr_y_nt_eq1 "     , div0(AND_plain_y_5xor[NBi_5xor]                                  , Full_5xor[NBi_5xor]), "  Pr_nt_y_eq1 "     , Pr_nt_y_eq1           ,
                                "  Pr_y_nt_eq0 "     , div0(Full_5xor[NBi_5xor] - AND_plain_y_5xor[NBi_5xor]            , Full_5xor[NBi_5xor]), "  Pr_nt_y_eq0 "     , Pr_nt_y_eq0           ,
                                "  Pr_xn1_nt_eq1 "   , div0(AND_plain_xn1_5xor[NBi_5xor]                                , Full_5xor[NBi_5xor]), "  Pr_nt_xn1_eq1 "   , Pr_nt_xn1_eq1         ,
                                "  Pr_xn1_nt_eq0 "   , div0(Full_5xor[NBi_5xor] - AND_plain_xn1_5xor[NBi_5xor]          , Full_5xor[NBi_5xor]), "  Pr_nt_xn1_eq0 "   , Pr_nt_xn1_eq0         ,
                                "  Pr_yn1_nt_eq1 "   , div0(AND_plain_yn1_5xor[NBi_5xor]                                , Full_5xor[NBi_5xor]), "  Pr_nt_yn1_eq1 "   , Pr_nt_yn1_eq1         ,
                                "  Pr_yn1_nt_eq0 "   , div0(Full_5xor[NBi_5xor] - AND_plain_yn1_5xor[NBi_5xor]          , Full_5xor[NBi_5xor]), "  Pr_nt_yn1_eq0 "   , Pr_nt_yn1_eq0         ,
                                "  Pr_xa_p_y_nt_eq1 ", div0(AND_plain_x7_p_y_5xor[NBi_5xor]                             , Full_5xor[NBi_5xor]), "  Pr_nt_xa_p_y_eq1 ", Pr_nt_xa_p_y_eq1      ,
                                "  Pr_xa_p_y_nt_eq0 ", div0(Full_5xor[NBi_5xor] - AND_plain_x7_p_y_5xor[NBi_5xor]       , Full_5xor[NBi_5xor]), "  Pr_nt_xa_p_y_eq0 ", Pr_nt_xa_p_y_eq0      ,
                                "  Pr_xa_x_c_nt_eq1 ", div0(AND_xa_x_c_5xor[NBi_5xor]                                   , Full_5xor[NBi_5xor]), "  Pr_nt_xa_x_c_eq1 ", Pr_nt_xa_x_c_eq1,
                                "  Pr_xa_x_c_nt_eq0 ", div0(Full_5xor[NBi_5xor] - AND_xa_x_c_5xor[NBi_5xor]             , Full_5xor[NBi_5xor]), "  Pr_nt_xa_x_c_eq0 ", Pr_nt_xa_x_c_eq0,
                                "  Pr_y_x_c_nt_eq1 " , div0(AND_y_x_c_5xor[NBi_5xor]                                    , Full_5xor[NBi_5xor]), "  Pr_nt_y_x_c_eq1 " , Pr_nt_y_x_c_eq1 ,
                                "  Pr_y_x_c_nt_eq0 " , div0(Full_5xor[NBi_5xor] - AND_y_x_c_5xor[NBi_5xor]              , Full_5xor[NBi_5xor]), "  Pr_nt_y_x_c_eq0 " , Pr_nt_y_x_c_eq0 ,
                                "  Pr_c_nt_eq1 "     , div0(AND_c_5xor[NBi_5xor]                                        , Full_5xor[NBi_5xor]), "  Pr_nt_c_eq1 "     , Pr_nt_c_eq1     ,
                                "  Pr_c_nt_eq0 "     , div0(Full_5xor[NBi_5xor] - AND_c_5xor[NBi_5xor]                  , Full_5xor[NBi_5xor]), "  Pr_nt_c_eq0 "     , Pr_nt_c_eq0     ,
                                file=logfile_5xor, flush=True)
                            if Full_5xor[NBi_5xor]/cnt_correct > threshold_freq:
                                print("neutral bit"  , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', '))                           , " neutral freq: "                                      , '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), 
                                "  Pr_x_nt_eq1 "     , div0(AND_plain_x_5xor[NBi_5xor]                                  , Full_5xor[NBi_5xor]), "  Pr_nt_x_eq1 "     , Pr_nt_x_eq1           ,
                                "  Pr_x_nt_eq0 "     , div0(Full_5xor[NBi_5xor] - AND_plain_x_5xor[NBi_5xor]            , Full_5xor[NBi_5xor]), "  Pr_nt_x_eq0 "     , Pr_nt_x_eq0           ,
                                "  Pr_y_nt_eq1 "     , div0(AND_plain_y_5xor[NBi_5xor]                                  , Full_5xor[NBi_5xor]), "  Pr_nt_y_eq1 "     , Pr_nt_y_eq1           ,
                                "  Pr_y_nt_eq0 "     , div0(Full_5xor[NBi_5xor] - AND_plain_y_5xor[NBi_5xor]            , Full_5xor[NBi_5xor]), "  Pr_nt_y_eq0 "     , Pr_nt_y_eq0           ,
                                "  Pr_xn1_nt_eq1 "   , div0(AND_plain_xn1_5xor[NBi_5xor]                                , Full_5xor[NBi_5xor]), "  Pr_nt_xn1_eq1 "   , Pr_nt_xn1_eq1         ,
                                "  Pr_xn1_nt_eq0 "   , div0(Full_5xor[NBi_5xor] - AND_plain_xn1_5xor[NBi_5xor]          , Full_5xor[NBi_5xor]), "  Pr_nt_xn1_eq0 "   , Pr_nt_xn1_eq0         ,
                                "  Pr_yn1_nt_eq1 "   , div0(AND_plain_yn1_5xor[NBi_5xor]                                , Full_5xor[NBi_5xor]), "  Pr_nt_yn1_eq1 "   , Pr_nt_yn1_eq1         ,
                                "  Pr_yn1_nt_eq0 "   , div0(Full_5xor[NBi_5xor] - AND_plain_yn1_5xor[NBi_5xor]          , Full_5xor[NBi_5xor]), "  Pr_nt_yn1_eq0 "   , Pr_nt_yn1_eq0         ,
                                "  Pr_xa_p_y_nt_eq1 ", div0(AND_plain_x7_p_y_5xor[NBi_5xor]                             , Full_5xor[NBi_5xor]), "  Pr_nt_xa_p_y_eq1 ", Pr_nt_xa_p_y_eq1      ,
                                "  Pr_xa_p_y_nt_eq0 ", div0(Full_5xor[NBi_5xor] - AND_plain_x7_p_y_5xor[NBi_5xor]       , Full_5xor[NBi_5xor]), "  Pr_nt_xa_p_y_eq0 ", Pr_nt_xa_p_y_eq0      ,
                                "  Pr_xa_x_c_nt_eq1 ", div0(AND_xa_x_c_5xor[NBi_5xor]                                   , Full_5xor[NBi_5xor]), "  Pr_nt_xa_x_c_eq1 ", Pr_nt_xa_x_c_eq1,
                                "  Pr_xa_x_c_nt_eq0 ", div0(Full_5xor[NBi_5xor] - AND_xa_x_c_5xor[NBi_5xor]             , Full_5xor[NBi_5xor]), "  Pr_nt_xa_x_c_eq0 ", Pr_nt_xa_x_c_eq0,
                                "  Pr_y_x_c_nt_eq1 " , div0(AND_y_x_c_5xor[NBi_5xor]                                    , Full_5xor[NBi_5xor]), "  Pr_nt_y_x_c_eq1 " , Pr_nt_y_x_c_eq1 ,
                                "  Pr_y_x_c_nt_eq0 " , div0(Full_5xor[NBi_5xor] - AND_y_x_c_5xor[NBi_5xor]              , Full_5xor[NBi_5xor]), "  Pr_nt_y_x_c_eq0 " , Pr_nt_y_x_c_eq0 ,
                                "  Pr_c_nt_eq1 "     , div0(AND_c_5xor[NBi_5xor]                                        , Full_5xor[NBi_5xor]), "  Pr_nt_c_eq1 "     , Pr_nt_c_eq1     ,
                                "  Pr_c_nt_eq0 "     , div0(Full_5xor[NBi_5xor] - AND_c_5xor[NBi_5xor]                  , Full_5xor[NBi_5xor]), "  Pr_nt_c_eq0 "     , Pr_nt_c_eq0     ,
                                file=logfile_alls, flush=True)
                            diff_Pr_nt_x_eq1       = Pr_nt_x_eq1       - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_x_eq1       = np.max(Pr_nt_x_eq1     );
                            diff_Pr_nt_x_eq0       = Pr_nt_x_eq0       - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_x_eq0       = np.max(Pr_nt_x_eq0     );
                            diff_Pr_nt_y_eq1       = Pr_nt_y_eq1       - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_y_eq1       = np.max(Pr_nt_y_eq1     );
                            diff_Pr_nt_y_eq0       = Pr_nt_y_eq0       - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_y_eq0       = np.max(Pr_nt_y_eq0     );
                            diff_Pr_nt_xn1_eq1     = Pr_nt_xn1_eq1     - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_xn1_eq1     = np.max(Pr_nt_xn1_eq1   );
                            diff_Pr_nt_xn1_eq0     = Pr_nt_xn1_eq0     - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_xn1_eq0     = np.max(Pr_nt_xn1_eq0   );
                            diff_Pr_nt_yn1_eq1     = Pr_nt_yn1_eq1     - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_yn1_eq1     = np.max(Pr_nt_yn1_eq1   );
                            diff_Pr_nt_yn1_eq0     = Pr_nt_yn1_eq0     - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_yn1_eq0     = np.max(Pr_nt_yn1_eq0   );
                            diff_Pr_nt_xa_p_y_eq1  = Pr_nt_xa_p_y_eq1  - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_xa_p_y_eq1  = np.max(Pr_nt_xa_p_y_eq1);
                            diff_Pr_nt_xa_p_y_eq0  = Pr_nt_xa_p_y_eq0  - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_xa_p_y_eq0  = np.max(Pr_nt_xa_p_y_eq0);
                            diff_Pr_nt_xa_x_c_eq1  = Pr_nt_xa_x_c_eq1  - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_xa_x_c_eq1  = np.max(Pr_nt_xa_x_c_eq1);
                            diff_Pr_nt_xa_x_c_eq0  = Pr_nt_xa_x_c_eq0  - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_xa_x_c_eq0  = np.max(Pr_nt_xa_x_c_eq0);
                            diff_Pr_nt_y_x_c_eq1   = Pr_nt_y_x_c_eq1   - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_y_x_c_eq1   = np.max(Pr_nt_y_x_c_eq1 );
                            diff_Pr_nt_y_x_c_eq0   = Pr_nt_y_x_c_eq0   - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_y_x_c_eq0   = np.max(Pr_nt_y_x_c_eq0 );
                            diff_Pr_nt_c_eq1       = Pr_nt_c_eq1       - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_c_eq1       = np.max(Pr_nt_c_eq1     );
                            diff_Pr_nt_c_eq0       = Pr_nt_c_eq0       - Full_5xor[NBi_5xor]/cnt_correct;  max_Pr_nt_c_eq0       = np.max(Pr_nt_c_eq0     );
                            if len(diff_Pr_nt_x_eq1     [diff_Pr_nt_x_eq1      >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_x_eq1     ), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_x_eq1      " , Pr_nt_x_eq1     , file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_x_eq0     [diff_Pr_nt_x_eq0      >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_x_eq0     ), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_x_eq0      " , Pr_nt_x_eq0     , file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_y_eq1     [diff_Pr_nt_y_eq1      >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_eq1     ), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_y_eq1      " , Pr_nt_y_eq1     , file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_y_eq0     [diff_Pr_nt_y_eq0      >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_eq0     ), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_y_eq0      " , Pr_nt_y_eq0     , file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_xn1_eq1   [diff_Pr_nt_xn1_eq1    >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xn1_eq1   ), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_xn1_eq1    " , Pr_nt_xn1_eq1   , file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_xn1_eq0   [diff_Pr_nt_xn1_eq0    >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xn1_eq0   ), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_xn1_eq0    " , Pr_nt_xn1_eq0   , file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_yn1_eq1   [diff_Pr_nt_yn1_eq1    >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_yn1_eq1   ), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_yn1_eq1    " , Pr_nt_yn1_eq1   , file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_yn1_eq0   [diff_Pr_nt_yn1_eq0    >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_yn1_eq0   ), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_yn1_eq0    " , Pr_nt_yn1_eq0   , file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_xa_p_y_eq1[diff_Pr_nt_xa_p_y_eq1 >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_p_y_eq1), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_xa_p_y_eq1 " , Pr_nt_xa_p_y_eq1, file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_xa_p_y_eq0[diff_Pr_nt_xa_p_y_eq0 >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_p_y_eq0), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_xa_p_y_eq0 " , Pr_nt_xa_p_y_eq0, file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_xa_x_c_eq1[diff_Pr_nt_xa_x_c_eq1 >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_x_c_eq1), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_xa_x_c_eq1 " , Pr_nt_xa_x_c_eq1, file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_xa_x_c_eq0[diff_Pr_nt_xa_x_c_eq0 >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_xa_x_c_eq0), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_xa_x_c_eq0 " , Pr_nt_xa_x_c_eq0, file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_y_x_c_eq1 [diff_Pr_nt_y_x_c_eq1  >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_x_c_eq1 ), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_y_x_c_eq1  " , Pr_nt_y_x_c_eq1 , file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_y_x_c_eq0 [diff_Pr_nt_y_x_c_eq0  >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_y_x_c_eq0 ), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_y_x_c_eq0  " , Pr_nt_y_x_c_eq0 , file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_c_eq1     [diff_Pr_nt_c_eq1      >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_c_eq1     ), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_c_eq1      " , Pr_nt_c_eq1     , file=logfile_cnds, flush=True)
                            if len(diff_Pr_nt_c_eq0     [diff_Pr_nt_c_eq0      >= GPT]) != 0:
                                print("neutral bit" , '{0: <60}'.format(np.array2string(np.array(cur_neutral_bits_5xor), separator=', ')) , " neutral freq: " , '{:0.3f}'.format(max_Pr_nt_c_eq0     ), ' from ', '{:0.3f}'.format(Full_5xor[NBi_5xor]/cnt_correct), " Pr_nt_c_eq0      " , Pr_nt_c_eq0     , file=logfile_cnds, flush=True)
                            NBi_5xor = NBi_5xor + 1
                            del Pr_nt_x_eq1       
                            del Pr_nt_x_eq0       
                            del Pr_nt_y_eq1       
                            del Pr_nt_y_eq0       
                            del Pr_nt_xn1_eq1     
                            del Pr_nt_xn1_eq0     
                            del Pr_nt_yn1_eq1     
                            del Pr_nt_yn1_eq0     
                            del Pr_nt_xa_p_y_eq1  
                            del Pr_nt_xa_p_y_eq0  
                            del Pr_nt_xa_x_c_eq1  
                            del Pr_nt_xa_x_c_eq0  
                            del Pr_nt_y_x_c_eq1   
                            del Pr_nt_y_x_c_eq0   
                            del Pr_nt_c_eq1       
                            del Pr_nt_c_eq0       
                            del diff_Pr_nt_x_eq1     
                            del diff_Pr_nt_x_eq0     
                            del diff_Pr_nt_y_eq1     
                            del diff_Pr_nt_y_eq0     
                            del diff_Pr_nt_xn1_eq1   
                            del diff_Pr_nt_xn1_eq0   
                            del diff_Pr_nt_yn1_eq1   
                            del diff_Pr_nt_yn1_eq0   
                            del diff_Pr_nt_xa_p_y_eq1
                            del diff_Pr_nt_xa_p_y_eq0
                            del diff_Pr_nt_xa_x_c_eq1
                            del diff_Pr_nt_xa_x_c_eq0
                            del diff_Pr_nt_y_x_c_eq1 
                            del diff_Pr_nt_y_x_c_eq0 
                            del diff_Pr_nt_c_eq1     
                            del diff_Pr_nt_c_eq0     
        logfile_5xor.close()
    logfile_allc.close()
    logfile_alls.close()
    logfile_cnds.close()
    del plain_1
    del plain_2
    del key
    del keys_st
    del keys
    gc.collect()

# 测试中性比特集合的整体中性概率
# 将正确的对使用所有独立的中性比特进行扩展形成明文structure，测试structure中所有对都正确对的概率
# 如果这个概率为 pr，那么采样时只需要采样 pr^-1 倍数据就可以期望得到一个全部都是正确对的structure
# 比如：如果整体的中性概率为 0.5，那么多采样2倍明文对，生成2倍的明文结构即可期望得到一个全部都是正确对的structure
def test_neutralbitsetPr(summaryfile, neutral_bits):
    global DIFF
    global end_r
    global start_r
    global rounds
    global tn
    global real_tn
    global input_diff
    global input_diff2
    global output_diff
    global filepath
    global XORN
    global threshold_freq
    global threshold_freq_low
    #
    cnt_correct = 0.0
    pool = mp.Pool(PN)
    idx_range = range(0 * PN, (0 + 1) * PN)
    result = pool.map_async(gen_correctpairs, idx_range).get() # 生成正确明文
    pool.close()
    pool.join()
    plain_1 = np.array([], dtype=versions[VER][5])
    plain_2 = np.array([], dtype=versions[VER][5])
    key     = np.array([], dtype=versions[VER][5])
    if keyschedule == 'free':
        key     = np.reshape(key, (rounds, -1))
    else:
        key     = np.reshape(key, (versions[VER][3], -1))
    for idx in range(PN):
        filename_prefix_cur = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_proc' + str(idx)
        if os.path.exists(filename_prefix_cur + '_CorPl.npy'):
            correct_plain_l_cur = np.load(filename_prefix_cur + '_CorPl.npy')
            plain_1 = np.concatenate((plain_1, correct_plain_l_cur), axis=0)
            os.remove(filename_prefix_cur + '_CorPl.npy')
    for idx in range(PN):
        filename_prefix_cur = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_proc' + str(idx)
        if os.path.exists(filename_prefix_cur + '_CorPr.npy'):
            correct_plain_r_cur = np.load(filename_prefix_cur + '_CorPr.npy')
            plain_2 = np.concatenate((plain_2, correct_plain_r_cur), axis=0)
            os.remove(filename_prefix_cur + '_CorPr.npy')
    for idx in range(PN):
        filename_prefix_cur = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_proc' + str(idx)
        if os.path.exists(filename_prefix_cur + '_CorK.npy'):
            correct_key_cur = np.load(filename_prefix_cur + '_CorK.npy')
            key = np.concatenate((key, correct_key_cur[:]), axis=1)
            os.remove(filename_prefix_cur + '_CorK.npy')
    cnt_correct = len(plain_1)
    if keyschedule == 'free':
        keys = key
    else:
        keys = speck.expand_key_subkey(key, rounds)
    keys_st1 = np.copy(keys)
    keys_st1 = np.reshape(keys_st1, (rounds, cnt_correct, -1))
    keys_st1 = np.repeat(keys_st1, 2, axis=2)
    print("Test the neutral probability using " + str(cnt_correct) + " fresh new correct pairs:", file=summaryfile, flush=True)
    for nbi in neutral_bits:
        plain_01, plain_02, plain_11, plain_12 = make_plain_structure(plain_1, plain_2, input_diff, [nbi])
        t_01, t_02 = speck.one_round_decrypt((plain_01, plain_02), zeo)
        t_11, t_12 = speck.one_round_decrypt((plain_11, plain_12), zeo)
        #
        t_01, t_02 = speck.encrypt((t_01, t_02), keys_st1)
        t_11, t_12 = speck.encrypt((t_11, t_12), keys_st1)
        #
        diff0 = t_01 ^ t_11
        diff1 = t_02 ^ t_12
        d0 = (diff0 == output_diff[0]);
        d1 = (diff1 == output_diff[1]);
        d = d0 * d1
        v = np.sum(d, axis=1)
        s = (v == 2)
        q = np.sum(s)
        print('{0: <60} {1:<6}/{2:>6} = {3:0.3f}'.format(np.array2string(nbi), q, cnt_correct, q/cnt_correct), file=summaryfile, flush=True)
    keys_st = np.copy(keys)
    keys_st = np.reshape(keys_st, (rounds, cnt_correct, -1))
    keys_st = np.repeat(keys_st, 2**(len(neutral_bits)), axis=2)
    plain_01, plain_02, plain_11, plain_12 = make_plain_structure(plain_1, plain_2, input_diff, neutral_bits)
    t_01, t_02 = speck.one_round_decrypt((plain_01, plain_02), zeo)
    t_11, t_12 = speck.one_round_decrypt((plain_11, plain_12), zeo)
    #
    t_01, t_02 = speck.encrypt((t_01, t_02), keys_st)
    t_11, t_12 = speck.encrypt((t_11, t_12), keys_st)
    #
    diff0 = t_01 ^ t_11
    diff1 = t_02 ^ t_12
    d0 = (diff0 == output_diff[0]);
    d1 = (diff1 == output_diff[1]);
    d = d0 * d1
    v = np.sum(d, axis=1)
    s = (v == (2**len(neutral_bits)))
    q = np.sum(s)
    print("Expand correct pairs with all independent neutral bits to form structures, test the probability that all pairs in a structure are correct:", file=summaryfile, flush=True)
    print(': {0:<6}/{1:>6} = {2:0.3f} = 2^-{3:0.3f}'.format(q, cnt_correct, q/cnt_correct, -log2(q/cnt_correct)), file=summaryfile, flush=True)


if __name__ == '__main__':
    global DIFF          # 全局变量，差分路径的名称
    global end_r         # 全局变量，差分路径的结束轮
    global start_r       # 全局变量，差分路径的起始轮
    global rounds        # 全局变量，差分路径的轮数
    global tn            # 每个线程生成的正确对个数
    global real_tn       # 用于搜索中性比特时，实际总共使用的正确对个数
    global input_diff    # 全局变量，输入差分
    global input_diff2   # 全局变量，邻接差分的输入差分
    global output_diff   # 全局变量，输出差分
    global filepath      # 读写文件的路径

    t0 = time();

    input_diff2 = None
    input_diff = None

    # 为了画图而设
    x1h = np.full(WORD_SIZE, 0.60-0.70)
    y1h = np.full(WORD_SIZE, 0.55-0.70)
    x2h = np.full(WORD_SIZE, 0.50-0.70)
    diff_mark = ['_', '*']

    # 一系列变量的初始化
    # 
    AND_plain_x_all         = np.full(WORD_SIZE, 0.0)
    AND_plain_y_all         = np.full(WORD_SIZE, 0.0)
    AND_plain_xn1_all       = np.full(WORD_SIZE, 0.0)
    AND_plain_yn1_all       = np.full(WORD_SIZE, 0.0)
    AND_plain_x7_p_y_all    = np.full(WORD_SIZE, 0.0)
    AND_xa_x_c_all          = np.full(WORD_SIZE, 0.0)
    AND_y_x_c_all           = np.full(WORD_SIZE, 0.0)
    AND_c_all               = np.full(WORD_SIZE, 0.0)
    #
    Full_1xor                = np.full(BLOCK_SIZE, 0.0)
    AND_plain_x_1xor         = np.full((BLOCK_SIZE, WORD_SIZE), 0.0)
    AND_plain_y_1xor         = np.full((BLOCK_SIZE, WORD_SIZE), 0.0)
    AND_plain_xn1_1xor       = np.full((BLOCK_SIZE, WORD_SIZE), 0.0)
    AND_plain_yn1_1xor       = np.full((BLOCK_SIZE, WORD_SIZE), 0.0)
    AND_plain_x7_p_y_1xor    = np.full((BLOCK_SIZE, WORD_SIZE), 0.0)
    AND_xa_x_c_1xor          = np.full((BLOCK_SIZE, WORD_SIZE), 0.0)
    AND_y_x_c_1xor           = np.full((BLOCK_SIZE, WORD_SIZE), 0.0)
    AND_c_1xor               = np.full((BLOCK_SIZE, WORD_SIZE), 0.0)
    
    if XORN >= 2:
        Full_2xor                = np.full( XORcN[1], 0.0)
        AND_plain_x_2xor         = np.full((XORcN[1], WORD_SIZE), 0.0)
        AND_plain_y_2xor         = np.full((XORcN[1], WORD_SIZE), 0.0)
        AND_plain_xn1_2xor       = np.full((XORcN[1], WORD_SIZE), 0.0)
        AND_plain_yn1_2xor       = np.full((XORcN[1], WORD_SIZE), 0.0)
        AND_plain_x7_p_y_2xor    = np.full((XORcN[1], WORD_SIZE), 0.0)
        AND_xa_x_c_2xor          = np.full((XORcN[1], WORD_SIZE), 0.0)
        AND_y_x_c_2xor           = np.full((XORcN[1], WORD_SIZE), 0.0)
        AND_c_2xor               = np.full((XORcN[1], WORD_SIZE), 0.0)
    if XORN >= 3:
        Full_3xor                = np.full( XORcN[2], 0.0)
        AND_plain_x_3xor         = np.full((XORcN[2], WORD_SIZE), 0.0)
        AND_plain_y_3xor         = np.full((XORcN[2], WORD_SIZE), 0.0)
        AND_plain_xn1_3xor       = np.full((XORcN[2], WORD_SIZE), 0.0)
        AND_plain_yn1_3xor       = np.full((XORcN[2], WORD_SIZE), 0.0)
        AND_plain_x7_p_y_3xor    = np.full((XORcN[2], WORD_SIZE), 0.0)
        AND_xa_x_c_3xor          = np.full((XORcN[2], WORD_SIZE), 0.0)
        AND_y_x_c_3xor           = np.full((XORcN[2], WORD_SIZE), 0.0)
        AND_c_3xor               = np.full((XORcN[2], WORD_SIZE), 0.0)
    if XORN >= 4:
        Full_4xor                = np.full( XORcN[3], 0.0)
        AND_plain_x_4xor         = np.full((XORcN[3], WORD_SIZE), 0.0)
        AND_plain_y_4xor         = np.full((XORcN[3], WORD_SIZE), 0.0)
        AND_plain_xn1_4xor       = np.full((XORcN[3], WORD_SIZE), 0.0)
        AND_plain_yn1_4xor       = np.full((XORcN[3], WORD_SIZE), 0.0)
        AND_plain_x7_p_y_4xor    = np.full((XORcN[3], WORD_SIZE), 0.0)
        AND_xa_x_c_4xor          = np.full((XORcN[3], WORD_SIZE), 0.0)
        AND_y_x_c_4xor           = np.full((XORcN[3], WORD_SIZE), 0.0)
        AND_c_4xor               = np.full((XORcN[3], WORD_SIZE), 0.0)
    if XORN >= 5:
        Full_5xor                = np.full( XORcN[4], 0.0)
        AND_plain_x_5xor         = np.full((XORcN[4], WORD_SIZE), 0.0)
        AND_plain_y_5xor         = np.full((XORcN[4], WORD_SIZE), 0.0)
        AND_plain_xn1_5xor       = np.full((XORcN[4], WORD_SIZE), 0.0)
        AND_plain_yn1_5xor       = np.full((XORcN[4], WORD_SIZE), 0.0)
        AND_plain_x7_p_y_5xor    = np.full((XORcN[4], WORD_SIZE), 0.0)
        AND_xa_x_c_5xor          = np.full((XORcN[4], WORD_SIZE), 0.0)
        AND_y_x_c_5xor           = np.full((XORcN[4], WORD_SIZE), 0.0)
        AND_c_5xor               = np.full((XORcN[4], WORD_SIZE), 0.0)
    #
    # 选择要进行搜索的差分路径
    # 如有新的差分路径需要在config.py的DIFFs里加路径名称，get_trails(DIFF)里添加具体的路径（从0开始对应路径从下往上扩展）
    for diffi in DIFF_idx: 
        t0 = time();
        DIFF = DIFFs[diffi]
        # 对应每种不同版本Speck的差分路径，差分路径表示从最后一轮到第一轮
        trail_base = get_trails(DIFF)
        # Speck32/64 的一些邻接差分
        #input_diff2 =  (0x7468, 0xB0F8)
        #input_diff2 =  (0x8020, 0x4101)
        #input_diff2 =  (0x8021, 0x4101)
        #input_diff2 =  (0x0A20, 0x4205)
        trail = [None for i in range(len(trail_base))]
        dir_top = "./" + str(BLOCK_SIZE) + "/"
        # 此处可以控制差分路径的轮数
        for rounds in range(1 + START_NRound, 1 + END_NRound - 1, -1):
            t0_nrounds = time();
            notsearched = False
            for start_r in range(START_FirstRoundIdx, END_FirstRoundIdx+1):
                end_r = start_r + (rounds - 1)
                #rounds = 1 + end_r - start_r
                sum_all_filename = dir_top + 'TrailRol' + '_' + str(2 * versions[VER][0]) + "_" + str(rounds-1) + "r_[" + str(end_r) + "_" + str(start_r) + "]_1-" + str(XORN) + 'xors_tredFq_' + str(threshold_freq) + "_rank.csv"
                sum_filename = dir_top + DIFF + '_' + 'TrailRol' + '_' + str(2 * versions[VER][0]) + "_" + str(rounds-1) + "r_[" + str(end_r) + "_" + str(start_r) + "]_1-" + str(XORN) + 'xors_tredFq_' + str(threshold_freq) + ".csv"
                # Rotation Pass_N  CdPass_N
                Rotation = np.array(range(WORD_SIZE))
                TWeight = np.zeros(WORD_SIZE, dtype=int)
                Pass_N = np.zeros(WORD_SIZE, dtype=int)
                CdPass_N = np.zeros(WORD_SIZE, dtype=int)
                # 根据trail_base，可以通过比特位循环平移来扩展新的差分路径
                for trail_rol in ROT_LIST:
                    result_PNB = []
                    t0_trail_rol = time();
                    for ri in range(len(trail_base)):
                        trail[ri] = (rol(trail_base[ri][0], trail_rol), rol(trail_base[ri][1], trail_rol))
                    # 判断新的差分路径是否是合法的
                    invalid_trail = False
                    for ri in range(end_r, start_r, -1):
                        dx0 = ror(trail[ri][0], ALPHA)
                        dy0 = trail[ri][1]
                        dz0 = trail[ri - 1][0]
                        dxyz0 = (dx0 ^ dy0 ^ dz0) & one # 合法差分路径需要满足的条件
                        if dxyz0 != zeo:
                            invalid_trail = True
                            break;
                    if invalid_trail:
                        continue
                    tn = ((1<<10) + PN - 1) // PN
                    #if input_diff == None:
                    input_diff = trail[end_r] # 输入差分为差分路径的最后一个
                    input_diff_next = trail[end_r-1]
                    output_diff = trail[start_r] # 输出差分为差分路径的第一个
                    # 
                    input_diff_mark_xa = []
                    input_diff_mark_y_ = []
                    input_diff_next_mark_xa = []
                    for ri in range(end_r, start_r, -1):
                        dx0 = ror(trail[ri][0], ALPHA)
                        dy0 = trail[ri][1]
                        dz0 = trail[ri - 1][0]
                        input_diff_mark_xa.append([diff_mark[GET1(dx0, mi)] for mi in range(WORD_SIZE - 1, -1, -1)])
                        input_diff_mark_y_.append([diff_mark[GET1(dy0, mi)] for mi in range(WORD_SIZE - 1, -1, -1)])
                        input_diff_next_mark_xa.append([diff_mark[GET1(dz0, mi)] for mi in range(WORD_SIZE - 1, -1, -1)])

                    #
                    if input_diff2 == None:
                        #filepath = dir_top + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_1-" + str(XORN) + "xors/"
                        filepath = dir_top + DIFF + '_' + 'TrailRol' + "{0:02}".format(trail_rol) + '_' + str(2 * versions[VER][0]) + "_" + str(rounds-1) + "r_[" + str(end_r) + "_" + str(start_r) + "]_1-" + str(XORN) + "xors/"
                    else:
                        filepath = dir_top + "2diff_" + DIFF + '_' + 'TrailRol' + "{0:02}".format(trail_rol) + '_' + "{0:0{width}x}".format(input_diff2[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff2[1], width=width4) + '_' + str(2 * versions[VER][0]) + "_" + str(rounds-1) + "r_[" + str(end_r) + "_" + str(start_r) + "]_1-" + str(XORN) + "xors/"
                    #if not os.path.exists(filepath):
                    #    #os.makedirs(filepath)
                    #    notsearched = True
                    #    break;
                    if True:
                        filepath = dir_top + DIFF + '_' + 'TrailRol' + "{0:02}".format(trail_rol) + '_' + str(2 * versions[VER][0]) + "_" + str(rounds-1) + "r_[" + str(end_r) + "_" + str(start_r) + "]_1-" + str(XORN) + "xors/"
                        if not os.path.exists(filepath):
                            os.makedirs(filepath)
                        print_trails(trail_rol)
                    if True:
                        ''''''
                        analysis_trails(TWeight, trail_rol)
                        #break;
                        real_tn = 1 << 10
                        filename_prefix_all = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_proc'
                        proc_allc_fn = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_allc_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc.txt'
                        proc_alls_fn = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_alls_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc.txt'
                        proc_cnd_fn = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_cnds_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc.txt'
                        summary_fn = filepath + "summary_" + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '.txt'
                        if True:
                            pool = mp.Pool(PN)
                            idx_range = range(0 * PN, (0 + 1) * PN)
                            result = pool.map_async(gen_correctpairs, idx_range).get() # 生成正确明文
                            pool.close()
                            pool.join()
                            correct_plain_l = np.array([], dtype=versions[VER][5])
                            correct_plain_r = np.array([], dtype=versions[VER][5])
                            correct_key     = np.array([], dtype=versions[VER][5])
                            if keyschedule == 'free':
                                correct_key     = np.reshape(correct_key, (rounds, -1))
                            else:
                                correct_key     = np.reshape(correct_key, (versions[VER][3], -1))
                            for idx in range(PN):
                                filename_prefix_cur = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_proc' + str(idx)
                                if os.path.exists(filename_prefix_cur + '_CorPl.npy'):
                                    correct_plain_l_cur = np.load(filename_prefix_cur + '_CorPl.npy')
                                    correct_plain_l = np.concatenate((correct_plain_l, correct_plain_l_cur), axis=0)
                                    os.remove(filename_prefix_cur + '_CorPl.npy')
                            np.save(filename_prefix_all + '_CorPl.npy', correct_plain_l)
                            for idx in range(PN):
                                filename_prefix_cur = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_proc' + str(idx)
                                if os.path.exists(filename_prefix_cur + '_CorPr.npy'):
                                    correct_plain_r_cur = np.load(filename_prefix_cur + '_CorPr.npy')
                                    correct_plain_r = np.concatenate((correct_plain_r, correct_plain_r_cur), axis=0)
                                    os.remove(filename_prefix_cur + '_CorPr.npy')
                            np.save(filename_prefix_all + '_CorPr.npy',correct_plain_r)
                            for idx in range(PN):
                                filename_prefix_cur = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_proc' + str(idx)
                                if os.path.exists(filename_prefix_cur + '_CorK.npy'):
                                    correct_key_cur = np.load(filename_prefix_cur + '_CorK.npy')
                                    correct_key = np.concatenate((correct_key, correct_key_cur[:]), axis=1)
                                    os.remove(filename_prefix_cur + '_CorK.npy')
                            np.save(filename_prefix_all + '_CorK.npy', correct_key)
                            #with open(filename_prefix_all + '.txt', 'w') as outfile:
                            #    for idx in range(PN):
                            #        filename_prefix_cur = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_proc' + str(idx)
                            #        with open(filename_prefix_cur + '.txt', 'r+') as infile:
                            #            outfile.write(infile.read())
                            #        os.remove(filename_prefix_cur + '.txt')
                            correct_plain_l = np.load(filename_prefix_all + '_CorPl.npy')
                            correct_plain_r = np.load(filename_prefix_all + '_CorPr.npy')
                            correct_key = np.load(filename_prefix_all + '_CorK.npy')
                            if keyschedule == 'free':
                                keys = correct_key
                            else:
                                keys = speck.expand_key_subkey(correct_key, rounds)
                            correct_plain_l_p = correct_plain_l ^ input_diff[0]
                            correct_plain_r_p = correct_plain_r ^ input_diff[1]
                            #
                            plain_01, plain_02 = speck.one_round_decrypt((correct_plain_l, correct_plain_r), zeo)
                            plain_11, plain_12 = speck.one_round_decrypt((correct_plain_l_p, correct_plain_r_p), zeo)
                            #
                            t_01, t_02 = speck.encrypt((plain_01, plain_02), keys)
                            t_11, t_12 = speck.encrypt((plain_11, plain_12), keys)
                            diff0 = t_01 ^ t_11
                            diff1 = t_02 ^ t_12
                            for ci in range(len(diff0)):
                                if diff0[ci] != output_diff[0] or (diff1[ci] != output_diff[1]):
                                    print("Something wrong with the correct pair ", ci)
                            ''''''
                            ''''''
                            real_tn = len(correct_plain_l)
                            del correct_plain_l
                            del correct_plain_r
                            del correct_key
                            del correct_plain_l_p
                            del correct_plain_r_p
                            del plain_01
                            del plain_02
                            del t_01
                            del t_02
                            del t_11
                            del t_12
                            del diff0
                            del diff1
                            gc.collect()
                            for it in range((2*versions[VER][0]) // PN): # 线程开始
                                pool = mp.Pool(PN)
                                idx_range = range(it * PN, (it + 1) * PN)
                                result = pool.map_async(searching_neutralbits, idx_range).get() # 多线程
                                pool.close()
                                pool.join()
                            # 将结果写入文件
                            with open(proc_allc_fn, 'w') as outfile:
                                for idx in range(BLOCK_SIZE):
                                    proc_cur_fn = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_allc_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc' + str(idx) + '.txt'
                                    with open(proc_cur_fn, 'r+') as infile:
                                        outfile.write(infile.read())
                                    outfile.write("\n")
                                    os.remove(proc_cur_fn)
                            with open(proc_alls_fn, 'w') as outfile:
                                for idx in range(BLOCK_SIZE):
                                    proc_cur_fn = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_alls_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc' + str(idx) + '.txt'
                                    with open(proc_cur_fn, 'r+') as infile:
                                        outfile.write(infile.read())
                                    outfile.write("\n")
                                    os.remove(proc_cur_fn)
                            with open(proc_cnd_fn, 'w') as outfile:
                                for idx in range(BLOCK_SIZE):
                                    proc_cur_fn = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_cnds_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc' + str(idx) + '.txt'
                                    with open(proc_cur_fn, 'r+') as infile:
                                        outfile.write(infile.read())
                                    outfile.write("\n")
                                    os.remove(proc_cur_fn)
                        all_matrix = np.zeros((1, WORD_SIZE * 2), dtype=int)
                        with open(summary_fn, 'a+') as summaryfile:
                            summaryfile.write("==== Neutral bits with Pr >= " + str(threshold_freq) + "\n")
                            with open(proc_alls_fn, 'r') as sumfile:
                                for line in sumfile:
                                    tmp = line.split()
                                    if len(tmp) != 0:
                                        temp = line
                                        freqstr = line[89:89+5]
                                        if float(freqstr) >= threshold_freq:
                                            summaryfile.write(line[0:89+5])
                                            summaryfile.write("\n")
                                            all_row = np.zeros(WORD_SIZE * 2, dtype=int)
                                            NBS_str = line[12:74]
                                            for char in '[,]':
                                                NBS_str = NBS_str.replace(char, ' ')
                                            NBS_str = NBS_str.split()
                                            for nbs in NBS_str:
                                                nb = int(nbs)
                                                all_row[nb] = 1
                                            all_matrix = np.vstack((all_matrix, all_row))
                            Pass_Ind = matrix(GF(2), all_matrix)
                            B = Pass_Ind.row_space().basis()
                            BM = np.array(B)
                            for br in BM:
                                cur_NB = np.where(br==1)[0]
                                result_PNB.append(cur_NB)
                                print(cur_NB, file=summaryfile)
                            Pass_N[trail_rol] = len(BM)
                            summaryfile.write("Number of independent neutral bits with Pr > " + str(threshold_freq) + ": " + str(Pass_N[trail_rol]) + "\n")
                            test_neutralbitsetPr(summaryfile, result_PNB)
                            summaryfile.write("==== (Conditional) Neutral bits with Pr > " + str(threshold_freq) + "\n")
                            with open(proc_cnd_fn, 'r') as sumfile:
                                for line in sumfile:
                                    tmp = line.split()
                                    if len(tmp) != 0:
                                        temp = line
                                        freqstr = line[89:89+5]
                                        if float(freqstr) > threshold_freq:
                                            summaryfile.write(line)
                                            all_row = np.zeros(WORD_SIZE * 2, dtype=int)
                                            NBS_str = line[12:74]
                                            for char in '[,]':
                                                NBS_str = NBS_str.replace(char, ' ')
                                            NBS_str = NBS_str.split()
                                            for nbs in NBS_str:
                                                nb = int(nbs)
                                                all_row[nb] = 1
                                            all_matrix = np.vstack((all_matrix, all_row))
                            Pass_Ind = matrix(GF(2), all_matrix)
                            B = Pass_Ind.row_space().basis()
                            BM = np.array(B)
                            for br in BM:
                                cur_NB = np.where(br==1)[0]
                                print(cur_NB, file=summaryfile)
                            CdPass_N[trail_rol] = len(BM)
                            summaryfile.write("Number of independent neutral bits with Pr > " + str(threshold_freq) + ": " + str(CdPass_N[trail_rol]) + "\n")
                    if True:
                        fig, ax = plt.subplots(figsize=(WORD_SIZE*1.1,10), dpi= 600)
                        plots_pdfs = []
                        handles = []
                        for xorn in range(1, XORN+1):
                            proc_fn = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_' + str(xorn) + 'xor_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc.txt'
                            with open(proc_fn, 'w') as outfile:
                                for idx in range(BLOCK_SIZE):
                                    proc_cur_fn = filepath + str(rounds-1) +"r_" + "{0:0{width}x}".format(input_diff[0], width=width4) + "_" + "{0:0{width}x}".format(input_diff[1], width=width4) + "_" + "{0:0{width}x}".format(output_diff[0], width=width4) + "_" + "{0:0{width}x}".format(output_diff[1], width=width4) + '_' + str(xorn) + 'xor_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '_proc' + str(idx) + '.txt'
                                    with open(proc_cur_fn, 'r+') as infile:
                                        outfile.write(infile.read())
                                    outfile.write("\n")
                                    os.remove(proc_cur_fn)
                            freq_fn = proc_fn[:-4] + '_freq.npy'
                            nbs_fn = proc_fn[:-4] + '_nbs.npy'
                            logfile = open(proc_fn, 'r')
                            freq = np.full(XORcN[xorn-1], 0.0)
                            nb_array = np.full((XORcN[xorn-1], xorn), -1, dtype=int)
                            cnt = 0
                            for line in logfile:
                                tmp = line.split()
                                if len(tmp) != 0:
                                    temp = line
                                    for char in '[,]':
                                        temp = temp.replace(char, ' ')
                                    temp = temp.split()
                                    if temp[0] == 'neutral' and int(temp[2]) != -1:
                                        for nbi in range(xorn):
                                            nb_array[cnt][nbi] = int(temp[2 + nbi])
                                        freq[cnt] = float(temp[2 + xorn + 2])
                                        cnt += 1
                            np.save(freq_fn, freq)
                            np.save(nbs_fn, nb_array)
                            x = np.arange(WORD_SIZE)
                            if xorn == 1:
                                nb_array_str = np.array([str(nbs) for nbs in nb_array])
                                dictionary = dict(zip(nb_array_str, freq))
                                sub_nb_str1 = [str(np.array([nbi])) for nbi in range(WORD_SIZE - 1, -1, -1)]
                                sub_nb_str2 = [str(np.array([((nbi + ALPHA) % WORD_SIZE) + WORD_SIZE])) for nbi in range(WORD_SIZE - 1, -1, -1)]
                                #labels2 = [sub_nb_str1[nbi] + ' ' + sub_nb_str2[nbi] for nbi in range(WORD_SIZE)]
                                freq1 = np.full(WORD_SIZE, 0.0)
                                freq2 = np.full(WORD_SIZE, 0.0)
                                for i in x:
                                    if sub_nb_str1[i] in dictionary:
                                        freq1[i] = dictionary[sub_nb_str1[i]]
                                    if sub_nb_str2[i] in dictionary:
                                        freq2[i] = dictionary[sub_nb_str2[i]]
                                width = 0.5
                                plot_1 = ax.bar(x - width/2, freq1, width, color= 'green', label='$y_i$', zorder=XORN + 1 - xorn)
                                plot_2 = ax.bar(x + width/2, freq2, width, color='orange', label='$x_{i + ' + str(ALPHA) + '}$', zorder=XORN + 1 - xorn)
                                #ax.set_xticklabels(labels2, rotation=40, ha='center')
                                #ax.legend(loc=2, prop=font)
                                addlabels(x - width/2, freq1, XORN + 1 - xorn)
                                addlabels(x + width/2, freq2, XORN + 1 - xorn)
                                #for cri in range(rounds - 1):
                                #    addmarks(x, x1h + (rounds - 1 - cri) * 0.2, input_diff_mark_xa[cri])
                                #    addmarks(x, y1h + (rounds - 1 - cri) * 0.2, input_diff_mark_y_[cri])
                                #    addmarks(x, x2h + (rounds - 1 - cri) * 0.2, input_diff_next_mark_xa[cri])
                            if xorn == 2:
                                nb_array_str = np.array([str(nbs) for nbs in nb_array])
                                dictionary = dict(zip(nb_array_str, freq))
                                sub_nb_str = [str(np.array([xi, ((xi + ALPHA) % WORD_SIZE) + WORD_SIZE])) for xi in range(WORD_SIZE - 1, -1, -1)]
                                sub_freq = np.full(WORD_SIZE, 0.0)
                                for i in x:
                                    if sub_nb_str[i] in dictionary:
                                        sub_freq[i] = dictionary[sub_nb_str[i]]
                                    else:
                                        sub_freq[i] = 0.0
                                #labels = sub_nb_str
                                width = 0.5
                                plot_1 = ax.bar(x, sub_freq, width, label='$[y_i, x_{i + ' + str(ALPHA) + '}]$', color= 'pink', zorder=XORN + 1 - xorn)
                                #ax.set_xticklabels(labels, rotation=40, ha='center')
                                addlabels(x, sub_freq, XORN + 1 - xorn)
                                #for cri in range(rounds - 1):
                                #    addmarks(x, x1h + (rounds - 1 - cri) * 0.2, input_diff_mark_xa[cri])
                                #    addmarks(x, y1h + (rounds - 1 - cri) * 0.2, input_diff_mark_y_[cri])
                                #    addmarks(x, x2h + (rounds - 1 - cri) * 0.2, input_diff_next_mark_xa[cri])
                                if VER == 0 and input_diff == (0x8020, 0x4101) and CONDITIONAL == 1:
                                    plot_1 = ax.bar([x[WORD_SIZE - 1 - 5]], [sub_freq[WORD_SIZE - 1 - 5]], width, linewidth=3, edgecolor = "#FCF75E", color= 'pink', zorder=XORN)
                                    handles.append(plot_1)
                                    addmark([x[WORD_SIZE - 1 - 5]], [x1h[0] + (rounds - 1) * 0.2], 'v', '#FCF75E')
                                    addmark([x[WORD_SIZE - 1 - 5]], [y1h[0] + (rounds - 1) * 0.2], '^', '#FCF75E')
                            if xorn == 3:
                                sub_nb_str = [str(np.array([xi, min(((xi + ALPHA) % WORD_SIZE) + WORD_SIZE, ((xi + ALPHA +  BETA) % WORD_SIZE) + WORD_SIZE), max(((xi + ALPHA) % WORD_SIZE) + WORD_SIZE, ((xi + ALPHA +  BETA) % WORD_SIZE) + WORD_SIZE)])) for xi in range(WORD_SIZE - 1, -1, -1)]
                                sub_freq = np.full(WORD_SIZE, 0.0)
                                nb_array_str = np.array([str(nbs) for nbs in nb_array])
                                dictionary = dict(zip(nb_array_str, freq))
                                for i in x:
                                    if sub_nb_str[i] in dictionary:
                                        sub_freq[i] = dictionary[sub_nb_str[i]]
                                    else:
                                        sub_freq[i] = 0.0
                                #labels = sub_nb_str
                                width = 0.5
                                plot_1 = ax.bar(x, sub_freq, width, label=r'$[y_i, x_{i + ' + str(ALPHA) + r'}, x_{i + ' + str(ALPHA) + ' + ' + str(BETA) + r'}]$', color= '#4166F5', zorder=XORN + 1 - xorn)
                                addlabels(x, sub_freq, XORN + 1 - xorn)
                                #
                                #for cri in range(rounds - 1):
                                #    addmarks(x, x1h + (rounds - 1 - cri) * 0.2, input_diff_mark_xa[cri])
                                #    addmarks(x, y1h + (rounds - 1 - cri) * 0.2, input_diff_mark_y_[cri])
                                #    addmarks(x, x2h + (rounds - 1 - cri) * 0.2, input_diff_next_mark_xa[cri])
                                if VER == 0 and input_diff == (0x8020, 0x4101) and CONDITIONAL == 1:
                                    plot_1 = ax.bar([x[WORD_SIZE - 1 - 15]], [sub_freq[WORD_SIZE - 1 - 15]], width, linewidth=3, edgecolor = "#7CFC00", color= '#4166F5', zorder=XORN)
                                    addmark([x[WORD_SIZE - 1 - 1]], [y1h[0] + (rounds - 1) * 0.2], '$0$', '#7CFC00')
                                    handles.append(plot_1)
                                    plot_1 = ax.bar([x[WORD_SIZE - 1 - 4]], [sub_freq[WORD_SIZE - 1 - 4]], width, linewidth=3, edgecolor = "#00FFFF", color= '#4166F5', zorder=XORN)
                                    addmark([x[WORD_SIZE - 1 - 4]], [x1h[0] + (rounds - 1) * 0.2], 'v', '#00FFFF')
                                    addmark([x[WORD_SIZE - 1 - 4]], [y1h[0] + (rounds - 1) * 0.2], '^', '#00FFFF')
                                    handles.append(plot_1)
                                #
                                sub_nb_str = [str(np.array([min(xi, (xi + 1 + WORD_SIZE - ALPHA - BETA) % WORD_SIZE), max(xi, (xi + 1 + WORD_SIZE - ALPHA - BETA) % WORD_SIZE), ((xi + ALPHA) % WORD_SIZE) + WORD_SIZE])) for xi in range(WORD_SIZE - 1, -1, -1)]
                                sub_freq = np.full(WORD_SIZE, 0.0)
                                nb_array_str = np.array([str(nbs) for nbs in nb_array])
                                dictionary = dict(zip(nb_array_str, freq))
                                for i in x:
                                    if sub_nb_str[i] in dictionary:
                                        sub_freq[i] = dictionary[sub_nb_str[i]]
                                    else:
                                        sub_freq[i] = 0.0
                                #labels = sub_nb_str
                                width = 0.5
                                plot_1 = ax.bar(x, sub_freq, width, label=r'$[y_i, x_{i + ' + str(ALPHA) + r'}, y_{i + 1 - ' + str(ALPHA) + ' - ' + str(BETA) + r'}]$', color= '#9966CC', zorder=XORN + 1 - xorn)
                                addlabels(x, sub_freq, XORN + 1 - xorn)
                            if xorn == XORN:
                                #for b in handles:
                                #    b[0].set_linewidth(3)
                                ax.set_ylabel('Neutral Frequency')
                                ax.set_xlabel(r'Bit index $i$')
                                #ax.set_title('Neutral frequency for size ' + str(XORN) + ' NBS')
                                ax.set_title(r'Neutral probability of candidate bit(sets) for a $' + str(rounds - 1) + '$-round differential trail')
                                ax.set_xticks(x)
                                ax.set_xlim(x[0]-1, x[-1]+1)
                                ax.set_ylim(0.0, 1.15)
                                labels = [str(nbi) for nbi in range(WORD_SIZE - 1, -1, -1)]
                                ax.set_xticklabels(labels, ha='center')
                                ax.legend(loc=1, prop=font)
                                ax.tick_params(which="both", bottom=True, labelsize=FntSize2)
                                for cri in range(rounds - 1):
                                    #xrol = [(xi + cri * ALPHA) % WORD_SIZE for xi in x]
                                    addmarks(x, x1h + (rounds - 1 - cri) * 0.2, input_diff_mark_xa[cri])
                                    addmarks(x, y1h + (rounds - 1 - cri) * 0.2, input_diff_mark_y_[cri])
                                    addmarks(x, x2h + (rounds - 1 - cri) * 0.2, input_diff_next_mark_xa[cri])
                                ax.yaxis.set_major_locator(MultipleLocator(0.2))
                                ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                                ax.yaxis.grid(True, which='minor', color='lightgrey', alpha=0.4)
                                ax.yaxis.grid(True, which='major', color='lightgrey', alpha=0.6)
                                plot_combined_pdf = filepath[2:-1] + '_test' + str(real_tn) + '_tredFq_' + str(threshold_freq) + '.pdf'
                                plt.savefig(plot_combined_pdf)
                                plt.clf()
                                plt.close()
                            del freq
                            del nb_array
                            gc.collect()
                            logfile.close()
                        ''''''
                        gc.collect()
                    t1_trail_rol = time();
                    print(DIFF + "_Trail_Rot", trail_rol, " wall time (in hours): ", (t1_trail_rol - t0_trail_rol)/3600.0);
                    print(DIFF + "_Trail_Rot", trail_rol, " wall time (in mins): ", (t1_trail_rol - t0_trail_rol)/60.0, "\n");
                if notsearched:
                    break;
                trailstr = DIFF + '_' + str(rounds-1) + 'r_'
                d = {trailstr+'TWeight': TWeight, trailstr+'Pass_N': Pass_N, trailstr+'CdPass_N': CdPass_N} #'Rotation': Rotation, 
                df = pd.DataFrame(data=d)
                df.to_csv(sum_filename, mode='a')
                df.to_csv(sum_all_filename, mode='a')
            t1_nrounds = time();
            print(DIFF, rounds - 1, " rounds", " wall time (in hours): ", (t1_nrounds - t0_nrounds)/3600.0);
            print(DIFF, rounds - 1, " rounds", " wall time (in mins): ", (t1_nrounds - t0_nrounds)/60.0, "\n");
        t1 = time();
        print(DIFF, " wall time (in hours): ", (t1 - t0)/3600.0);
        print(DIFF, " wall time (in mins): ", (t1 - t0)/60.0, "\n");
