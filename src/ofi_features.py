import pandas as pd
import numpy as np
import math

def compute_best_level_ofi(df):
    df = df.sort_values('ts_recv').copy()
    d_bid_sz = df['bid_sz_00'].diff()
    d_ask_sz = df['ask_sz_00'].diff()
    d_bid_px = df['bid_px_00'].diff()
    d_ask_px = df['ask_px_00'].diff()
    bid_contrib = np.where(d_bid_px >= 0, d_bid_sz, 0)
    ask_contrib = np.where(d_ask_px <= 0, d_ask_sz, 0)
    df['ofi_best'] = bid_contrib - ask_contrib
    return df[['ts_recv', 'ofi_best']]

def compute_multi_level_ofi(df, depth_levels=10):
    df = df.sort_values('ts_recv').copy()
    total_ofi = np.zeros(len(df))
    for i in range(depth_levels):
        bid_px = df[f'bid_px_0{i}'].diff()
        ask_px = df[f'ask_px_0{i}'].diff()
        bid_sz = df[f'bid_sz_0{i}'].diff()
        ask_sz = df[f'ask_sz_0{i}'].diff()
        bid_contrib = np.where(bid_px >= 0, bid_sz, 0)
        ask_contrib = np.where(ask_px <= 0, ask_sz, 0)
        total_ofi += (bid_contrib - ask_contrib)
    df['ofi_multi'] = total_ofi
    return df[['ts_recv', 'ofi_multi']]

def compute_integrated_ofi(df, depth_levels=10, alpha=0.2):
    df = df.sort_values('ts_recv').copy()
    integrated_ofi = np.zeros(len(df))
    for i in range(depth_levels):
        weight = math.exp(-alpha * i)
        bid_px = df[f'bid_px_0{i}'].diff()
        ask_px = df[f'ask_px_0{i}'].diff()
        bid_sz = df[f'bid_sz_0{i}'].diff()
        ask_sz = df[f'ask_sz_0{i}'].diff()
        bid_contrib = np.where(bid_px >= 0, bid_sz, 0)
        ask_contrib = np.where(ask_px <= 0, ask_sz, 0)
        integrated_ofi += weight * (bid_contrib - ask_contrib)
    df['ofi_integrated'] = integrated_ofi
    return df[['ts_recv', 'ofi_integrated']]