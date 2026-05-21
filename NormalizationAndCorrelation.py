#互相關（Cross-Correlation）
#用於衡量兩個訊號在不同時間延遲下的相似程度。
#一般 PC 端、伺服器端應用、有大量高頻白雜訊（White Noise）的環境，互相關可以用 FFT（快速傅立葉變換） 進行全域加速。
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wav
from scipy.signal import correlate

def load_normalize_and_align_files():
    # ==========================================
    # 1. 載入與 Peak Normalization
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    MIC1_FILENAME = os.path.join(BASE_DIR, "mic1.wav")
    MIC2_FILENAME = os.path.join(BASE_DIR, "mic2.wav")

    try:
        rate1, data1 = wav.read(MIC1_FILENAME)
        rate2, data2 = wav.read(MIC2_FILENAME)
        RATE = rate1
        
        raw_mic1 = np.clip(data1, -32768, 32767).astype(np.float32)
        raw_mic2 = np.clip(data2, -32768, 32767).astype(np.float32)
        
        min_len = min(len(raw_mic1), len(raw_mic2))
        raw_mic1 = raw_mic1[:min_len]
        raw_mic2 = raw_mic2[:min_len]
    except Exception as e:
        print(f"❌ 載入檔案失敗: {e}")
        return None, None, 44100

    mic1_centered = raw_mic1 - np.mean(raw_mic1)
    mic2_centered = raw_mic2 - np.mean(raw_mic2)
    norm_mic1 = mic1_centered / np.max(np.abs(mic1_centered))
    norm_mic2 = mic2_centered / np.max(np.abs(mic2_centered))

    # ==========================================
    # 2. 計算時差與位移點數
    # ==========================================
    correlation = correlate(norm_mic1, norm_mic2, mode='full')
    lags = np.arange(-len(norm_mic1) + 1, len(norm_mic1))
    
    best_lag_idx = np.argmax(correlation)
    lag_samples = lags[best_lag_idx]
    total_shifted_samples = abs(lag_samples)
    detected_time_delay_ms = (lag_samples / RATE) * 1000
    
    print("-" * 50)
    print(f"📊 [自動對齊分析結果]")
    print(f" ⏱ 兩者時間延遲了：{detected_time_delay_ms:.2f} ms (共 {total_shifted_samples} 個樣點)")
    print("-" * 50)

    # ==========================================
    # 3. 進行訊號對齊裁切
    # ==========================================
    if lag_samples > 0:
        aligned_mic1 = norm_mic1[lag_samples:]
        aligned_mic2 = norm_mic2[:len(norm_mic1)-lag_samples]
    elif lag_samples < 0:
        abs_lag = abs(lag_samples)
        aligned_mic1 = norm_mic1[:len(norm_mic2)-abs_lag]
        aligned_mic2 = norm_mic2[abs_lag:]
    else:
        aligned_mic1 = norm_mic1.copy()
        aligned_mic2 = norm_mic2.copy()

    return norm_mic1, norm_mic2, aligned_mic1, aligned_mic2, RATE

# ==========================================
# 新增功能：波形反轉與抵消實驗
# ==========================================
def phase_cancellation_experiment(norm_mic1, norm_mic2, aligned_mic1, aligned_mic2, rate):
    """將 Mic 1 波形反轉，與 Mic 2 重疊相加，觀察聲音抵消效果"""
    print("⚡ 正在執行相位反轉抵消實驗...")
    
    # 步驟 1：將對齊後的 Mic 1 波形做反轉（正負號顛倒）
    inverted_mic1 = aligned_mic1 * -1.0
    
    # 步驟 2：將反轉後的 Mic 1 與原始 Mic 2 相加（重疊混音）
    cancelled_signal = inverted_mic1 + aligned_mic2
    
    # 計算抵消比例 (評估整體能量減少了多少)
    original_energy = np.sum(aligned_mic2**2)
    cancelled_energy = np.sum(cancelled_signal**2)
    cancellation_ratio = (1.0 - (cancelled_energy / original_energy)) * 100 if original_energy > 0 else 0
    
    print("-" * 50)
    print(f"🔊 [相位抵消實驗結果]")
    print(f"   ➤ 訊號抵消率: {cancellation_ratio:.2f} %")
    if cancellation_ratio > 50:
        print("   ➤ 結論：成功大幅抵消聲音！說明兩者波形高度一致且對齊精準。")
    else:
        print("   ➤ 結論：抵消效果有限，可能是兩支麥克風的頻率響應、距離或環境雜訊落差較大。")
    print("-" * 50)
    
    # ==========================================
    # 視覺化呈現（三張子圖）
    # 用於原始訊號的時間軸
    t_raw = np.linspace(0, len(norm_mic1)/rate, len(norm_mic1))
    # 用於對齊後訊號的時間軸
    t_aligned = np.linspace(0, len(aligned_mic1)/rate, len(aligned_mic1))
    
    # 調整為長寬比更適中的 14x12 畫布
    plt.figure(figsize=(14, 12))
    
    # --- 子圖 1：展示完整 3 秒的 norm_mic1 與 norm_mic2 ---
    plt.subplot(3, 1, 1)
    plt.plot(t_raw, norm_mic1, label='norm_mic1 (Original)', color='blue', alpha=0.6)
    plt.plot(t_raw, norm_mic2, label='norm_mic2 (Original)', color='orange', alpha=0.6)
    plt.title("1. Original Signal Comparison: norm_mic1 vs. norm_mic2 (Full 3s)", fontsize=12, pad=10)
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':')

    # --- 子圖 2：展示對齊後的 Mic 2 與「反轉後的 Mic 1」之放大 0.1 秒局部波形 ---
    plt.subplot(3, 1, 2)
    mid_idx = len(aligned_mic1) // 2
    show_range = slice(mid_idx, mid_idx + int(0.1 * rate))
    
    plt.plot(t_aligned[show_range], aligned_mic2[show_range], label='Mic 2 (Aligned)', color='orange', alpha=0.8)
    plt.plot(t_aligned[show_range], inverted_mic1[show_range], label='Inverted Mic 1 (Mic 1 * -1)', color='blue', linestyle='--', alpha=0.8)
    plt.title("2. Comparison: Aligned Mic 2 vs. Inverted Mic 1 (Zoomed 0.1s)", fontsize=12, pad=10)
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':')

    # --- 子圖 3：展示相加混音後的結果 ---
    plt.subplot(3, 1, 3)
    plt.plot(t_aligned, aligned_mic2, label='Original Mic 2 Signal', color='orange', alpha=0.4)
    plt.plot(t_aligned, cancelled_signal, label=f'Cancelled Result (Remaining Signal)', color='red', alpha=0.8)
    plt.title(f"3. Phase Cancellation Result [Energy Cancelled: {cancellation_ratio:.2f}%]", fontsize=12, pad=10)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':')
    
    # 關鍵修正：透過增加 pad 參數，強制拉開圖與圖之間的上下安全距離，防止標題擋到字
    plt.tight_layout(pad=3.0) 
    plt.show()
    
    # 儲存抵消後的結果為 wav 檔
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    output_path = os.path.join(BASE_DIR, "cancelled_result_Correlation.wav")
    
    max_val = np.max(np.abs(cancelled_signal))
    if max_val > 0:
        cancelled_save = cancelled_signal / max_val
    else:
        cancelled_save = cancelled_signal
        
    wav.write(output_path, rate, np.clip(cancelled_save * 32767, -32768, 32767).astype(np.int16))
    print(f"💾 已儲存抵消後的音訊至 {output_path}")

# ==========================================
# 主程式執行流程
# ==========================================
# 1. 執行對齊
norm1, norm2, aligned1, aligned2, sample_rate = load_normalize_and_align_files()

# 2. 如果成功，接著執行反轉抵消實驗（傳入 4 個訊號）
if aligned1 is not None and aligned2 is not None:
    phase_cancellation_experiment(norm1, norm2, aligned1, aligned2, sample_rate)