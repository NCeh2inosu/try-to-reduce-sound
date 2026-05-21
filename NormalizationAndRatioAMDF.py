#基於 AMDF 計算最小值與最大值之比，或者局部谷值與峰值之間的比率。
#用於硬體效能有限，但麥克風增益不對等、音量會隨時間動態飄移、或有低頻直流偏壓的現實環境。
#訊號中完全沒有聲音（極度安靜的靜音段）。此時分母會趨近於 0，會導致程式出現 Divide by zero（除以零）的崩潰風險。
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wav

def load_normalize_and_align_files_ratio_amdf():
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
    # 2. 使用 Ratio AMDF 計算時差與位移點數
    # ==========================================
    print("⚡ 正在計算 Ratio AMDF (比例平均幅差) 以尋找最佳對齊點...")
    
    # 限制搜尋範圍在正負 0.25 秒內
    max_delay_samples = int(0.25 * RATE)
    lags = np.arange(-max_delay_samples, max_delay_samples + 1)
    ratio_amdf_values = np.zeros(len(lags))
    
    # 為了加速與穩定度，固定計算中間重疊區域
    start_idx = max_delay_samples
    end_idx = min_len - max_delay_samples
    
    ref_signal = norm_mic1[start_idx:end_idx]
    sum_abs_ref = np.sum(np.abs(ref_signal)) # 分母的一部分：Mic 1 的絕對值總和
    
    for i, lag in enumerate(lags):
        shifted_signal = norm_mic2[start_idx - lag : end_idx - lag]
        
        # 分子：兩者訊號的絕對差值和
        numerator = np.sum(np.abs(ref_signal - shifted_signal))
        # 分母：兩者各自絕對值和相加
        denominator = sum_abs_ref + np.sum(np.abs(shifted_signal))
        
        # 計算 Ratio AMDF 比例值
        ratio_amdf_values[i] = numerator / denominator if denominator > 0 else 1.0
    
    # 尋找 Ratio AMDF 的最低谷底 (Global Minimum)
    best_lag_idx = np.argmin(ratio_amdf_values)
    lag_samples = lags[best_lag_idx]       # 帶正負號的相對位移量
    total_shifted_samples = abs(lag_samples) # 一共位移的絕對點數
    
    detected_time_delay_ms = (lag_samples / RATE) * 1000
    
    print("-" * 50)
    print(f"📊 [Ratio AMDF 自動對齊分析結果]")
    print(f" ⏱ 兩者時間延遲了：{detected_time_delay_ms:.2f} ms (共偏離 {total_shifted_samples} 個樣點)")
    print(f" 📉 谷底 Ratio 數值：{ratio_amdf_values[best_lag_idx]:.4f} (越接近 0 代表對齊重合度越高)")
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

    # 視覺化 Ratio AMDF 函數圖
    plt.figure(figsize=(10, 4))
    plt.plot(lags, ratio_amdf_values, color='dodgerblue', label='Ratio AMDF Value')
    plt.axvline(x=lag_samples, color='red', linestyle='--', 
                label=f'Global Minimum (Lag = {lag_samples})')
    plt.title("Ratio AMDF Function - Search for Minimum (Bounded 0 to 1)", fontsize=12)
    plt.xlabel("Sample Offset (lags)")
    plt.ylabel("Ratio Amplitude Difference")
    plt.ylim(0, 1.05) # 鎖定 Y 軸範圍在 0 ~ 1 方便觀察歸一化效果
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()

    return aligned_mic1, aligned_mic2, RATE

# ==========================================
# 4. 波形反轉與抵消實驗
# ==========================================
def phase_cancellation_experiment(aligned_mic1, aligned_mic2, rate):
    """將 Mic 1 波形反轉，與 Mic 2 重疊相加，觀察聲音抵消效果"""
    print("⚡ 正在執行相位反轉抵消實驗...")
    
    inverted_mic1 = aligned_mic1 * -1.0
    cancelled_signal = inverted_mic1 + aligned_mic2
    
    original_energy = np.sum(aligned_mic2**2)
    cancelled_energy = np.sum(cancelled_signal**2)
    cancellation_ratio = (1.0 - (cancelled_energy / original_energy)) * 100 if original_energy > 0 else 0
    
    print("-" * 50)
    print(f"🔊 [相位抵消實驗結果]")
    print(f"   ➤ 訊號抵消率: {cancellation_ratio:.2f} %")
    print("-" * 50)
    
    t_aligned = np.linspace(0, len(aligned_mic1)/rate, len(aligned_mic1))
    
    plt.figure(figsize=(14, 10))
    
    # 子圖 1：局部波形對比
    plt.subplot(2, 1, 1)
    mid_idx = len(aligned_mic1) // 2
    show_range = slice(mid_idx, mid_idx + int(0.1 * rate))
    
    plt.plot(t_aligned[show_range], aligned_mic2[show_range], label='Mic 2 (Aligned)', color='orange', alpha=0.8)
    plt.plot(t_aligned[show_range], inverted_mic1[show_range], label='Inverted Mic 1 (Mic 1 * -1)', color='blue', linestyle='--', alpha=0.8)
    plt.title("1. Comparison: Aligned Mic 2 vs. Inverted Mic 1 (Zoomed 0.1s)", fontsize=12)
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, linestyle=':')

    # 子圖 2：抵消波形
    plt.subplot(2, 1, 2)
    plt.plot(t_aligned, aligned_mic2, label='Original Mic 2 Signal', color='orange', alpha=0.4)
    plt.plot(t_aligned, cancelled_signal, label=f'Cancelled Result (Remaining Signal)', color='red', alpha=0.8)
    plt.title(f"2. Phase Cancellation Result [Energy Cancelled: {cancellation_ratio:.2f}%]", fontsize=12)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, linestyle=':')
    
    plt.tight_layout()
    plt.show()
    
    # 儲存
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    output_path = os.path.join(BASE_DIR, "cancelled_result_ratio_AMDF.wav")
    
    max_val = np.max(np.abs(cancelled_signal))
    cancelled_save = cancelled_signal / max_val if max_val > 0 else cancelled_signal
    wav.write(output_path, rate, np.clip(cancelled_save * 32767, -32768, 32767).astype(np.int16))
    print(f"💾 已儲存抵消後的音訊至 {output_path}")

# ==========================================
# 主程式執行流程
# ==========================================
aligned1, aligned2, sample_rate = load_normalize_and_align_files_ratio_amdf()

if aligned1 is not None and aligned2 is not None:
    phase_cancellation_experiment(aligned1, aligned2, sample_rate)