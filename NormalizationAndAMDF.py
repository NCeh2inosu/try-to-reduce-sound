#AMDF (Average Magnitude Difference Function, 平均振幅差分函數)
#將訊號對齊後，逐點相減取絕對值並累加。當延遲時間恰好等於訊號週期時，對應的 AMDF 值會出現局部最低點（谷值）。
#快速，在硬體實現上非常有效率。
#微控制器（MCU）、晶片底層、DSP、嵌入式系統（如藍牙耳機晶片），可以用極少的時鐘週期（Clock Cycles）搞定。
#兩支麥克風硬體增益（音量）差很多，或是環境雜訊很大時，絕對不能用，它會找不到明確的谷底。
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wav

def load_normalize_and_align_files_amdf():
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
    # 2. 使用 AMDF 計算時差與位移點數
    # ==========================================
    print("⚡ 正在計算 AMDF (平均幅差) 以尋找最佳對齊點...")
    
    # 為了避免全面積滑動導致計算時間過長（時間複雜度 $O(N^2)$）
    # 我們限制搜尋範圍在正負 0.25 秒內（通常錄音時間差不會超過這個範圍）
    max_delay_samples = int(0.25 * RATE)
    lags = np.arange(-max_delay_samples, max_delay_samples + 1)
    amdf_values = np.zeros(len(lags))
    
    # 執行 AMDF 計算：D(k) = (1/N) * sum(|x(n) - y(n-k)|)
    # 為了加速運算，固定計算中間重疊區域
    start_idx = max_delay_samples
    end_idx = min_len - max_delay_samples
    N = end_idx - start_idx
    
    ref_signal = norm_mic1[start_idx:end_idx]
    
    for i, lag in enumerate(lags):
        # 根據不同的 lag 滑動移位 Mic 2
        shifted_signal = norm_mic2[start_idx - lag : end_idx - lag]
        # 計算平均絕對誤差
        amdf_values[i] = np.mean(np.abs(ref_signal - shifted_signal))
    
    # 【關鍵：找 AMDF 的最低谷】
    best_lag_idx = np.argmin(amdf_values)
    lag_samples = lags[best_lag_idx]       # 帶正負號的相對位移量
    total_shifted_samples = abs(lag_samples) # 一共位移的絕對點數
    
    detected_time_delay_ms = (lag_samples / RATE) * 1000
    
    print("-" * 50)
    print(f"📊 [AMDF 自動對齊分析結果]")
    print(f" ⏱ 兩者時間延遲了：{detected_time_delay_ms:.2f} ms (共偏離 {total_shifted_samples} 個樣點)")
    print(" 💡 提示：AMDF 的對齊依據是尋找函數圖形的「最低谷 (Minimum)」")
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

    # ==========================================
    # 視覺化 AMDF 特有圖形
    # ==========================================
    plt.figure(figsize=(10, 4))
    plt.plot(lags, amdf_values, color='teal', label='AMDF Value')
    plt.axvline(x=lag_samples, color='red', linestyle='--', 
                label=f'Global Minimum (Lag = {lag_samples})')
    plt.title("AMDF (Average Magnitude Difference Function) - Search for Minimum", fontsize=12)
    plt.xlabel("Sample Offset (lags)")
    plt.ylabel("Average Absolute Difference")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()

    return aligned_mic1, aligned_mic2, RATE

# ==========================================
# 4. 波形反轉與抵消實驗 (保持相同邏輯)
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
    if cancellation_ratio > 50:
        print("   ➤ 結論：成功大幅抵消聲音！證明 AMDF 對齊相當精準。")
    else:
        print("   ➤ 結論：抵消效果有限，可能是硬體落差或環境雜訊落差較大。")
    print("-" * 50)
    
    # 視覺化呈現
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
    output_path = os.path.join(BASE_DIR, "cancelled_result_AMDF.wav")
    
    max_val = np.max(np.abs(cancelled_signal))
    cancelled_save = cancelled_signal / max_val if max_val > 0 else cancelled_signal
    wav.write(output_path, rate, np.clip(cancelled_save * 32767, -32768, 32767).astype(np.int16))
    print(f"💾 已儲存抵消後的音訊至 {output_path}")

# ==========================================
# 主程式執行流程
# ==========================================
aligned1, aligned2, sample_rate = load_normalize_and_align_files_amdf()

if aligned1 is not None and aligned2 is not None:
    phase_cancellation_experiment(aligned1, aligned2, sample_rate)