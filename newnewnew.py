import time
import numpy as np
from scipy.fft import fft, fftfreq
import pygame
from pygame.locals import *
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# Constants
BOARD_ID = BoardIds.CYTON_DAISY_BOARD.value
SAMPLE_RATE = BoardShim.get_sampling_rate(BOARD_ID)
NUM_CHANNELS = 16
WINDOW_SIZE = 256
FFT_SIZE = 1024

SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440

TOP_HEIGHT = SCREEN_HEIGHT // 2
BOTTOM_HEIGHT = SCREEN_HEIGHT // 2
LEFT_WIDTH = SCREEN_WIDTH // 2
RIGHT_WIDTH = SCREEN_WIDTH // 2

CHANNEL_HEIGHT = TOP_HEIGHT // NUM_CHANNELS

SMOOTHING_FACTOR = 0.98
FREQ_SMOOTH_WINDOW = 5
TIME_DOMAIN_SCALE_UV = 180

CHANNEL_COLORS = [
    (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80),
    (255, 80, 255), (80, 255, 255), (255, 180, 50), (180, 80, 180),
    (80, 180, 80), (180, 180, 80), (80, 180, 180), (180, 180, 180),
    (255, 150, 180), (200, 80, 80), (120, 80, 200), (240, 220, 150)
]

BANDS = [
    ("Delta",  0.5, 4),
    ("Theta",  4,   8),
    ("Alpha",  8,  13),
    ("Low β", 13,  20),
    ("High β",20,  30),
    ("Gamma", 30,  50),
]
NUM_BANDS = len(BANDS)

TOTAL_BAND_COLORS = [
    (255, 60, 60),    # Delta  - 红
    (255, 140, 50),   # Theta  - 橙
    (255, 220, 60),   # Alpha  - 黄
    (100, 220, 100),  # Low β  - 绿
    (80,  200, 255),  # High β - 青
    (120, 100, 255),  # Gamma  - 紫
]

def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyACM0'  # 请修改为实际端口
    board = BoardShim(BOARD_ID, params)

    try:
        board.prepare_session()
        board.start_stream()
        board.config_board("~4")
        print("Stream started successfully.")
    except BrainFlowError as e:
        print(f"Error starting stream: {e}")
        return

    time.sleep(2)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BrainWave Monitor - Clean Minimal Layout")
    clock = pygame.time.Clock()

    large_font = pygame.font.SysFont("Arial", 36)
    font = pygame.font.SysFont("Arial", 28)
    small_font = pygame.font.SysFont("Arial", 24)

    BUFFER_CAPACITY = FFT_SIZE * 4
    circular_buffer = np.zeros((NUM_CHANNELS, BUFFER_CAPACITY))
    buffer_pos = 0
    has_data = False

    time_data = np.zeros((NUM_CHANNELS, WINDOW_SIZE))
    hamming_win = np.hamming(WINDOW_SIZE)
    win_sum = np.sum(hamming_win)

    prev_power_db = np.zeros((NUM_CHANNELS, FFT_SIZE // 2))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # 数据采集与处理（不变）
        data = board.get_current_board_data(SAMPLE_RATE)
        num_samples = data.shape[1]
        new_sample = None
        if num_samples > 0:
            has_data = True
            eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
            eeg_data = data[eeg_channels[:NUM_CHANNELS]]

            for ch in range(NUM_CHANNELS):
                DataFilter.detrend(eeg_data[ch], DetrendOperations.LINEAR.value)
                DataFilter.perform_bandpass(eeg_data[ch], SAMPLE_RATE, 1.0, 50.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)

            new_sample = eeg_data[:, -1]

            end_pos = buffer_pos + num_samples
            if end_pos <= BUFFER_CAPACITY:
                circular_buffer[:, buffer_pos:end_pos] = eeg_data
            else:
                split = BUFFER_CAPACITY - buffer_pos
                circular_buffer[:, buffer_pos:] = eeg_data[:, :split]
                circular_buffer[:, :end_pos - BUFFER_CAPACITY] = eeg_data[:, split:]
            buffer_pos = end_pos % BUFFER_CAPACITY

        if not has_data:
            screen.fill((10, 10, 20))
            text = large_font.render("Waiting for data...", True, (255, 255, 255))
            screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 - text.get_height()//2))
            pygame.display.flip()
            clock.tick(30)
            continue

        time_data = np.roll(time_data, -1, axis=1)
        if new_sample is not None:
            time_data[:, -1] = new_sample
        else:
            time_data[:, -1] = time_data[:, -2]

        start_idx = buffer_pos - WINDOW_SIZE
        if start_idx >= 0:
            recent = circular_buffer[:, start_idx:buffer_pos]
        else:
            recent = np.hstack((circular_buffer[:, start_idx:], circular_buffer[:, :buffer_pos]))

        freqs = fftfreq(FFT_SIZE, 1 / SAMPLE_RATE)[:FFT_SIZE // 2]
        smoothed_fft = np.zeros((NUM_CHANNELS, FFT_SIZE // 2))
        band_powers = np.zeros((NUM_CHANNELS, NUM_BANDS))

        for ch in range(NUM_CHANNELS):
            windowed = recent[ch] * hamming_win
            padded = np.pad(windowed, (0, FFT_SIZE - WINDOW_SIZE), 'constant')
            yf = fft(padded)
            amplitude = (2.0 / win_sum) * np.abs(yf[:FFT_SIZE // 2])
            power = np.clip(amplitude ** 2, 1e-10, None)
            power_db = 10 * np.log10(power)

            smoothed_db = (1 - SMOOTHING_FACTOR) * power_db + SMOOTHING_FACTOR * prev_power_db[ch]
            prev_power_db[ch] = smoothed_db

            smoothed_db_smooth = np.convolve(smoothed_db, np.ones(FREQ_SMOOTH_WINDOW)/FREQ_SMOOTH_WINDOW, mode='same')
            smoothed_power = 10 ** (smoothed_db_smooth / 10)
            smoothed_fft[ch] = np.sqrt(smoothed_power)

            for b, (_, f_low, f_high) in enumerate(BANDS):
                mask = (freqs >= f_low) & (freqs <= f_high)
                band_powers[ch, b] = np.sum(smoothed_power[mask])

        total_band_powers = np.sum(band_powers, axis=0)

        # === 绘图 ===
        screen.fill((10, 10, 20))

        # 左上：时间域波形
        for ch in range(NUM_CHANNELS):
            color = CHANNEL_COLORS[ch]
            y_offset = ch * CHANNEL_HEIGHT + 60
            scale_factor = CHANNEL_HEIGHT / 2.2 / TIME_DOMAIN_SCALE_UV
            scaled = (time_data[ch] * scale_factor) + (CHANNEL_HEIGHT / 2 + y_offset)
            scaled = np.clip(scaled, y_offset, y_offset + CHANNEL_HEIGHT - 1)

            pygame.draw.line(screen, (40, 40, 40), (40, y_offset + CHANNEL_HEIGHT//2), (LEFT_WIDTH - 40, y_offset + CHANNEL_HEIGHT//2), 1)

            for i in range(1, WINDOW_SIZE):
                x1 = 40 + (i-1) * ((LEFT_WIDTH - 80) / WINDOW_SIZE)
                x2 = 40 + i * ((LEFT_WIDTH - 80) / WINDOW_SIZE)
                pygame.draw.line(screen, color, (x1, scaled[i-1]), (x2, scaled[i]), 3)

        # 右上：各通道 Band Power —— 关键修改：频段名移到下方
        bar_width = 40
        bar_spacing = 20
        total_bar_w = NUM_BANDS * (bar_width + bar_spacing) - bar_spacing
        bar_center_x = LEFT_WIDTH + (RIGHT_WIDTH - total_bar_w) // 2

        global_max = np.max(band_powers) if np.max(band_powers) > 0 else 1

        for ch in range(NUM_CHANNELS):
            y_base = 100 + ch * CHANNEL_HEIGHT  # 柱子区域起始Y（留出上方空间）
            color = CHANNEL_COLORS[ch]
            ch_label = font.render(f"Ch{ch+1}", True, color)
            screen.blit(ch_label, (LEFT_WIDTH + 80, y_base + CHANNEL_HEIGHT // 2 - 10))

            for b in range(NUM_BANDS):
                power = band_powers[ch, b]
                height = max((power / global_max) * (CHANNEL_HEIGHT - 40), 4)
                x = bar_center_x + b * (bar_width + bar_spacing)

                # 绘制柱子（从底部向上生长）
                pygame.draw.rect(screen, color,
                                 (x, y_base + (CHANNEL_HEIGHT - height), bar_width, height))

        # 频段名称放在所有通道的最下方（只显示一次）
        band_name_y = 100 + NUM_CHANNELS * CHANNEL_HEIGHT - 40
        for b, (name, _, _) in enumerate(BANDS):
            x = bar_center_x + b * (bar_width + bar_spacing)
            label = large_font.render(name, True, (230, 230, 230))
            screen.blit(label, (x + bar_width // 2 - label.get_width() // 2, band_name_y))

        # 左下：FFT 频谱
        max_smooth = np.max(smoothed_fft)
        if max_smooth == 0:
            max_smooth = 1e-6
        max_smooth *= 1.05

        plot_bottom = SCREEN_HEIGHT - 80
        plot_top = TOP_HEIGHT + 80

        for ch in range(NUM_CHANNELS):
            color = CHANNEL_COLORS[ch]
            scaled = plot_bottom - (smoothed_fft[ch] / max_smooth * (plot_bottom - plot_top))
            for i in range(1, len(freqs)):
                x1 = 80 + (freqs[i-1] / 50.0) * (LEFT_WIDTH - 160)
                x2 = 80 + (freqs[i] / 50.0) * (LEFT_WIDTH - 160)
                pygame.draw.line(screen, color, (x1, scaled[i-1]), (x2, scaled[i]), 2)

        for f_mark in [0, 10, 20, 30, 40, 50]:
            x = 80 + (f_mark / 50.0) * (LEFT_WIDTH - 160)
            pygame.draw.line(screen, (80, 80, 80), (x, plot_bottom), (x, plot_bottom + 20), 2)
            label = font.render(str(f_mark), True, (200, 200, 200))
            screen.blit(label, (x - 15, plot_bottom + 30))
        screen.blit(font.render("Hz", True, (200, 200, 200)), (LEFT_WIDTH - 100, plot_bottom + 30))

        # 右下：Total Band Power（保持不变）
        total_bar_width = 100
        total_spacing = 60
        total_start_x = LEFT_WIDTH + (RIGHT_WIDTH - (NUM_BANDS * (total_bar_width + total_spacing) - total_spacing)) // 2
        
        total_base_y = TOP_HEIGHT + BOTTOM_HEIGHT // 2 + 120
        total_max_h = BOTTOM_HEIGHT // 2 - 280

        total_max_power = np.max(total_band_powers) if np.max(total_band_powers) > 0 else 1

        for b in range(NUM_BANDS):
            power = total_band_powers[b]
            height = max((power / total_max_power) * total_max_h, 12)

            x = total_start_x + b * (total_bar_width + total_spacing)
            color = TOTAL_BAND_COLORS[b]

            pygame.draw.rect(screen, color,
                             (x, total_base_y - height, total_bar_width, height * 2),
                             border_radius=16)

            pygame.draw.rect(screen, (min(color[0]+60,255), min(color[1]+60,255), min(color[2]+60,255)),
                             (x, total_base_y - height, total_bar_width, height * 2),
                             width=4, border_radius=16)

            val_text = font.render(f"{power:.1f}", True, (255, 255, 200))
            screen.blit(val_text, (x + total_bar_width//2 - val_text.get_width()//2,
                                   total_base_y - height - 70))

            name_text = large_font.render(BANDS[b][0], True, (255, 255, 255))
            screen.blit(name_text, (x + total_bar_width//2 - name_text.get_width()//2,
                                    total_base_y + total_max_h//2 + 80))

        pygame.display.flip()
        clock.tick(30)

    # Cleanup
    try:
        board.stop_stream()
        board.release_session()
    except:
        pass
    pygame.quit()

if __name__ == "__main__":
    main()
