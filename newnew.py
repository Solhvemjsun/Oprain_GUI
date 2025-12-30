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
WINDOW_SIZE = 256          # Samples for display and FFT window
FFT_SIZE = 1024            # Zero-padding for frequency resolution

# Pygame settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
LEFT_PLOT_WIDTH = SCREEN_WIDTH // 2
RIGHT_PLOT_WIDTH = SCREEN_WIDTH // 2
PLOT_HEIGHT = SCREEN_HEIGHT
CHANNEL_HEIGHT = PLOT_HEIGHT // NUM_CHANNELS

SMOOTHING_FACTOR = 0.98
FREQ_SMOOTH_WINDOW = 5

# Time-domain scaling (smaller = larger waveform)
TIME_DOMAIN_SCALE_UV = 200

CHANNEL_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
    (0, 128, 0), (128, 128, 0), (0, 128, 128), (128, 128, 128),
    (255, 192, 203), (165, 42, 42), (75, 0, 130), (240, 230, 140)
]

def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyACM0'  # Change to your actual port
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
    pygame.display.set_caption("BrainWave - Smooth Time Domain (Left) | Ultra-Smooth FFT (Right)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    title_font = pygame.font.SysFont(None, 32)

    # Circular buffer for robust FFT
    BUFFER_CAPACITY = FFT_SIZE * 4
    circular_buffer = np.zeros((NUM_CHANNELS, BUFFER_CAPACITY))
    buffer_pos = 0
    has_data = False

    # Rolling buffer for smooth time-domain display (advances by exactly 1 sample per frame)
    time_data = np.zeros((NUM_CHANNELS, WINDOW_SIZE))

    # Pre-compute Hamming window
    hamming_win = np.hamming(WINDOW_SIZE)
    win_sum = np.sum(hamming_win)

    # Smoothed power history for FFT
    prev_power_db = np.zeros((NUM_CHANNELS, FFT_SIZE // 2))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # === Unified data acquisition ===
        data = board.get_current_board_data(SAMPLE_RATE)  # Get latest available data (up to ~1 sec)
        num_samples = data.shape[1]

        new_sample = None  # Will hold the newest single sample if available
        if num_samples > 0:
            has_data = True
            eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
            eeg_data = data[eeg_channels[:NUM_CHANNELS]]

            # Preprocessing for all new samples
            for ch in range(NUM_CHANNELS):
                DataFilter.detrend(eeg_data[ch], DetrendOperations.LINEAR.value)
                DataFilter.perform_bandpass(eeg_data[ch], SAMPLE_RATE, 1.0, 50.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)

            # Take only the newest sample for time-domain smooth scrolling
            new_sample = eeg_data[:, -1]

            # Update circular buffer with ALL new samples (for stable FFT)
            end_pos = buffer_pos + num_samples
            if end_pos <= BUFFER_CAPACITY:
                circular_buffer[:, buffer_pos:end_pos] = eeg_data
            else:
                split = BUFFER_CAPACITY - buffer_pos
                circular_buffer[:, buffer_pos:] = eeg_data[:, :split]
                circular_buffer[:, :end_pos - BUFFER_CAPACITY] = eeg_data[:, split:]
            buffer_pos = end_pos % BUFFER_CAPACITY

        if not has_data:
            screen.fill((0, 0, 0))
            text = font.render("Waiting for data... Check connection.", True, (255, 255, 255))
            screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2))
            pygame.display.flip()
            clock.tick(30)
            continue

        # === Update time-domain buffer: always advance by exactly 1 sample ===
        time_data = np.roll(time_data, -1, axis=1)
        if new_sample is not None:
            time_data[:, -1] = new_sample                    # New real sample
        else:
            time_data[:, -1] = time_data[:, -2]              # Hold previous (no gap or jump)

        # === Extract latest WINDOW_SIZE samples from circular buffer for FFT ===
        start_idx = buffer_pos - WINDOW_SIZE
        if start_idx >= 0:
            recent = circular_buffer[:, start_idx:buffer_pos]
        else:
            recent = np.hstack((circular_buffer[:, start_idx:], circular_buffer[:, :buffer_pos]))

        # === Ultra-Smooth FFT Computation ===
        freqs = fftfreq(FFT_SIZE, 1 / SAMPLE_RATE)[:FFT_SIZE // 2]
        smoothed_fft = np.zeros((NUM_CHANNELS, FFT_SIZE // 2))

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
            smoothed_fft[ch] = np.sqrt(10 ** (smoothed_db_smooth / 10))

        # === Drawing ===
        screen.fill((0, 0, 0))

        # Left: Smooth continuous time-domain waveforms
        for ch in range(NUM_CHANNELS):
            color = CHANNEL_COLORS[ch]
            y_offset = ch * CHANNEL_HEIGHT

            scale_factor = CHANNEL_HEIGHT / 2 / TIME_DOMAIN_SCALE_UV
            scaled = (time_data[ch] * scale_factor) + (CHANNEL_HEIGHT / 2 + y_offset)
            scaled = np.clip(scaled, y_offset, y_offset + CHANNEL_HEIGHT - 1)

            pygame.draw.line(screen, (50, 50, 50), (0, y_offset), (LEFT_PLOT_WIDTH, y_offset), 1)
            if ch == NUM_CHANNELS - 1:
                pygame.draw.line(screen, (50, 50, 50), (0, PLOT_HEIGHT), (LEFT_PLOT_WIDTH, PLOT_HEIGHT), 1)

            for i in range(1, WINDOW_SIZE):
                x1 = (i-1) * (LEFT_PLOT_WIDTH / WINDOW_SIZE)
                x2 = i * (LEFT_PLOT_WIDTH / WINDOW_SIZE)
                pygame.draw.line(screen, color, (x1, scaled[i-1]), (x2, scaled[i]), 2)

        # Divider and title
        pygame.draw.line(screen, (255, 255, 255), (LEFT_PLOT_WIDTH, 0), (LEFT_PLOT_WIDTH, PLOT_HEIGHT), 3)
        smooth_title = title_font.render("Ultra-Smooth FFT", True, (200, 200, 200))
        screen.blit(smooth_title, (LEFT_PLOT_WIDTH + 20, 20))

        # Right: Ultra-Smooth FFT (stable normalization)
        max_smooth = np.max(smoothed_fft)
        if max_smooth == 0:
            max_smooth = 1e-6
        max_smooth *= 1.05

        for ch in range(NUM_CHANNELS):
            color = CHANNEL_COLORS[ch]
            scaled = PLOT_HEIGHT - (smoothed_fft[ch] / max_smooth * PLOT_HEIGHT)
            for i in range(1, len(freqs)):
                x1 = LEFT_PLOT_WIDTH + (freqs[i-1] / freqs[-1]) * RIGHT_PLOT_WIDTH
                x2 = LEFT_PLOT_WIDTH + (freqs[i] / freqs[-1]) * RIGHT_PLOT_WIDTH
                pygame.draw.line(screen, color, (x1, scaled[i-1]), (x2, scaled[i]), 1)

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
