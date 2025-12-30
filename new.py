import time
import numpy as np
from scipy.fft import fft, fftfreq
import pygame
from pygame.locals import *
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# Constants
BOARD_ID = BoardIds.CYTON_DAISY_BOARD.value  # Cyton + Daisy: 16 channels
SAMPLE_RATE = BoardShim.get_sampling_rate(BOARD_ID)
NUM_CHANNELS = 16
WINDOW_SIZE = 256  # Time domain display samples
FFT_SIZE = 256     # FFT computation size

# Pygame settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
LEFT_PLOT_WIDTH = SCREEN_WIDTH // 2
RIGHT_PLOT_WIDTH = SCREEN_WIDTH // 2
PLOT_HEIGHT = SCREEN_HEIGHT
CHANNEL_HEIGHT = PLOT_HEIGHT // NUM_CHANNELS

# Right side split: top raw FFT, bottom smoothed FFT
RIGHT_TOP_HEIGHT = PLOT_HEIGHT // 2
RIGHT_BOTTOM_HEIGHT = PLOT_HEIGHT // 2

# Smoothing factor (0.75 = OpenBCI default, 0.9~0.98 more smooth)
SMOOTHING_FACTOR = 0.90

# Channel colors
CHANNEL_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
    (0, 128, 0), (128, 128, 0), (0, 128, 128), (128, 128, 128),
    (255, 192, 203), (165, 42, 42), (75, 0, 130), (240, 230, 140)
]

def main():
    # BrainFlow setup
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyACM0'  # Linux: /dev/ttyACM0 or /dev/ttyUSB0; Windows: 'COMx'
    board = BoardShim(BOARD_ID, params)

    try:
        board.prepare_session()
        board.start_stream()
        board.config_board("~5")  # Optional: improve data quality on Cyton
        print("Stream started successfully.")
    except BrainFlowError as e:
        print(f"Error starting stream: {e}")
        return

    time.sleep(2)  # Wait for initial data

    # Pygame initialization
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BrainWave Visualization - Time (Left) | Raw FFT (Top Right) | Smoothed FFT (Bottom Right)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    title_font = pygame.font.SysFont(None, 28)

    # Data buffers
    time_data = np.zeros((NUM_CHANNELS, WINDOW_SIZE))
    prev_power_db = np.zeros((NUM_CHANNELS, FFT_SIZE // 2))  # For smoothing
    has_data = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # Get new data
        data = board.get_board_data()
        num_samples = data.shape[1]

        if num_samples > 0:
            has_data = True
            eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
            eeg_data = data[eeg_channels[:NUM_CHANNELS]]

            # Preprocessing
            for ch in range(NUM_CHANNELS):
                DataFilter.detrend(eeg_data[ch], DetrendOperations.LINEAR.value)
                DataFilter.perform_bandpass(eeg_data[ch], SAMPLE_RATE, 1.0, 50.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)

            # Update time domain buffer
            new_samples = min(num_samples, WINDOW_SIZE)
            time_data = np.roll(time_data, -new_samples, axis=1)
            time_data[:, -new_samples:] = eeg_data[:, -new_samples:]
        else:
            if not has_data:
                screen.fill((0, 0, 0))
                text = font.render("Waiting for data... Check connection and serial port.", True, (255, 255, 255))
                screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2))
                pygame.display.flip()
                clock.tick(30)
                continue
            else:
                # No new data: linear extrapolation
                time_data = np.roll(time_data, -1, axis=1)
                for ch in range(NUM_CHANNELS):
                    if WINDOW_SIZE >= 2:
                        delta = time_data[ch, -1] - time_data[ch, -2]
                        time_data[ch, -1] = time_data[ch, -1] + delta
                    # else keep last value

        # Compute FFT
        freqs = fftfreq(FFT_SIZE, 1 / SAMPLE_RATE)[:FFT_SIZE // 2]
        raw_fft = np.zeros((NUM_CHANNELS, FFT_SIZE // 2))
        smoothed_fft = np.zeros((NUM_CHANNELS, FFT_SIZE // 2))

        for ch in range(NUM_CHANNELS):
            window = time_data[ch, -FFT_SIZE:]
            if len(window) < FFT_SIZE:
                window = np.pad(window, (FFT_SIZE - len(window), 0))

            yf = fft(window)
            amplitude = 2.0 / FFT_SIZE * np.abs(yf[:FFT_SIZE // 2])
            raw_fft[ch] = amplitude

            # Power in dB + exponential moving average smoothing
            power = np.clip(amplitude ** 2, 1e-10, None)
            power_db = 10 * np.log10(power)
            smoothed_db = (1 - SMOOTHING_FACTOR) * power_db + SMOOTHING_FACTOR * prev_power_db[ch]
            prev_power_db[ch] = smoothed_db

            # Back to amplitude for display
            smoothed_fft[ch] = np.sqrt(10 ** (smoothed_db / 10))

        # Drawing
        screen.fill((0, 0, 0))

        # Left: Time domain
        for ch in range(NUM_CHANNELS):
            color = CHANNEL_COLORS[ch]
            y_offset = ch * CHANNEL_HEIGHT
            scale = CHANNEL_HEIGHT / 2 / 1000.0  # Adjust if signal too small/large
            scaled = (time_data[ch] * scale) + (CHANNEL_HEIGHT / 2 + y_offset)
            scaled = np.clip(scaled, y_offset, y_offset + CHANNEL_HEIGHT - 1)

            pygame.draw.line(screen, (50, 50, 50), (0, y_offset), (LEFT_PLOT_WIDTH, y_offset), 1)
            if ch == NUM_CHANNELS - 1:
                pygame.draw.line(screen, (50, 50, 50), (0, PLOT_HEIGHT), (LEFT_PLOT_WIDTH, PLOT_HEIGHT), 1)

            for i in range(1, WINDOW_SIZE):
                x1 = (i-1) * (LEFT_PLOT_WIDTH / WINDOW_SIZE)
                x2 = i * (LEFT_PLOT_WIDTH / WINDOW_SIZE)
                pygame.draw.line(screen, color, (x1, scaled[i-1]), (x2, scaled[i]), 1)

        # Separator and titles
        pygame.draw.line(screen, (255, 255, 255), (LEFT_PLOT_WIDTH, 0), (LEFT_PLOT_WIDTH, PLOT_HEIGHT), 2)
        pygame.draw.line(screen, (255, 255, 255), (LEFT_PLOT_WIDTH, RIGHT_TOP_HEIGHT), (SCREEN_WIDTH, RIGHT_TOP_HEIGHT), 2)

        raw_title = title_font.render("Raw FFT", True, (200, 200, 200))
        smooth_title = title_font.render("Smoothed FFT (OpenBCI-style)", True, (200, 200, 200))
        screen.blit(raw_title, (LEFT_PLOT_WIDTH + 20, 20))
        screen.blit(smooth_title, (LEFT_PLOT_WIDTH + 20, RIGHT_TOP_HEIGHT + 20))

        # Right top: Raw FFT
        max_raw = np.max(raw_fft) * 1.1 if np.max(raw_fft) > 0 else 1
        for ch in range(NUM_CHANNELS):
            color = CHANNEL_COLORS[ch]
            scaled = RIGHT_TOP_HEIGHT - (raw_fft[ch] / max_raw * RIGHT_TOP_HEIGHT)
            for i in range(1, len(freqs)):
                x1 = LEFT_PLOT_WIDTH + (freqs[i-1] / freqs[-1]) * RIGHT_PLOT_WIDTH
                x2 = LEFT_PLOT_WIDTH + (freqs[i] / freqs[-1]) * RIGHT_PLOT_WIDTH
                pygame.draw.line(screen, color, (x1, scaled[i-1]), (x2, scaled[i]), 1)

        # Right bottom: Smoothed FFT
        max_smooth = np.max(smoothed_fft) * 1.1 if np.max(smoothed_fft) > 0 else 1
        for ch in range(NUM_CHANNELS):
            color = CHANNEL_COLORS[ch]
            scaled = PLOT_HEIGHT - (smoothed_fft[ch] / max_smooth * RIGHT_BOTTOM_HEIGHT)
            scaled += RIGHT_TOP_HEIGHT
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
    except BrainFlowError as e:
        print(f"Error stopping stream: {e}")
    pygame.quit()

if __name__ == "__main__":
    main()
