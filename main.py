import time
import numpy as np
from scipy.fft import fft, fftfreq
import pygame
from pygame.locals import *
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# Constants
BOARD_ID = BoardIds.CYTON_DAISY_BOARD.value  # Cyton + Daisy has 16 channels
SAMPLE_RATE = BoardShim.get_sampling_rate(BOARD_ID)
NUM_CHANNELS = 16
WINDOW_SIZE = 256  # Number of samples to display per channel in time domain
FFT_SIZE = 256  # For FFT computation

# Pygame window settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
LEFT_PLOT_WIDTH = SCREEN_WIDTH // 2
RIGHT_PLOT_WIDTH = SCREEN_WIDTH // 2
PLOT_HEIGHT = SCREEN_HEIGHT
CHANNEL_HEIGHT = PLOT_HEIGHT // NUM_CHANNELS  # Height per channel track on left

# Colors for each channel (16 different colors)
CHANNEL_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 0),    # Dark Green
    (128, 128, 0),  # Olive
    (0, 128, 128),  # Teal
    (128, 128, 128),# Gray
    (255, 192, 203),# Pink
    (165, 42, 42),  # Brown
    (75, 0, 130),   # Indigo
    (240, 230, 140) # Khaki
]

def main():
    # Initialize BrainFlow with logging
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyACM0'  # Change to your actual serial port, e.g., '/dev/ttyUSB0' on Linux/Mac, or check device manager on Windows
    board = BoardShim(BOARD_ID, params)

    try:
        board.prepare_session()
        board.start_stream()
        print("Stream started successfully.")
    except BrainFlowError as e:
        print(f"Error starting stream: {e}")
        return

    # Allow some time for data to accumulate
    time.sleep(2)

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BrainWave Visualization - Time Domain (Left) & Frequency Domain (Right)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)  # For displaying messages on screen

    # Buffers for time domain data (rolling buffer)
    time_data = np.zeros((NUM_CHANNELS, WINDOW_SIZE))
    has_data = False  # Flag to check if we have received any data yet

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # Get all available data from BrainFlow (better for real-time to clear buffer)
        data = board.get_board_data()
        num_samples = data.shape[1]

        if num_samples > 0:
            print(f"Received {num_samples} samples.")
            has_data = True
            eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
            eeg_data = data[eeg_channels[:NUM_CHANNELS]]  # Get first 16 EEG channels

            # Preprocess: Detrend and filter (optional, similar to OpenBCI)
            for ch in range(NUM_CHANNELS):
                DataFilter.detrend(eeg_data[ch], DetrendOperations.LINEAR.value)
                DataFilter.perform_bandpass(eeg_data[ch], SAMPLE_RATE, 1.0, 50.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)

            # Update time domain buffer with latest samples (up to WINDOW_SIZE)
            new_samples = min(num_samples, WINDOW_SIZE)
            time_data = np.roll(time_data, -new_samples, axis=1)
            time_data[:, -new_samples:] = eeg_data[:, -new_samples:]
        else:
            if not has_data:
                # No data yet, display message
                screen.fill((0, 0, 0))
                text = font.render("No data available from board. Check connection and serial port.", True, (255, 255, 255))
                screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - text.get_height() // 2))
                pygame.display.flip()
                clock.tick(30)
                continue
            else:
                # No new data, perform linear interpolation for one new sample (extrapolation)
                print("No new data, performing linear interpolation.")
                new_samples = np.zeros(NUM_CHANNELS)
                for ch in range(NUM_CHANNELS):
                    if WINDOW_SIZE >= 2:
                        last = time_data[ch, -1]
                        prev = time_data[ch, -2]
                        new_samples[ch] = last + (last - prev)  # Linear extrapolation
                    else:
                        new_samples[ch] = time_data[ch, -1]  # Repeat last if not enough history

                # Roll and add the new interpolated sample
                time_data = np.roll(time_data, -1, axis=1)
                time_data[:, -1] = new_samples

        # Compute FFT using the last FFT_SIZE samples from time_data
        freqs = fftfreq(FFT_SIZE, 1 / SAMPLE_RATE)[:FFT_SIZE // 2]
        fft_data = np.zeros((NUM_CHANNELS, FFT_SIZE // 2))
        for ch in range(NUM_CHANNELS):
            recent_data = time_data[ch, -FFT_SIZE:]
            if len(recent_data) < FFT_SIZE:
                recent_data = np.pad(recent_data, (FFT_SIZE - len(recent_data), 0), 'constant')
            yf = fft(recent_data)
            fft_data[ch] = 2.0 / FFT_SIZE * np.abs(yf[:FFT_SIZE // 2])

        # Clear screen
        screen.fill((0, 0, 0))

        # Draw left side: Time domain, vertical tracks
        for ch in range(NUM_CHANNELS):
            color = CHANNEL_COLORS[ch]
            y_offset = ch * CHANNEL_HEIGHT
            # Scale data to fit channel height (adjust scale factor if needed, assuming uV)
            scale_factor = CHANNEL_HEIGHT / 2 / 1000.0  # Adjust based on expected signal amplitude
            scaled_data = (time_data[ch] * scale_factor) + (CHANNEL_HEIGHT / 2 + y_offset)
            scaled_data = np.clip(scaled_data, y_offset, y_offset + CHANNEL_HEIGHT - 1)

            # Draw horizontal lines for each channel
            pygame.draw.line(screen, (50, 50, 50), (0, y_offset), (LEFT_PLOT_WIDTH, y_offset), 1)
            if ch == NUM_CHANNELS - 1:
                pygame.draw.line(screen, (50, 50, 50), (0, PLOT_HEIGHT), (LEFT_PLOT_WIDTH, PLOT_HEIGHT), 1)

            # Draw waveform
            for i in range(1, WINDOW_SIZE):
                x1 = (i - 1) * (LEFT_PLOT_WIDTH / WINDOW_SIZE)
                x2 = i * (LEFT_PLOT_WIDTH / WINDOW_SIZE)
                y1 = scaled_data[i - 1]
                y2 = scaled_data[i]
                pygame.draw.line(screen, color, (x1, y1), (x2, y2), 1)

        # Draw vertical separator
        pygame.draw.line(screen, (255, 255, 255), (LEFT_PLOT_WIDTH, 0), (LEFT_PLOT_WIDTH, PLOT_HEIGHT), 2)

        # Draw right side: Frequency domain, overlaid
        max_amp = np.max(fft_data) * 1.1 if np.max(fft_data) > 0 else 1
        for ch in range(NUM_CHANNELS):
            color = CHANNEL_COLORS[ch]
            scaled_fft = PLOT_HEIGHT - (fft_data[ch] / max_amp * PLOT_HEIGHT)  # Invert Y-axis for bottom-up plotting

            # Draw spectrum lines
            for i in range(1, len(freqs)):
                x1 = LEFT_PLOT_WIDTH + (freqs[i - 1] / np.max(freqs)) * RIGHT_PLOT_WIDTH
                x2 = LEFT_PLOT_WIDTH + (freqs[i] / np.max(freqs)) * RIGHT_PLOT_WIDTH
                y1 = scaled_fft[i - 1]
                y2 = scaled_fft[i]
                pygame.draw.line(screen, color, (x1, y1), (x2, y2), 1)

        # Update display
        pygame.display.flip()
        clock.tick(30)  # ~30 FPS

    # Cleanup
    try:
        board.stop_stream()
        board.release_session()
    except BrainFlowError as e:
        print(f"Error stopping stream: {e}")
    pygame.quit()

if __name__ == "__main__":
    main()
