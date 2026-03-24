% MODULE 1: Signal Generation
% Generates synthetic time-series signals with normal and anomalous behaviors.

% Define standard parameters
fs = 1000;             % Sample rate (1 kHz)
t = (0:1/fs:10-1/fs)'; % 10 seconds of data
n_samples = length(t);
label = zeros(n_samples, 1);

% 1. Normal Signal: Sine waves + Gaussian Noise
f1 = 5;  % 5 Hz prominent frequency
f2 = 12; % 12 Hz secondary frequency
clean_signal = 1.2 * sin(2*pi*f1*t) + 0.5 * sin(2*pi*f2*t);
noise = 0.2 * randn(n_samples, 1);
signal = clean_signal + noise;

% 2. Anomalous Signal Injection

% a. Sudden spikes (at t=2s and t=7s)
spike_idx = [2000, 7000];
signal(spike_idx) = signal(spike_idx) + [5; -6];
% Label short windows around spikes as anomaly
label(spike_idx(1)-5:spike_idx(1)+5) = 1; 
label(spike_idx(2)-5:spike_idx(2)+5) = 1;

% b. Gradual drift (from t=4s to t=5s)
drift_start = 4000;
drift_end = 5000;
drift = linspace(0, 3, drift_end - drift_start + 1)';
signal(drift_start:drift_end) = signal(drift_start:drift_end) + drift;
label(drift_start:drift_end) = 1;

% c. Frequency shift (from t=8s to t=9s)
shift_start = 8000;
shift_end = 9000;
t_shift = t(shift_start:shift_end);
freq_shift_signal = 1.2 * sin(2*pi*(f1+15)*t_shift); % Shifted drastically setup to 20 Hz
signal(shift_start:shift_end) = freq_shift_signal + noise(shift_start:shift_end);
label(shift_start:shift_end) = 1;

% 3. Plot signals clearly
figure('Name', 'Synthetic Signal Generation', 'Position', [100, 100, 1000, 400]);
plot(t, signal, 'Color', [0 0.447 0.741], 'LineWidth', 1); hold on;

% Highlight anomalies
anomaly_idx = find(label == 1);
scatter(t(anomaly_idx), signal(anomaly_idx), 15, 'r', 'filled');

title('Synthetic Signal Sequence with Injected Anomalies');
xlabel('Time (Seconds)');
ylabel('Signal Amplitude');
legend('Base Signal', 'Anomalous Regions', 'Location', 'best');
grid on;

% 4. Save raw signal to CSV
data_dir = '../../data/raw';
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end
output_file = fullfile(data_dir, 'signals.csv');
T_out = table(t, signal, label, 'VariableNames', {'time', 'value', 'label'});
writetable(T_out, output_file);
fprintf('Module 1: Successfully generated and saved raw signals to %s\n', output_file);
