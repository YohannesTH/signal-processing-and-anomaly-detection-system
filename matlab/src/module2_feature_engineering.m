% MODULE 2: Feature Engineering
% Loads raw signals, applies FFT and Wavelets, extracts features, saves to CSV.

% 1. Load generated signal CSV
data_dir = '../../data/raw';
input_file = fullfile(data_dir, 'signals.csv');
opts = detectImportOptions(input_file);
T_in = readtable(input_file, opts);

signal = T_in.value;
label = T_in.label;
t = T_in.time;
fs = 1000; % Known sample rate
n_samples = length(signal);

% Windowing parameters for extracting features over time
window_size = 100; % 0.1 second window length
step_size = 50;    % 50% overlap (0.05 seconds step)
n_windows = floor((n_samples - window_size) / step_size) + 1;

% Initialize feature vectors
mean_feat = zeros(n_windows, 1);
var_feat = zeros(n_windows, 1);
energy_feat = zeros(n_windows, 1);
dom_freq_feat = zeros(n_windows, 1);
entropy_feat = zeros(n_windows, 1);
window_label = zeros(n_windows, 1);
window_time = zeros(n_windows, 1);

fprintf('Module 2: Extracting features across %d windows...\n', n_windows);

% 2 & 3. Apply Transforms and Extract Features
for i = 1:n_windows
    idx_start = (i-1)*step_size + 1;
    idx_end = idx_start + window_size - 1;
    sig_win = signal(idx_start:idx_end);
    
    % Basic statistical features
    mean_feat(i) = mean(sig_win);
    var_feat(i) = var(sig_win);
    energy_feat(i) = sum(sig_win.^2); % Signal energy
    
    % Fast Fourier Transform (FFT) - Dominant frequency
    Y = fft(sig_win);
    L = length(sig_win);
    P2 = abs(Y/L);
    P1 = P2(1:floor(L/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = fs*(0:(L/2))/L;
    [~, max_idx] = max(P1);
    dom_freq_feat(i) = f(max_idx);
    
    % Wavelet Transform - Spectral Entropy
    % Using discrete wavelet transform (Daubechies 4) from Wavelet Toolbox
    [~, cd] = dwt(sig_win, 'db4');
    
    % Calculate normalized energy probability of detail coefficients
    p = abs(cd).^2 / sum(abs(cd).^2);
    p = p(p > 0); % Remove exactly zero probabilities to avoid log(0)
    entropy_feat(i) = -sum(p .* log2(p));
    
    % Aggregate Label: if any true anomaly falls within this window
    window_label(i) = any(label(idx_start:idx_end));
    window_time(i) = t(idx_start);
end

% 4. Create feature matrix
features_T = table(window_time, mean_feat, var_feat, energy_feat, dom_freq_feat, entropy_feat, window_label, ...
    'VariableNames', {'time', 'mean', 'variance', 'energy', 'dominant_frequency', 'entropy', 'label'});

% 5. Save output as features.csv
feat_dir = '../../data/features';
if ~exist(feat_dir, 'dir')
    mkdir(feat_dir);
end
output_file = fullfile(feat_dir, 'features.csv');
writetable(features_T, output_file);
fprintf('Module 2: Successfully saved feature matrix to %s\n', output_file);
