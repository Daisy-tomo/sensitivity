%% sobol_sensitivity_engine.m
%  Sobol/Saltelli-based Global Sensitivity Analysis
%  for an aircraft engine thermodynamic cycle model (13 parameters).
%
%  Outputs  : y = [R_ud, C_ud]
%  Estimators: Jansen (1999) for STi; Saltelli (2010) for Si
%  Sampling  : Saltelli scheme — matrices A, B, A_Bi (i = 1..k)
%  Total runs: N * (2*k + 2)
%
%  Requires MATLAB R2019b or later.
%  sobolset (Statistics & Machine Learning Toolbox) is used when available;
%  falls back to pseudo-random uniform sampling otherwise.
%
% -------------------------------------------------------------------------

clear; close all; clc;

%% =========================================================
%  1.  PARAMETER DEFINITIONS  (edit here as needed)
% ==========================================================
param_names = { ...
    'eta_k',    'eta_t',     'eta_m',    'eta_v',   'eta_tv', ...
    'eta_c1',   'eta_c2',    'sigma_cc', 'sigma_kan', ...
    'sigma_kask','sigma_ks', 'eta_T',    'lambda'  };

% Nominal values
theta0 = [0.85,  0.89,  0.988, 0.860, 0.910, ...
          0.945, 0.930, 0.990, 0.985, ...
          0.985, 0.950, 0.980, 1.030];

% Lower bounds  (adjust as needed; keep physically meaningful)
lb     = [0.80,  0.84,  0.970, 0.820, 0.870, ...
          0.920, 0.900, 0.975, 0.970, ...
          0.970, 0.920, 0.960, 0.990];

% Upper bounds  (adjust as needed)
ub     = [0.90,  0.94,  0.999, 0.900, 0.950, ...
          0.970, 0.960, 1.000, 1.000, ...
          1.000, 0.980, 1.000, 1.070];

k = numel(theta0);   % number of parameters (13)

%% =========================================================
%  2.  OPERATING CONDITIONS  (edit here as needed)
% ==========================================================
opcond.T_H      = 288;    % Ambient temperature [K]
opcond.M_flight = 0.0;    % Flight Mach number  [-]
opcond.m        = 10.0;   % Bypass ratio        [-]
opcond.pi_k     = 33.0;   % Compressor pressure ratio [-]
opcond.T_g      = 1700.0; % Turbine inlet temperature [K]

%% =========================================================
%  3.  SOBOL SAMPLING SETTINGS
% ==========================================================
N    = 2000;    % Base sample size  (total calls = N*(2k+2) = N*28)
seed = 42;

fprintf('============================================\n');
fprintf('  Sobol Global Sensitivity Analysis\n');
fprintf('============================================\n');
fprintf('Parameters k = %d,  Base samples N = %d\n', k, N);
fprintf('Total model evaluations: %d\n', N*(2*k+2));
fprintf('\n');

%% =========================================================
%  4.  GENERATE SAMPLE MATRICES  A and B  in [lb, ub]
% ==========================================================
rng(seed, 'twister');

try
    % Sobol quasi-random (requires Statistics & ML Toolbox)
    p  = sobolset(2*k, 'Skip', 1e3, 'Leap', 1e2);
    p  = scramble(p, 'MatousekAffineOwen');
    UV = net(p, N);            % N x 2k, values in [0,1]
    fprintf('Sampling: Sobol quasi-random sequence (sobolset).\n');
catch
    UV = rand(N, 2*k);
    fprintf('Sampling: sobolset unavailable — using pseudo-random uniform.\n');
end

U_A = UV(:, 1:k);          % N x k
U_B = UV(:, k+1:2*k);      % N x k

% Scale [0,1] -> [lb, ub]
A = bsxfun(@plus, lb, bsxfun(@times, U_A, (ub - lb)));  % N x k
B = bsxfun(@plus, lb, bsxfun(@times, U_B, (ub - lb)));  % N x k

%% =========================================================
%  5.  EVALUATE  f(A)  and  f(B)
% ==========================================================
n_out = 2;                    % y = [R_ud, C_ud]
fA    = nan(N, n_out);
fB    = nan(N, n_out);
fail_A = 0;
fail_B = 0;

fprintf('Evaluating f(A) and f(B) ...\n');
for i = 1:N
    try
        [y, ~] = engine_forward(A(i,:), opcond);
        if all(isfinite(y)) && all(y > 0)
            fA(i,:) = y;
        else
            fail_A = fail_A + 1;
        end
    catch
        fail_A = fail_A + 1;
    end

    try
        [y, ~] = engine_forward(B(i,:), opcond);
        if all(isfinite(y)) && all(y > 0)
            fB(i,:) = y;
        else
            fail_B = fail_B + 1;
        end
    catch
        fail_B = fail_B + 1;
    end
end

fprintf('  f(A) failures : %d / %d  (%.1f%%)\n', fail_A, N, 100*fail_A/N);
fprintf('  f(B) failures : %d / %d  (%.1f%%)\n', fail_B, N, 100*fail_B/N);

%% =========================================================
%  6.  EVALUATE  f(A_Bi)  for each parameter i
%      A_Bi = copy of A with column i replaced by B(:,i)
% ==========================================================
fABi     = nan(N, n_out, k);   % N x n_out x k
fail_ABi = zeros(1, k);

fprintf('\nEvaluating f(A_Bi) for %d parameters ...\n', k);
for j = 1:k
    ABi        = A;
    ABi(:, j)  = B(:, j);

    for i = 1:N
        try
            [y, ~] = engine_forward(ABi(i,:), opcond);
            if all(isfinite(y)) && all(y > 0)
                fABi(i,:,j) = y;
            else
                fail_ABi(j) = fail_ABi(j) + 1;
            end
        catch
            fail_ABi(j) = fail_ABi(j) + 1;
        end
    end

    if mod(j,3) == 0 || j == k
        fprintf('  param %2d/%d  %-12s  failures: %d\n', ...
                j, k, param_names{j}, fail_ABi(j));
    end
end

%% =========================================================
%  7.  SALTELLI / JANSEN ESTIMATORS
%
%  Var(Y)   : variance over combined [fA; fB] (valid rows only)
%
%  STi (Jansen 1999):
%    STi = mean( (fA - fABi).^2 ) / ( 2 * Var(Y) )
%
%  Si  (Saltelli 2010, eq. 21):
%    Si  = mean( fB .* (fABi - fA) ) / Var(Y)
% ==========================================================
Si  = nan(k, n_out);
STi = nan(k, n_out);

valid_AB = all(isfinite(fA), 2) & all(isfinite(fB), 2);  % N x 1

for o = 1:n_out
    fA_o  = fA(:, o);
    fB_o  = fB(:, o);

    % Variance from combined valid A+B samples
    combined = [fA_o(valid_AB); fB_o(valid_AB)];
    if numel(combined) < 20
        warning('Output %d: too few valid samples (%d).', o, numel(combined));
        continue;
    end
    VarY = var(combined);
    if VarY < 1e-20
        warning('Output %d: near-zero variance (%.3e), skipping.', o, VarY);
        continue;
    end

    for j = 1:k
        fABi_o = fABi(:, o, j);

        % Rows where all three evaluations are valid
        valid = valid_AB & isfinite(fABi_o);
        N_eff = sum(valid);
        if N_eff < 20
            warning('Output %d, param %d: only %d valid rows.', o, j, N_eff);
            continue;
        end

        fa  = fA_o(valid);
        fb  = fB_o(valid);
        fab = fABi_o(valid);

        % Jansen estimator — total-order
        STi(j, o) = mean((fa - fab).^2) / (2 * VarY);

        % Jansen estimator — first-order (more stable than Saltelli 2010;
        % avoids spurious negative values for near-zero Si)
        % Si = 1 - E[(fB - fABi)^2] / (2*Var(Y))
        Si(j, o)  = 1 - mean((fb - fab).^2) / (2 * VarY);
    end
end

%% =========================================================
%  8.  DISPLAY RESULTS TABLE  (sorted by STi, descending)
% ==========================================================
out_labels = {'R_ud', 'C_ud'};

fprintf('\n');
fprintf('============================================================\n');
fprintf('  Sobol Sensitivity Indices  (sorted by STi descending)\n');
fprintf('============================================================\n');

for o = 1:n_out
    fprintf('\n--- Output: %s ---\n', out_labels{o});
    fprintf('  %-14s   %10s   %10s\n', 'Parameter', 'Si', 'STi');
    fprintf('  %s\n', repmat('-', 1, 40));

    [STi_sorted, idx] = sort(STi(:, o), 'descend');
    Si_sorted = Si(idx, o);

    for j = 1:k
        si_str  = format_val(Si_sorted(j));
        sti_str = format_val(STi_sorted(j));
        fprintf('  %-14s   %10s   %10s\n', param_names{idx(j)}, si_str, sti_str);
    end
end

%% =========================================================
%  9.  FAILURE STATISTICS SUMMARY
% ==========================================================
fprintf('\n--- Failure counts ---\n');
fprintf('  f(A)  : %d,  f(B) : %d\n', fail_A, fail_B);
for j = 1:k
    if fail_ABi(j) > 0
        fprintf('  f(A_B%d) [%-12s]: %d\n', j, param_names{j}, fail_ABi(j));
    end
end
total_fail = fail_A + fail_B + sum(fail_ABi);
total_runs = N * (2*k + 2);
fprintf('  Total failures: %d / %d  (%.2f%%)\n', ...
        total_fail, total_runs, 100*total_fail/total_runs);

%% =========================================================
%  10.  BAR CHARTS  Si and STi  for each output
% ==========================================================
colors_Si  = [0.22 0.51 0.80];   % blue
colors_STi = [0.88 0.30 0.18];   % red-orange

for o = 1:n_out
    [~, idx] = sort(STi(:, o), 'descend');

    Si_p  = Si(idx, o);
    STi_p = STi(idx, o);

    % Replace NaN with 0 for plotting
    Si_p(isnan(Si_p))   = 0;
    STi_p(isnan(STi_p)) = 0;

    pnames = param_names(idx);
    x_pos  = 1:k;
    bw     = 0.35;   % bar width

    figure('Name', ['Sensitivity: ' out_labels{o}], ...
           'NumberTitle', 'off', ...
           'Position', [80 + (o-1)*680, 80, 680, 450]);

    b1 = bar(x_pos - bw/2, Si_p,  bw, 'FaceColor', colors_Si,  'EdgeColor', 'none');
    hold on;
    b2 = bar(x_pos + bw/2, STi_p, bw, 'FaceColor', colors_STi, 'EdgeColor', 'none');
    hold off;

    % Axis labels and formatting
    ax = gca;
    set(ax, 'XTick', x_pos, 'XTickLabel', pnames, ...
            'XTickLabelRotation', 40, 'FontSize', 10, ...
            'TickLabelInterpreter', 'none');
    ylabel('Sensitivity index', 'FontSize', 11);
    xlabel('Parameter',         'FontSize', 11);
    title(['Sobol sensitivity indices — ' out_labels{o}], 'FontSize', 12);

    legend([b1, b2], {'S_i  (first-order)', 'ST_i  (total-order)'}, ...
           'Location', 'northeast', 'FontSize', 10);

    y_top  = max([STi_p; 0.05]);
    y_bot  = min([Si_p; STi_p; 0]);          % show negatives if they exist
    ylim([y_bot - 0.01, y_top * 1.18]);
    grid on;  box on;

    % Annotate bar heights (skip very small values)
    thresh = 0.005;
    for j = 1:k
        if abs(Si_p(j)) > thresh
            text(j - bw/2, Si_p(j) + y_top*0.01, ...
                 sprintf('%.3f', Si_p(j)), ...
                 'HorizontalAlignment', 'center', ...
                 'FontSize', 7, 'Color', colors_Si*0.65);
        end
        if abs(STi_p(j)) > thresh
            text(j + bw/2, STi_p(j) + y_top*0.01, ...
                 sprintf('%.3f', STi_p(j)), ...
                 'HorizontalAlignment', 'center', ...
                 'FontSize', 7, 'Color', colors_STi*0.75);
        end
    end
end

fprintf('\n=== Analysis complete ===\n');

%% =========================================================
%  11.  PARAMETER IDENTIFIABILITY ANALYSIS
%
%  Step A — Jacobian at theta0 (central finite differences)
%            J  [n_out x k],  J_ij = dy_i/dtheta_j
%
%  Step B — Normalised (relative) Jacobian
%            Jn_ij = (theta0_j / y0_i) * J_ij    (dimensionless)
%
%  Step C — Fisher Information Matrix (FIM)
%            F = J' * diag(1./sigma_obs.^2) * J
%            Assumes independent Gaussian observation noise.
%            Two noise scenarios are swept: 1 % and 3 % of y0.
%
%  Step D — Metrics derived from FIM
%            * Condition number kappa(F)
%            * Cramer-Rao lower bound: std_CR_i = sqrt( (F^-1)_ii )
%            * Parameter correlation matrix C = D^-1 * F^-1 * D^-1
%              where D = diag(sqrt(diag(F^-1)))
%            * Collinearity index gamma_i: how much param i can be
%              mimicked by linear combination of the others
%              gamma_i = 1/sqrt(min eigenvalue of Jn'*Jn with col i removed)
%              (large gamma_i => near-collinear => hard to identify alone)
%
%  Step E — Visualisation
%            * Bar chart: normalised Cramer-Rao bounds per noise level
%            * Heatmap: parameter correlation matrix
% ==========================================================

fprintf('\n============================================\n');
fprintf('  Identifiability Analysis (FIM)\n');
fprintf('============================================\n');

%% -- A. Evaluate model at nominal --
[y0_raw, ~] = engine_forward(theta0, opcond);
if ~all(isfinite(y0_raw)) || ~all(y0_raw > 0)
    error('engine_forward failed at nominal theta0 — check model inputs.');
end
y0 = y0_raw(:);   % column vector [R_ud; C_ud]

%% -- B. Jacobian via central finite differences --
fd_step = 1e-4;   % relative step size
J = zeros(n_out, k);

for j = 1:k
    theta_fwd = theta0;  theta_fwd(j) = theta0(j) * (1 + fd_step);
    theta_bwd = theta0;  theta_bwd(j) = theta0(j) * (1 - fd_step);

    try
        [yf, ~] = engine_forward(theta_fwd, opcond);
        [yb, ~] = engine_forward(theta_bwd, opcond);
        if all(isfinite(yf)) && all(isfinite(yb))
            J(:, j) = (yf(:) - yb(:)) / (2 * theta0(j) * fd_step);
        else
            warning('FD step failed for parameter %d (%s)', j, param_names{j});
        end
    catch
        warning('FD step exception for parameter %d (%s)', j, param_names{j});
    end
end

% Normalised (relative/dimensionless) Jacobian
Jn = bsxfun(@rdivide, bsxfun(@times, J, theta0), y0);  % n_out x k

fprintf('\nNominal outputs:  R_ud = %.4f m/s,  C_ud = %.6f kg/(N·h)\n', ...
        y0(1), y0(2));
fprintf('\nNormalised Jacobian  Jn_ij = (theta_j/y_i)*dy_i/dtheta_j:\n');
fprintf('  %-14s  %10s  %10s\n', 'Parameter', 'dR_ud', 'dC_ud');
fprintf('  %s\n', repmat('-', 1, 38));
for j = 1:k
    fprintf('  %-14s  %10.4f  %10.4f\n', param_names{j}, Jn(1,j), Jn(2,j));
end

%% -- C & D. FIM metrics for each noise scenario --
noise_levels = [0.01, 0.03];   % 1 % and 3 % of y0
noise_labels = {'1%', '3%'};
n_noise      = numel(noise_levels);

CR_bounds  = nan(k, n_noise);   % Cramer-Rao std dev per parameter, per noise
kappa_F    = nan(1, n_noise);   % condition numbers
Corr_all   = cell(1, n_noise);  % correlation matrices

for ni = 1:n_noise
    sigma_obs = noise_levels(ni) * y0;          % absolute noise per output
    W         = diag(1 ./ sigma_obs.^2);        % n_out x n_out weight matrix
    F         = J' * W * J;                     % k x k FIM

    kappa_F(ni) = cond(F);

    % Invert FIM (use pinv for near-singular cases)
    tol_pinv = max(size(F)) * eps(norm(F));
    Finv     = pinv(F, tol_pinv);

    % Cramer-Rao lower bounds
    cr_var = diag(Finv);
    cr_var(cr_var < 0) = NaN;           % negative => parameter not identified
    CR_bounds(:, ni) = sqrt(cr_var);

    % Normalise CR bound relative to theta0 (as %)
    CR_rel = CR_bounds(:, ni) ./ theta0(:) * 100;

    fprintf('\n--- Noise scenario: %s of y0  (sigma_R=%.3f, sigma_C=%.5f) ---\n', ...
            noise_labels{ni}, sigma_obs(1), sigma_obs(2));
    fprintf('  Condition number of FIM: %.3e\n', kappa_F(ni));
    fprintf('  %-14s  %12s  %12s\n', 'Parameter', 'CR std dev', 'CR rel (%)');
    fprintf('  %s\n', repmat('-', 1, 44));

    % Sort by CR_rel ascending (best identified first)
    [cr_sorted, idx_s] = sort(CR_rel, 'ascend');
    for j = 1:k
        if isnan(cr_sorted(j))
            flag = '  [NOT IDENTIFIABLE]';
        elseif cr_sorted(j) < 5
            flag = '  *** well identified';
        elseif cr_sorted(j) < 20
            flag = '  ** marginally identified';
        else
            flag = '  *  poorly identified';
        end
        fprintf('  %-14s  %12.4f  %11.2f%%%s\n', ...
                param_names{idx_s(j)}, CR_bounds(idx_s(j),ni), cr_sorted(j), flag);
    end

    % Parameter correlation matrix from FIM inverse
    D_inv   = diag(1 ./ sqrt(max(diag(Finv), eps)));
    Corr    = D_inv * Finv * D_inv;
    Corr_all{ni} = Corr;
end

%% -- D.  Collinearity index (based on normalised Jacobian) --
fprintf('\n--- Collinearity index (normalised Jacobian, gamma_i) ---\n');
fprintf('  gamma_i > 15 : near-collinear, Bayesian prior strongly needed\n');
fprintf('  %-14s  %12s\n', 'Parameter', 'gamma_i');
fprintf('  %s\n', repmat('-', 1, 30));

gamma = nan(1, k);
for j = 1:k
    cols_excl = setdiff(1:k, j);
    Jn_sub    = Jn(:, cols_excl);
    ev        = eig(Jn_sub' * Jn_sub);
    ev_min    = min(ev(ev > 0));
    if ~isempty(ev_min) && isfinite(ev_min) && ev_min > 0
        gamma(j) = 1 / sqrt(ev_min);
    end
end
[gamma_s, idx_g] = sort(gamma, 'descend');
for j = 1:k
    if gamma_s(j) > 15
        flag = '  *** near-collinear';
    elseif gamma_s(j) > 5
        flag = '  **  moderate';
    else
        flag = '';
    end
    fprintf('  %-14s  %12.2f%s\n', param_names{idx_g(j)}, gamma_s(j), flag);
end

%% -- E.  Visualisation --

% E1. Cramer-Rao relative bounds (sorted by 1% noise scenario)
[~, idx_cr] = sort(CR_bounds(:,1) ./ theta0(:), 'ascend');
cr_plot = (CR_bounds(idx_cr, :) ./ theta0(idx_cr)' * 100);   % in %  (theta0 transposed to k×1 for broadcast)
cr_plot(cr_plot > 200) = 200;    % cap display at 200%

figure('Name', 'Identifiability: Cramer-Rao bounds', ...
       'NumberTitle', 'off', 'Position', [80, 600, 700, 400]);

bw2   = 0.35;
x_cr  = 1:k;
c_n1  = [0.20 0.63 0.37];
c_n2  = [0.93 0.65 0.13];

b_n1 = bar(x_cr - bw2/2, cr_plot(:,1), bw2, 'FaceColor', c_n1, 'EdgeColor', 'none');
hold on;
b_n2 = bar(x_cr + bw2/2, cr_plot(:,2), bw2, 'FaceColor', c_n2, 'EdgeColor', 'none');
yline(5,  '--', '5% threshold',  'Color', [0.3 0.3 0.8], 'LineWidth', 1.2);
yline(20, '--', '20% threshold', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.2);
hold off;

set(gca, 'XTick', x_cr, 'XTickLabel', param_names(idx_cr), ...
         'XTickLabelRotation', 40, 'FontSize', 10, ...
         'TickLabelInterpreter', 'none');
ylabel('Relative CR bound  (%)', 'FontSize', 11);
xlabel('Parameter (sorted by identifiability)', 'FontSize', 11);
title('Cramer-Rao lower bounds — relative to \theta_0', 'FontSize', 12);
legend([b_n1, b_n2], {'noise 1%', 'noise 3%'}, ...
       'Location', 'northwest', 'FontSize', 10);
ylim([0, min(max(cr_plot(:))*1.2, 210)]);
grid on;  box on;

% E2. Parameter correlation matrix (1% noise, sorted same as CR chart)
Corr_plot = Corr_all{1}(idx_cr, idx_cr);
pnames_sorted = param_names(idx_cr);

figure('Name', 'Identifiability: Correlation matrix', ...
       'NumberTitle', 'off', 'Position', [800, 600, 560, 480]);

imagesc(abs(Corr_plot));
colormap(flipud(hot));
clim([0, 1]);
cb = colorbar;
cb.Label.String = '|Correlation|';
cb.FontSize = 10;

set(gca, 'XTick', 1:k, 'XTickLabel', pnames_sorted, ...
         'XTickLabelRotation', 40, 'FontSize', 9, ...
         'YTick', 1:k, 'YTickLabel', pnames_sorted, ...
         'TickLabelInterpreter', 'none');
title('Parameter correlation matrix  |C_{ij}|  (noise 1%)', 'FontSize', 12);

% Annotate cells with value
for r = 1:k
    for c = 1:k
        v = abs(Corr_plot(r,c));
        if v > 0.3
            text(c, r, sprintf('%.2f', v), ...
                 'HorizontalAlignment', 'center', ...
                 'FontSize', 7, ...
                 'Color', ifelse_color(v));
        end
    end
end

fprintf('\n=== Identifiability analysis complete ===\n');

%% -- Summary recommendation table --
fprintf('\n============================================================\n');
fprintf('  Parameter Selection Summary for Bayesian Inversion\n');
fprintf('============================================================\n');
fprintf('  Criteria: STi(R_ud or C_ud) > 0.05  AND  CR_rel(1%%) < 20%%\n\n');

STi_max = max(STi, [], 2);   % max STi across both outputs
CR_rel1 = CR_bounds(:,1) ./ theta0(:) * 100;

fprintf('  %-14s  %8s  %10s  %10s  %s\n', ...
        'Parameter', 'STi_max', 'CR_rel_1%', 'Collinearity', 'Verdict');
fprintf('  %s\n', repmat('-', 1, 62));

for j = 1:k
    sti_j = STi_max(j);
    cr_j  = CR_rel1(j);
    gam_j = gamma(j);

    if sti_j >= 0.05 && cr_j < 20
        verdict = 'INCLUDE';
    elseif sti_j >= 0.02 && cr_j < 50
        verdict = 'consider';
    else
        verdict = 'fix at prior';
    end

    fprintf('  %-14s  %8.3f  %10.1f  %10.2f  %s\n', ...
            param_names{j}, sti_j, cr_j, gam_j, verdict);
end


%% =========================================================
%%  LOCAL HELPER  (not a model function)
% ==========================================================

function c = ifelse_color(v)
% White text for dark cells, black for light cells
if v > 0.6
    c = [1 1 1];
else
    c = [0 0 0];
end
end
function s = format_val(v)
% Return formatted string; 'NaN' if not finite.
if isfinite(v)
    s = sprintf('%10.4f', v);
else
    s = '       NaN';
end
end


%% =========================================================
%%  LOCAL FUNCTIONS — ENGINE FORWARD MODEL
%   (Do NOT modify the physics below)
% ==========================================================

function [y, aux] = engine_forward(theta, cond)
%ENGINE_FORWARD  Aircraft engine thermodynamic cycle model.
%
%  theta(1)  eta_k     compressor isentropic efficiency
%  theta(2)  eta_t     turbine   isentropic efficiency
%  theta(3)  eta_m     mechanical efficiency
%  theta(4)  eta_v     nozzle velocity coefficient (cold)
%  theta(5)  eta_tv    nozzle velocity coefficient (hot)
%  theta(6)  eta_c1    inner-duct nozzle efficiency
%  theta(7)  eta_c2    outer-duct nozzle efficiency
%  theta(8)  sigma_cc  combustor total-pressure recovery
%  theta(9)  sigma_kan inlet total-pressure recovery
%  theta(10) sigma_kask inter-turbine duct pressure recovery
%  theta(11) sigma_ks  nozzle pressure recovery
%  theta(12) eta_T     turbine cooling effectiveness
%  theta(13) lambda    nozzle area ratio

eta_k     = theta(1);   eta_t     = theta(2);   eta_m     = theta(3);
eta_v     = theta(4);   eta_tv    = theta(5);
eta_c1    = theta(6);   eta_c2    = theta(7);
sigma_cc  = theta(8);   sigma_kan = theta(9);
sigma_kask= theta(10);  sigma_ks  = theta(11);
eta_T     = theta(12);  lambda    = theta(13);

T_H      = cond.T_H;
M_flight = cond.M_flight;
m        = cond.m;
pi_k     = cond.pi_k;
T_g      = cond.T_g;

y   = [NaN, NaN];
aux = struct();

k_air = 1.4;   R_air = 287.3;

% --- Speed of sound and flight velocity ---
a = sqrt(k_air * R_air * T_H);
if ~isfinite(a) || a <= 0, return; end
V_flight = a * M_flight;

% --- Gas property tables ---
kT = piecewise_kT(T_g);
RT = piecewise_RT(T_g);
d  = delta_cooling(T_g);

% --- Ram pressure/temperature ---
inner = 1 + V_flight^2 / (2 * (k_air/(k_air-1)) * R_air * T_H);
if inner <= 0, return; end
tau_v = inner^(k_air / (k_air - 1));
T_B   = T_H * (inner^k_air);
if ~isfinite(T_B) || T_B <= 0, return; end

% --- Compressor exit temperature ---
pi_k_ratio = pi_k^((k_air-1)/k_air);
if ~isfinite(pi_k_ratio) || pi_k_ratio < 1, return; end
T_k = T_B * (1 + (pi_k_ratio - 1) / eta_k);
if ~isfinite(T_k) || T_k <= 0, return; end

% --- Cooling air mass fraction ---
g_T = 3e-5 * T_g - 2.69e-5 * T_k - 0.003;
if ~isfinite(g_T) || g_T <= 0, return; end

% --- Heat-release ratio (lambda_heat) ---
compress_work = (k_air/(k_air-1)) * R_air * T_B * (pi_k_ratio - 1);
gas_enthalpy  = (kT/(kT-1)) * RT * T_g;
if abs(gas_enthalpy) < 1e-6, return; end

num_lambda = 1 - compress_work / (gas_enthalpy * eta_k);
den_lambda = 1 - compress_work / (gas_enthalpy * eta_k * eta_t);
if abs(den_lambda) < 1e-10, return; end
lambda_heat = num_lambda / den_lambda;
if ~isfinite(lambda_heat), return; end

% --- Turbine expansion ---
sigma_bx           = sigma_cc * sigma_kan;
exp_T              = (kT - 1) / kT;
expansion_pr_denom = tau_v * sigma_bx * pi_k * sigma_kask * sigma_ks;
if expansion_pr_denom <= 0, return; end
expansion_term = (1.0 / expansion_pr_denom)^exp_T;
if ~isfinite(expansion_term), return; end

term1 = (kT / (kT - 1)) * RT * T_g * (1 - expansion_term);

% Re-evaluate compressor work (same formula, explicit for clarity)
compress_work = (k_air / (k_air - 1)) * R_air * T_B * (pi_k^((k_air - 1) / k_air) - 1);

denom2 = (1 + g_T) * eta_k * eta_T * eta_t * eta_m * (1 - d);
if abs(denom2) < 1e-10, return; end
term2 = compress_work / denom2;

L_sv = lambda_heat * (term1 - term2);
if ~isfinite(L_sv) || L_sv <= 0, return; end

% --- Nozzle area split x_pc ---
V2_term = m * V_flight^2;
num_xpc = 1 + V2_term / (2 * L_sv * eta_tv * eta_v * eta_c2);
den_xpc = 1 + (m * eta_tv * eta_v * eta_c2) / (eta_c1 * lambda);
if abs(den_xpc) < 1e-10, return; end
x_pc = num_xpc / den_xpc;
if ~isfinite(x_pc) || x_pc <= 0 || x_pc >= 1
    if x_pc <= 0, return; end
end

% --- Jet velocities ---
inner_sq1 = 2 * eta_c1 * lambda * x_pc * L_sv;
if inner_sq1 < 0, return; end
V_j1 = (1 + g_T) * sqrt(inner_sq1) - V_flight;

inner_sq2 = 2 * (1 - x_pc) / m * L_sv * eta_tv * eta_v * eta_c2 + V_flight^2;
if inner_sq2 < 0, return; end
V_j2 = sqrt(inner_sq2) - V_flight;

% --- Performance outputs ---
R_ud = (1/(1+m)) * V_j1 + (m/(1+m)) * V_j2;
if ~isfinite(R_ud) || R_ud <= 0, return; end

denom_C = R_ud * (1 + m);
if abs(denom_C) < 1e-10, return; end
C_ud = 3600 * g_T * (1 - d) / denom_C;
if ~isfinite(C_ud) || C_ud <= 0, return; end

y = [R_ud, C_ud];

aux.T_B         = T_B;
aux.T_k         = T_k;
aux.tau_v       = tau_v;
aux.g_T         = g_T;
aux.lambda_heat = lambda_heat;
aux.sigma_bx    = sigma_bx;
aux.L_sv        = L_sv;
aux.x_pc        = x_pc;
aux.kT          = kT;
aux.RT          = RT;
aux.delta       = d;
end % engine_forward


%% =========================================================
%%  GAS PROPERTY AND COOLING FUNCTIONS
% ==========================================================

function kT = piecewise_kT(T_g)
if     T_g > 800  && T_g <= 1400,  kT = 1.33;
elseif T_g > 1400 && T_g <= 1600,  kT = 1.30;
elseif T_g > 1600,                  kT = 1.25;
else,                                kT = 1.33;
end
end

function RT = piecewise_RT(T_g)
if     T_g > 800  && T_g <= 1400,  RT = 287.6;
elseif T_g > 1400 && T_g <= 1600,  RT = 288.0;
elseif T_g > 1600,                  RT = 288.6;
else,                                RT = 287.6;
end
end

function d = delta_cooling(T_g)
d = 0.02 + (T_g - 1200) / 100 * 0.02;
d = max(0.0, min(d, 0.15));
end
