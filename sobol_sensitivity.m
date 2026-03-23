clear; clc; close all;

%% =========================================================================
%% Sobol 方差敏感度分析
%%
%% 前向模型：双涵道涡扇发动机性能模型 (engine_forward)
%% 参数先验：13 个参数服从独立均匀分布 U(lb_i, ub_i)
%% 输出量：比推力 R_ud [N·s/kg]，比油耗 C_ud [kg/(N·h)]
%% 敏感度：一阶主效应指标 S_i，总效应指标 ST_i
%% 估计方法：Saltelli (2010) / Jansen (1999)
%% =========================================================================

%% =========================================================================
%% 1. 参数定义
%% =========================================================================

param_labels = { ...
    'eta\_k', 'eta\_t', 'eta\_T', 'eta\_m', ...
    'lambda', 'eta\_v', 'eta\_tv', 'eta\_c1', 'eta\_c2', ...
    'sigma\_cc', 'sigma\_kan', 'sigma\_kask', 'sigma\_ks'};

param_labels_latex = { ...
    '$\eta_k$',    '$\eta_t$',    '$\eta_T$',    '$\eta_m$', ...
    '$\lambda$',   '$\eta_v$',    '$\eta_{tv}$', '$\eta_{c1}$', '$\eta_{c2}$', ...
    '$\sigma_{cc}$','$\sigma_{kan}$','$\sigma_{kask}$','$\sigma_{ks}$'};

n_params = 13;

% 先验均匀分布上下界
lb = [0.84, 0.86, 0.97, 0.98, ...
      1.00, 0.84, 0.88, 0.90, 0.90, ...
      0.96, 0.95, 0.96, 0.93];

ub = [0.86, 0.92, 0.99, 0.995, ...
      1.06, 0.90, 0.96, 0.98,  0.98, ...
      1.04, 1.01, 1.02, 0.99];

% 工况参数（与 tmcmc13.m 一致）
cond.T_H      = 288;
cond.M_flight = 0.0;
cond.m        = 10.0;
cond.pi_k     = 33.0;
cond.T_g      = 1700.0;

theta_fixed = [];

%% =========================================================================
%% 2. 采样设置
%% =========================================================================

N = 5000;   % 基础样本数；建议 N >= 10000 以提高收敛精度
            % 总模型调用次数 = N*(n_params+2) ≈ 75000（N=5000）
rng(42);    % 固定随机种子

fprintf('===== Sobol 敏感度分析 =====\n');
fprintf('  基础样本数 N = %d\n', N);
fprintf('  参数维数   k = %d\n', n_params);
fprintf('  总模型调用 = %d\n\n', N*(n_params+2));

% 生成两个独立的拉丁超立方样本矩阵 A, B（N × k）
% 使用拉丁超立方采样提高空间覆盖均匀性
% 若无 Statistics Toolbox，将 lhsdesign 替换为 rand
try
    A_unit = lhsdesign(N, n_params);
    B_unit = lhsdesign(N, n_params);
catch
    warning('lhsdesign 不可用，使用伪随机均匀采样');
    A_unit = rand(N, n_params);
    B_unit = rand(N, n_params);
end

% 将 [0,1] 样本映射到参数实际范围 [lb, ub]
A = bsxfun(@plus, lb, bsxfun(@times, A_unit, ub - lb));
B = bsxfun(@plus, lb, bsxfun(@times, B_unit, ub - lb));

%% =========================================================================
%% 3. 模型批量评估
%% =========================================================================

fprintf('正在评估矩阵 A  (%d 次)...\n', N);
[yA_R, yA_C] = eval_model_batch(A, theta_fixed, cond);

fprintf('正在评估矩阵 B  (%d 次)...\n', N);
[yB_R, yB_C] = eval_model_batch(B, theta_fixed, cond);

% 构造并评估 AB_i 矩阵
%   AB_i: 第 i 列取自 B，其余列取自 A
%   对应"固定参数 i，对其余参数采样"的重新参数化
yABi_R = zeros(N, n_params);
yABi_C = zeros(N, n_params);

for i = 1:n_params
    fprintf('正在评估 AB_%02d/%d...\n', i, n_params);
    AB_i        = A;
    AB_i(:, i)  = B(:, i);
    [yABi_R(:,i), yABi_C(:,i)] = eval_model_batch(AB_i, theta_fixed, cond);
end

fprintf('\n模型评估完毕。\n\n');

%% =========================================================================
%% 4. 计算 Sobol 指标
%% =========================================================================

% 筛除无效样本行（NaN / Inf）
valid_R = isfinite(yA_R) & isfinite(yB_R) & all(isfinite(yABi_R), 2);
valid_C = isfinite(yA_C) & isfinite(yB_C) & all(isfinite(yABi_C), 2);

fprintf('有效样本行（R_ud）：%d / %d\n', sum(valid_R), N);
fprintf('有效样本行（C_ud）：%d / %d\n\n', sum(valid_C), N);

[S_R, ST_R] = compute_sobol_indices( ...
    yA_R(valid_R), yB_R(valid_R), yABi_R(valid_R,:));

[S_C, ST_C] = compute_sobol_indices( ...
    yA_C(valid_C), yB_C(valid_C), yABi_C(valid_C,:));

%% =========================================================================
%% 5. 打印结果
%% =========================================================================

param_names_plain = strrep(param_labels, '\_', '_');

fprintf('%-14s %10s %10s   %10s %10s\n', ...
    '参数', 'S_i(R)', 'ST_i(R)', 'S_i(C)', 'ST_i(C)');
fprintf('%s\n', repmat('-', 1, 60));
for i = 1:n_params
    fprintf('%-14s %10.4f %10.4f   %10.4f %10.4f\n', ...
        param_names_plain{i}, S_R(i), ST_R(i), S_C(i), ST_C(i));
end
fprintf('%s\n', repmat('-', 1, 60));
fprintf('%-14s %10.4f %10.4f   %10.4f %10.4f\n', ...
    '求和', sum(S_R), sum(ST_R), sum(S_C), sum(ST_C));
fprintf('\n');
fprintf('注：一阶指标之和理论上 <= 1；总效应指标之和 >= 1。\n');
fprintf('    差值反映参数间高阶交互作用的贡献比例。\n\n');

%% =========================================================================
%% 6. 可视化
%% =========================================================================

plot_sobol(S_R, ST_R, S_C, ST_C, param_labels, param_labels_latex, n_params);

%% =========================================================================
%%                           辅助函数
%% =========================================================================

%% --- 批量模型评估 ---
function [yR, yC] = eval_model_batch(theta_mat, theta_f, cond)
    N  = size(theta_mat, 1);
    yR = NaN(N, 1);
    yC = NaN(N, 1);
    for j = 1:N
        [y, ~] = engine_forward(theta_mat(j,:), theta_f, cond);
        if numel(y) == 2 && all(isfinite(y))
            yR(j) = y(1);
            yC(j) = y(2);
        end
    end
end

%% --- Sobol 指标估计 ---
%
%  一阶主效应指标（Saltelli 2010）：
%    S_i = [1/N * Σ_j f(B)_j * (f(AB_i)_j - f(A)_j)] / Var(Y)
%
%  总效应指标（Jansen 1999）：
%    ST_i = [1/(2N) * Σ_j (f(A)_j - f(AB_i)_j)^2] / Var(Y)
%
%  总方差由 A 与 B 合并样本估计。
%
function [S, ST] = compute_sobol_indices(yA, yB, yABi)
    N         = length(yA);
    k         = size(yABi, 2);
    Var_total = var([yA; yB], 0);

    if Var_total < 1e-14
        warning('总方差近似为零，指标无法归一化。');
        S  = zeros(1, k);
        ST = zeros(1, k);
        return;
    end

    S  = zeros(1, k);
    ST = zeros(1, k);

    for i = 1:k
        yci   = yABi(:, i);
        S(i)  =  mean(yB .* (yci - yA))      / Var_total;
        ST(i) =  mean((yA - yci).^2) / 2     / Var_total;
    end

    % 裁剪数值误差引起的微小负值
    S  = max(0, S);
    ST = max(0, ST);
end

%% --- 可视化 ---
function plot_sobol(S_R, ST_R, S_C, ST_C, param_labels, param_labels_latex, k)

    col_S  = [0.18, 0.53, 0.78];   % 蓝：主效应
    col_ST = [0.92, 0.33, 0.15];   % 橙红：总效应
    x      = 1:k;
    w      = 0.36;

    %% ---------------------------------------------------------------
    %% 图 1：比推力 R_ud 的 Sobol 指标（分组柱状图）
    %% ---------------------------------------------------------------
    figure('Name', 'R_ud - Sobol 指标', 'Position', [60, 400, 920, 360]);

    bar(x - w/2, S_R,  w, 'FaceColor', col_S,  'DisplayName', '主效应 S_i');
    hold on;
    bar(x + w/2, ST_R, w, 'FaceColor', col_ST, 'DisplayName', '总效应 ST_i');

    set(gca, 'XTick', x, 'XTickLabel', param_labels, ...
        'XTickLabelRotation', 45, 'TickLabelInterpreter', 'latex', 'FontSize', 10);
    ylabel('Sobol 指标值', 'FontSize', 12);
    title('比推力 $R_{ud}$ — Sobol 方差敏感度指标', ...
        'FontSize', 13, 'Interpreter', 'latex');
    legend('Location', 'northeast', 'FontSize', 10);
    ylim([0, max([ST_R(:); 0.05]) * 1.25]);
    grid on; grid minor;

    %% ---------------------------------------------------------------
    %% 图 2：比油耗 C_ud 的 Sobol 指标（分组柱状图）
    %% ---------------------------------------------------------------
    figure('Name', 'C_ud - Sobol 指标', 'Position', [60, 20, 920, 360]);

    bar(x - w/2, S_C,  w, 'FaceColor', col_S,  'DisplayName', '主效应 S_i');
    hold on;
    bar(x + w/2, ST_C, w, 'FaceColor', col_ST, 'DisplayName', '总效应 ST_i');

    set(gca, 'XTick', x, 'XTickLabel', param_labels, ...
        'XTickLabelRotation', 45, 'TickLabelInterpreter', 'latex', 'FontSize', 10);
    ylabel('Sobol 指标值', 'FontSize', 12);
    title('比油耗 $C_{ud}$ — Sobol 方差敏感度指标', ...
        'FontSize', 13, 'Interpreter', 'latex');
    legend('Location', 'northeast', 'FontSize', 10);
    ylim([0, max([ST_C(:); 0.05]) * 1.25]);
    grid on; grid minor;

    %% ---------------------------------------------------------------
    %% 图 3：总效应指标降序排列（水平柱状，双子图）
    %% ---------------------------------------------------------------
    figure('Name', 'Sobol 总效应排序对比', 'Position', [100, 200, 1050, 430]);

    [~, ord_R] = sort(ST_R, 'descend');
    [~, ord_C] = sort(ST_C, 'descend');

    subplot(1, 2, 1);
    idx_R = ord_R(end:-1:1);
    barh(1:k, ST_R(idx_R), 'FaceColor', col_ST, 'DisplayName', '总效应 ST_i');
    hold on;
    barh(1:k, S_R(idx_R),  'FaceColor', col_S,  'DisplayName', '主效应 S_i');
    set(gca, 'YTick', 1:k, 'YTickLabel', param_labels(idx_R), ...
        'TickLabelInterpreter', 'latex', 'FontSize', 9);
    xlabel('Sobol 指标值', 'FontSize', 11);
    title('比推力 $R_{ud}$（按 $ST_i$ 降序）', ...
        'FontSize', 11, 'Interpreter', 'latex');
    legend('Location', 'southeast', 'FontSize', 9);
    grid on;

    subplot(1, 2, 2);
    idx_C = ord_C(end:-1:1);
    barh(1:k, ST_C(idx_C), 'FaceColor', col_ST, 'DisplayName', '总效应 ST_i');
    hold on;
    barh(1:k, S_C(idx_C),  'FaceColor', col_S,  'DisplayName', '主效应 S_i');
    set(gca, 'YTick', 1:k, 'YTickLabel', param_labels(idx_C), ...
        'TickLabelInterpreter', 'latex', 'FontSize', 9);
    xlabel('Sobol 指标值', 'FontSize', 11);
    title('比油耗 $C_{ud}$（按 $ST_i$ 降序）', ...
        'FontSize', 11, 'Interpreter', 'latex');
    legend('Location', 'southeast', 'FontSize', 9);
    grid on;

    sgtitle('各参数 Sobol 方差敏感度指标（总效应降序排列）', ...
        'FontSize', 12, 'FontWeight', 'bold');

    %% ---------------------------------------------------------------
    %% 图 4：雷达图（极坐标蜘蛛网）——直观对比 13 个参数
    %% ---------------------------------------------------------------
    figure('Name', 'Sobol 雷达图', 'Position', [600, 200, 960, 430]);

    angles = linspace(0, 2*pi, k+1);
    angles = angles(1:end-1);

    subplot(1, 2, 1);
    radar_plot(angles, S_R, ST_R, param_labels_latex, ...
        '比推力 $R_{ud}$', col_S, col_ST);

    subplot(1, 2, 2);
    radar_plot(angles, S_C, ST_C, param_labels_latex, ...
        '比油耗 $C_{ud}$', col_S, col_ST);

    sgtitle('Sobol 敏感度指标雷达图', 'FontSize', 12, 'FontWeight', 'bold');
end

%% --- 雷达图辅助函数 ---
function radar_plot(angles, S, ST, labels, ttl, col_S, col_ST)
    k      = length(S);
    % 闭合多边形
    th     = [angles, angles(1)];
    rS     = [S,  S(1)];
    rST    = [ST, ST(1)];

    polarplot(th, rST, '-o', 'Color', col_ST, 'LineWidth', 2.0, ...
        'MarkerFaceColor', col_ST, 'MarkerSize', 5, 'DisplayName', 'ST_i');
    hold on;
    polarplot(th, rS,  '-s', 'Color', col_S,  'LineWidth', 1.8, ...
        'MarkerFaceColor', col_S,  'MarkerSize', 5, 'DisplayName', 'S_i');

    ax = gca;
    ax.ThetaTick     = rad2deg(angles);
    ax.ThetaTickLabel = labels;
    ax.TickLabelInterpreter = 'latex';
    ax.FontSize      = 8;
    ax.RLim          = [0, max([ST(:); 0.1]) * 1.1];
    legend('Location', 'southoutside', 'Orientation', 'horizontal', ...
        'FontSize', 9);
    title(ttl, 'FontSize', 11, 'Interpreter', 'latex');
end

%% =========================================================================
%%                         前向模型（与 tmcmc13.m 一致）
%% =========================================================================

function kT = piecewise_kT(T_g)
    if     T_g > 1600;              kT = 1.25;
    elseif T_g > 1400 && T_g <= 1600; kT = 1.30;
    else;                           kT = 1.33;
    end
end

function RT = piecewise_RT(T_g)
    if     T_g > 1600;              RT = 288.6;
    elseif T_g > 1400 && T_g <= 1600; RT = 288.0;
    else;                           RT = 287.6;
    end
end

function d = delta_cooling(T_g)
    d = max(0.0, min(0.15, 0.02 + (T_g - 1200)/100*0.02));
end

function [y, aux] = engine_forward(theta_s, ~, cond)
    eta_k      = theta_s(1);
    eta_t      = theta_s(2);
    eta_T      = theta_s(3);
    eta_m      = theta_s(4);
    lambda     = theta_s(5);
    eta_v      = theta_s(6);
    eta_tv     = theta_s(7);
    eta_c1     = theta_s(8);
    eta_c2     = theta_s(9);
    sigma_cc   = theta_s(10);
    sigma_kan  = theta_s(11);
    sigma_kask = theta_s(12);
    sigma_ks   = theta_s(13);

    T_H      = cond.T_H;
    M_flight = cond.M_flight;
    m        = cond.m;
    pi_k     = cond.pi_k;
    T_g      = cond.T_g;

    y   = [NaN, NaN];
    aux = struct();

    try
        k_air = 1.4;
        R_air = 287.3;
        a = sqrt(k_air * R_air * T_H);
        if ~isfinite(a) || a <= 0; return; end
        V_flight = a * M_flight;

        kT = piecewise_kT(T_g);
        RT = piecewise_RT(T_g);
        d  = delta_cooling(T_g);

        inner = 1 + V_flight^2 / (2*(k_air/(k_air-1))*R_air*T_H);
        if inner <= 0; return; end
        tau_v = inner^(k_air/(k_air-1));
        T_B   = T_H * inner^k_air;
        if ~isfinite(T_B) || T_B <= 0; return; end

        pi_k_ratio = pi_k^((k_air-1)/k_air);
        if ~isfinite(pi_k_ratio) || pi_k_ratio < 1; return; end
        T_k = T_B * (1 + (pi_k_ratio-1)/eta_k);
        if ~isfinite(T_k) || T_k <= 0; return; end

        g_T = 3e-5*T_g - 2.69e-5*T_k - 0.003;
        if ~isfinite(g_T) || g_T <= 0; return; end

        compress_work = (k_air/(k_air-1))*R_air*T_B*(pi_k_ratio-1);
        gas_enthalpy  = (kT/(kT-1))*RT*T_g;
        if abs(gas_enthalpy) < 1e-6; return; end

        num_lh = 1 - compress_work/(gas_enthalpy*eta_k);
        den_lh = 1 - compress_work/(gas_enthalpy*eta_k*eta_t);
        if abs(den_lh) < 1e-10; return; end
        lambda_heat = num_lh / den_lh;
        if ~isfinite(lambda_heat); return; end

        sigma_bx = sigma_cc * sigma_kan;

        exp_T  = (kT-1)/kT;
        exp_pr = tau_v * sigma_bx * pi_k * sigma_kask * sigma_ks;
        if exp_pr <= 0; return; end
        exp_term = (1.0/exp_pr)^exp_T;
        if ~isfinite(exp_term); return; end

        term1 = (kT/(kT-1))*RT*T_g*(1-exp_term);
        term2 = compress_work / ((1+g_T)*eta_k*eta_T*eta_t*eta_m*(1-d));
        if abs((1+g_T)*eta_k*eta_T*eta_t*eta_m*(1-d)) < 1e-10; return; end

        L_sv = lambda_heat*(term1-term2);
        if ~isfinite(L_sv) || L_sv <= 0; return; end

        num_xpc = 1 + m*V_flight^2/(2*L_sv*eta_tv*eta_v*eta_c2);
        den_xpc = 1 + (m*eta_tv*eta_v*eta_c2)/(eta_c1*lambda);
        if abs(den_xpc) < 1e-10; return; end
        x_pc = num_xpc / den_xpc;
        if ~isfinite(x_pc) || x_pc <= 0; return; end

        sq1 = 2*eta_c1*lambda*x_pc*L_sv;
        if sq1 < 0; return; end
        V_j1 = (1+g_T)*sqrt(sq1) - V_flight;

        sq2 = 2*(1-x_pc)/m*L_sv*eta_tv*eta_v*eta_c2 + V_flight^2;
        if sq2 < 0; return; end
        V_j2 = sqrt(sq2) - V_flight;

        R_ud = (1/(1+m))*V_j1 + (m/(1+m))*V_j2;
        if ~isfinite(R_ud) || R_ud <= 0; return; end

        denom_C = R_ud*(1+m);
        if abs(denom_C) < 1e-10; return; end
        C_ud = 3600*g_T*(1-d)/denom_C;
        if ~isfinite(C_ud) || C_ud <= 0; return; end

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

    catch ME
        warning('engine_forward: %s', ME.message);
    end
end
