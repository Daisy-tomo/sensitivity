
clear; close all; clc;

%% =========================================================
%  1. 参数定义（如有需要可在此处修改）
% ==========================================================
param_names = { ...
    'eta_k',    'eta_t',     'eta_m',    'eta_v',   'eta_tv', ...
    'eta_c1',   'eta_c2',    'sigma_cc', 'sigma_kan', ...
    'sigma_kask','sigma_ks', 'eta_T',    'lambda'  };

% 名义值
theta0 = [0.85,  0.89,  0.988, 0.860, 0.910, ...
          0.945, 0.930, 0.990, 0.985, ...
          0.985, 0.950, 0.980, 1.030];

% 下限（根据图片范围更新）
lb     = [0.84,  0.86,  0.980, 0.850, 0.900, ...
          0.940, 0.920, 0.980, 0.980, ...
          0.980, 0.940, 0.970, 1.020];

% 上限（根据图片范围更新）
ub     = [0.86,  0.92,  0.995, 0.870, 0.920, ...
          0.950, 0.940, 1.000, 0.990, ...
          0.990, 0.960, 0.990, 1.040];

k = numel(theta0);   % 参数个数（13）

%% =========================================================
%  2. 工况条件（如有需要可在此处修改）
% ==========================================================
cond.T_H      = 288;    % 环境温度 [K]
cond.M_flight = 0.0;    % 飞行马赫数  [-]
cond.m        = 10.0;   % 旁通比        [-]
cond.pi_k     = 33.0;   % 压气机压比 [-]
cond.T_g      = 1700.0; % 涡轮进口温度 [K]

%% =========================================================
%  3. SOBOL 采样设置
% ==========================================================
N    = 10000;    % 基础样本数（总调用次数 = N*(2k+2) = N*28）
% seed = 325;

fprintf('============================================\n');
fprintf('  Sobol 全局敏感性分析\n');
fprintf('============================================\n');
fprintf('参数个数 k = %d，基础样本数 N = %d\n', k, N);
fprintf('模型总评估次数：%d\n', N*(2*k+2));
fprintf('\n');

%% =========================================================
%  4. 在 [lb, ub] 内生成采样矩阵 A 和 B
% ==========================================================
% rng(seed, 'twister');

try
    % Sobol 准随机采样（需要 Statistics & ML Toolbox）
    p  = sobolset(2*k, 'Skip', 1e3, 'Leap', 1e2);
    p  = scramble(p, 'MatousekAffineOwen');
    UV = net(p, N);            % N x 2k，取值范围在 [0,1]
    fprintf('采样方式：Sobol 准随机序列（sobolset）。\n');
catch
    UV = rand(N, 2*k);
    fprintf('采样方式：sobolset 不可用，改用伪随机均匀采样。\n');
end

U_A = UV(:, 1:k);          % N x k
U_B = UV(:, k+1:2*k);      % N x k

% 将 [0,1] 缩放到 [lb, ub]
A = bsxfun(@plus, lb, bsxfun(@times, U_A, (ub - lb)));  % N x k
B = bsxfun(@plus, lb, bsxfun(@times, U_B, (ub - lb)));  % N x k

%% =========================================================
%  5. 计算 f(A) 和 f(B)
% ==========================================================
n_out = 2;                    % y = [R_ud, C_ud]
fA    = nan(N, n_out);
fB    = nan(N, n_out);
fail_A = 0;
fail_B = 0;

fprintf('正在计算 f(A) 和 f(B) ...\n');
for i = 1:N
    try
        [y, ~] = engine_forward(A(i,:), cond);
        if all(isfinite(y)) && all(y > 0)
            fA(i,:) = y;
        else
            fail_A = fail_A + 1;
        end
    catch
        fail_A = fail_A + 1;
    end

    try
        [y, ~] = engine_forward(B(i,:), cond);
        if all(isfinite(y)) && all(y > 0)
            fB(i,:) = y;
        else
            fail_B = fail_B + 1;
        end
    catch
        fail_B = fail_B + 1;
    end
end

fprintf('  f(A) 失败次数：%d / %d  (%.1f%%)\n', fail_A, N, 100*fail_A/N);
fprintf('  f(B) 失败次数：%d / %d  (%.1f%%)\n', fail_B, N, 100*fail_B/N);

%% =========================================================
%  6. 计算每个参数 i 对应的 f(A_Bi)
%      A_Bi = A 的副本，但第 i 列替换为 B(:,i)
% ==========================================================
fABi     = nan(N, n_out, k);   % N x n_out x k
fail_ABi = zeros(1, k);

fprintf('\n正在对 %d 个参数计算 f(A_Bi) ...\n', k);
for j = 1:k
    ABi        = A;
    ABi(:, j)  = B(:, j);

    for i = 1:N
        try
            [y, ~] = engine_forward(ABi(i,:), cond);
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
        fprintf('  参数 %2d/%d  %-12s  失败次数：%d\n', ...
                j, k, param_names{j}, fail_ABi(j));
    end
end

%% =========================================================
%  7. SALTELLI / JANSEN 估计器
%
%  Var(Y)   : 在合并后的 [fA; fB] 上、仅使用有效样本计算方差
%
%  STi（Jansen 1999）:
%    STi = mean( (fA - fABi).^2 ) / ( 2 * Var(Y) )
%
%  Si（Saltelli 2010，公式 21）:
%    Si  = mean( fB .* (fABi - fA) ) / Var(Y)
% ==========================================================
Si  = nan(k, n_out);
STi = nan(k, n_out);

valid_AB = all(isfinite(fA), 2) & all(isfinite(fB), 2);  % N x 1

for o = 1:n_out
    fA_o  = fA(:, o);
    fB_o  = fB(:, o);

    % 使用有效的 A+B 样本计算合并方差
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

        % 仅保留三次计算都有效的行
        valid = valid_AB & isfinite(fABi_o);
        N_eff = sum(valid);
        if N_eff < 20
            warning('Output %d, param %d: only %d valid rows.', o, j, N_eff);
            continue;
        end

        fa  = fA_o(valid);
        fb  = fB_o(valid);
        fab = fABi_o(valid);

        % Jansen 估计器——总效应指标
        STi(j, o) = mean((fa - fab).^2) / (2 * VarY);

        % Jansen 估计器——一阶指标（比 Saltelli 2010 更稳定；
        % 可减少 Si 接近 0 时出现的伪负值）
        % Si = 1 - E[(fB - fABi)^2] / (2*Var(Y))
        Si(j, o)  = 1 - mean((fb - fab).^2) / (2 * VarY);
    end
end

%% =========================================================
%  8. 显示结果表（按 STi 降序排序）
% ==========================================================
out_labels = {'R_ud', 'C_ud'};

fprintf('\n');
fprintf('============================================================\n');
fprintf('  Sobol 敏感性指标（按 STi 降序排序）\n');
fprintf('============================================================\n');

for o = 1:n_out
    fprintf('\n--- 输出：%s ---\n', out_labels{o});
    fprintf('  %-14s   %10s   %10s\n', '参数', 'Si', 'STi');
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
%  9. 失效统计汇总
% ==========================================================
fprintf('\n--- 失败次数统计 ---\n');
fprintf('  f(A)  : %d,  f(B) : %d\n', fail_A, fail_B);
for j = 1:k
    if fail_ABi(j) > 0
        fprintf('  f(A_B%d) [%-12s]: %d\n', j, param_names{j}, fail_ABi(j));
    end
end
total_fail = fail_A + fail_B + sum(fail_ABi);
total_runs = N * (2*k + 2);
fprintf('  总失败次数：%d / %d  (%.2f%%)\n', ...
        total_fail, total_runs, 100*total_fail/total_runs);

%% =========================================================
%  10. 为每个输出绘制 Si 和 STi 柱状图
% ==========================================================
colors_Si  = [0.22 0.51 0.80];   % 蓝色
colors_STi = [0.88 0.30 0.18];   % 红橙色

for o = 1:n_out
    [~, idx] = sort(STi(:, o), 'descend');

    Si_p  = Si(idx, o);
    STi_p = STi(idx, o);

    % 绘图时将 NaN 替换为 0
    Si_p(isnan(Si_p))   = 0;
    STi_p(isnan(STi_p)) = 0;

    pnames = param_names(idx);
    x_pos  = 1:k;
    bw     = 0.35;   % 柱宽

    figure('Name', ['敏感性分析：' out_labels{o}], ...
           'NumberTitle', 'off', ...
           'Position', [80 + (o-1)*680, 80, 680, 450]);

    b1 = bar(x_pos - bw/2, Si_p,  bw, 'FaceColor', colors_Si,  'EdgeColor', 'none');
    hold on;
    b2 = bar(x_pos + bw/2, STi_p, bw, 'FaceColor', colors_STi, 'EdgeColor', 'none');
    hold off;

    % 坐标轴标签与格式设置
    ax = gca;
    set(ax, 'XTick', x_pos, 'XTickLabel', pnames, ...
            'XTickLabelRotation', 40, 'FontSize', 10, ...
            'TickLabelInterpreter', 'none');
    ylabel('Sensitivity index', 'FontSize', 11);
    xlabel('Parameter',         'FontSize', 11);
    title(['Sobol 敏感性指标 — ' out_labels{o}], 'FontSize', 12);

    legend([b1, b2], {'S_i  (first-order)', 'ST_i  (total-order)'}, ...
           'Location', 'northeast', 'FontSize', 10);

    y_top  = max([STi_p; 0.05]);
    y_bot  = min([Si_p; STi_p; 0]);          % show negatives if they exist
    ylim([y_bot - 0.01, y_top * 1.18]);
    grid on;  box on;

    % 标注柱高（过小的数值不标注）
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

fprintf('\n=== 分析完成 ===\n');


%% =========================================================
%%  局部辅助函数（不是模型函数）
% ==========================================================
function s = format_val(v)
% 返回格式化后的字符串；若不是有限值则返回 'NaN'。
if isfinite(v)
    s = sprintf('%10.4f', v);
else
    s = '       NaN';
end
end


%% =========================================================
%%  局部函数——发动机前向模型
%   （请勿修改下面的物理模型部分）
% ==========================================================

function [y, aux] = engine_forward(theta, cond)
%ENGINE_FORWARD 航空发动机热力循环前向模型。
%
%  theta(1)  eta_k      压气机等熵效率
%  theta(2)  eta_t      涡轮等熵效率
%  theta(3)  eta_m      机械效率
%  theta(4)  eta_v      冷喷管速度系数
%  theta(5)  eta_tv     热喷管速度系数
%  theta(6)  eta_c1     内涵道喷管效率
%  theta(7)  eta_c2     外涵道喷管效率
%  theta(8)  sigma_cc   燃烧室总压恢复系数
%  theta(9)  sigma_kan  进气道总压恢复系数
%  theta(10) sigma_kask 级间导管总压恢复系数
%  theta(11) sigma_ks   喷管总压恢复系数
%  theta(12) eta_T      涡轮冷却效率
%  theta(13) lambda     喷管面积比

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

% --- 声速与飞行速度 ---
a = sqrt(k_air * R_air * T_H);
if ~isfinite(a) || a <= 0, return; end
V_flight = a * M_flight;

% --- 燃气物性表 ---
kT = piecewise_kT(T_g);
RT = piecewise_RT(T_g);
d  = delta_cooling(T_g);

% --- 进气冲压总压与总温 ---
inner = 1 + V_flight^2 / (2 * (k_air/(k_air-1)) * R_air * T_H);
if inner <= 0, return; end
tau_v = inner^(k_air / (k_air - 1));
T_B   = T_H * (inner^k_air);
if ~isfinite(T_B) || T_B <= 0, return; end

% --- 压气机出口温度 ---
pi_k_ratio = pi_k^((k_air-1)/k_air);
if ~isfinite(pi_k_ratio) || pi_k_ratio < 1, return; end
T_k = T_B * (1 + (pi_k_ratio - 1) / eta_k);
if ~isfinite(T_k) || T_k <= 0, return; end

% --- 冷却空气质量分数 ---
g_T = 3e-5 * T_g - 2.69e-5 * T_k - 0.003;
if ~isfinite(g_T) || g_T <= 0, return; end

% --- 放热系数（lambda_heat）---
compress_work = (k_air/(k_air-1)) * R_air * T_B * (pi_k_ratio - 1);
gas_enthalpy  = (kT/(kT-1)) * RT * T_g;
if abs(gas_enthalpy) < 1e-6, return; end

num_lambda = 1 - compress_work / (gas_enthalpy * eta_k);
den_lambda = 1 - compress_work / (gas_enthalpy * eta_k * eta_t);
if abs(den_lambda) < 1e-10, return; end
lambda_heat = num_lambda / den_lambda;
if ~isfinite(lambda_heat), return; end

% --- 涡轮膨胀 ---
sigma_bx           = sigma_cc * sigma_kan;
exp_T              = (kT - 1) / kT;
expansion_pr_denom = tau_v * sigma_bx * pi_k * sigma_kask * sigma_ks;
if expansion_pr_denom <= 0, return; end
expansion_term = (1.0 / expansion_pr_denom)^exp_T;
if ~isfinite(expansion_term), return; end

term1 = (kT / (kT - 1)) * RT * T_g * (1 - expansion_term);

% 重新计算压气机功（与上式相同，此处显式写出便于阅读）
compress_work = (k_air / (k_air - 1)) * R_air * T_B * (pi_k^((k_air - 1) / k_air) - 1);

denom2 = (1 + g_T) * eta_k * eta_T * eta_t * eta_m * (1 - d);
if abs(denom2) < 1e-10, return; end
term2 = compress_work / denom2;

L_sv = lambda_heat * (term1 - term2);
if ~isfinite(L_sv) || L_sv <= 0, return; end

% --- 喷管面积分配系数 x_pc ---
V2_term = m * V_flight^2;
num_xpc = 1 + V2_term / (2 * L_sv * eta_tv * eta_v * eta_c2);
den_xpc = 1 + (m * eta_tv * eta_v * eta_c2) / (eta_c1 * lambda);
if abs(den_xpc) < 1e-10, return; end
x_pc = num_xpc / den_xpc;
if ~isfinite(x_pc) || x_pc <= 0 || x_pc >= 1
    if x_pc <= 0, return; end
end

% --- 喷流速度 ---
inner_sq1 = 2 * eta_c1 * lambda * x_pc * L_sv;
if inner_sq1 < 0, return; end
V_j1 = (1 + g_T) * sqrt(inner_sq1) - V_flight;

inner_sq2 = 2 * (1 - x_pc) / m * L_sv * eta_tv * eta_v * eta_c2 + V_flight^2;
if inner_sq2 < 0, return; end
V_j2 = sqrt(inner_sq2) - V_flight;

% --- 性能输出 ---
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
%%  燃气物性与冷却函数
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
