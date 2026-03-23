clear; clc; close all;

%% =====================================================================
%% 1. 参数定义
%% =====================================================================

param_labels = { ...
    'eta_k', 'eta_t', 'eta_T', 'eta_m', ...
    'lambda', 'eta_v', 'eta_tv', 'eta_c1', 'eta_c2', ...
    'sigma_cc', 'sigma_kan', 'sigma_kask', 'sigma_ks'};

param_names = param_labels;

param_labels_latex = { ...
    '$\eta_k$', '$\eta_t$', '$\eta_T$', '$\eta_m$', ...
    '$\lambda$', '$\eta_v$', '$\eta_{tv}$', '$\eta_{c1}$', '$\eta_{c2}$', ...
    '$\sigma_{cc}$', '$\sigma_{kan}$', '$\sigma_{kask}$', '$\sigma_{ks}$'};

n_params = 13;

% 13个参数的先验上下界
lb = [ ...
    0.84, 0.86, 0.97, 0.98, ...
    1.00, 0.84, 0.88, 0.90, 0.90, ...
    0.96, 0.95, 0.96, 0.93];

ub = [ ...
    0.86, 0.92, 0.99, 0.995, ...
    1.06, 0.90, 0.96, 0.98, 0.98, ...
    1.04, 1.01, 1.02, 0.99];

% 13个参数的真值
theta_true = [ ...
    0.855, 0.900, 0.980, 0.990, ...
    1.030, 0.870, 0.920, 0.950, 0.940, ...
    1.000, 0.980, 0.990, 0.960];

% 不再有固定参数
theta_fixed = [];

% 工况参数
cond.T_H      = 288;     % 大气总温 (K)
cond.M_flight = 0.0;     % 飞行马赫数（地面静止）
cond.m        = 10.0;    % 涵道比
cond.pi_k     = 33.0;    % 压气机压比
cond.T_g      = 1700.0;  % 涡轮前总温 (K)


assert(all(theta_true >= lb) && all(theta_true <= ub), 'theta_true 超出先验范围');

fprintf('===== 工况参数 =====\n');
fprintf('  T_H=%g K  M_flight=%g  m=%g  pi_k=%g  T_g=%g K\n\n', ...
    cond.T_H, cond.M_flight, cond.m, cond.pi_k, cond.T_g);

%% =====================================================================
%% 2. 前向模型验证
%% =====================================================================

fprintf('===== 前向模型验证 =====\n');
[y_true, ~] = engine_forward(theta_true, theta_fixed, cond);
if ~all(isfinite(y_true))
    error('前向模型在 theta_true 处输出非有限值，请检查参数设置');
end
fprintf('  R_ud = %.4f  [N·s/kg]\n', y_true(1));
fprintf('  C_ud = %.6f [kg/(N·h)]\n\n', y_true(2));

%% =====================================================================
%% 3. 生成虚拟观测数据
%% =====================================================================

rng(2026);  
data = generate_virtual_data(theta_true, theta_fixed, cond, 0.001, 0.001);

fprintf('===== 虚拟观测数据 =====\n');
fprintf('  R: 真值=%.4f, 观测=%.4f, sigma_R=%.4f\n', ...
    data.R_true, data.R_obs, data.sigma_R);
fprintf('  C: 真值=%.6f, 观测=%.6f, sigma_C=%.6f\n\n', ...
    data.C_true, data.C_obs, data.sigma_C);

%% =====================================================================
%% 4. TMCMC 算法配置
%% =====================================================================

opts.N          = 20000;   % 或 30000
opts.COV_target = 1.0;
opts.N_MCMC     = 12;      % 或 15
opts.scale      = 0.25;    % 先缩小，避免13维下接受率太低
opts.max_stages = 100;

fprintf('===== TMCMC 配置 =====\n');
fprintf('  粒子数 N=%d, COV_target=%.1f, N_MCMC=%d, scale=%.2f\n\n', ...
    opts.N, opts.COV_target, opts.N_MCMC, opts.scale);

%% =====================================================================
%% 5. 运行 TMCMC
%% =====================================================================

fprintf('===== 运行 TMCMC =====\n');
results = run_tmcmc(data, theta_fixed, cond, lb, ub, opts);
fprintf('\nTMCMC 完成，共 %d 个过渡阶段\n\n', results.n_stages);

%% =====================================================================
%% 6. 后验统计结果
%% =====================================================================

fprintf('后验统计结果\n');
fprintf('%-12s %8s %10s %10s %10s %10s\n', ...
    '参数','真值','后验均值','MAP','CI95_低','CI95_高');
fprintf('%s\n', repmat('-',1,66));
for i = 1:n_params
    fprintf('%-12s %8.4f %10.4f %10.4f %10.4f %10.4f\n', ...
        param_names{i}, theta_true(i), results.theta_mean(i), ...
        results.theta_map(i), results.theta_ci95(i,1), results.theta_ci95(i,2));
end
fprintf('\n');

%% =====================================================================
%% 7. 后验预测对比
%% =====================================================================

[y_mn, ~] = engine_forward(results.theta_mean, theta_fixed, cond);
[y_mp, ~] = engine_forward(results.theta_map,  theta_fixed, cond);

fprintf('后验预测对比\n');
fprintf('  %-18s %14s %16s\n', '','R_ud [N·s/kg]','C_ud [kg/(N·h)]');
fprintf('  %-18s %14.4f %16.6f\n','真值',        data.R_true, data.C_true);
fprintf('  %-18s %14.4f %16.6f\n','观测值',      data.R_obs,  data.C_obs);
fprintf('  %-18s %14.4f %16.6f\n','后验均值预测', y_mn(1),    y_mn(2));
fprintf('  %-18s %14.4f %16.6f\n','MAP 预测',    y_mp(1),    y_mp(2));
fprintf('\n');
fprintf('  后验均值相对误差: R=%.3f%%, C=%.3f%%\n', ...
    abs(y_mn(1)-data.R_true)/data.R_true*100, ...
    abs(y_mn(2)-data.C_true)/data.C_true*100);
fprintf('  MAP    相对误差: R=%.3f%%, C=%.3f%%\n\n', ...
    abs(y_mp(1)-data.R_true)/data.R_true*100, ...
    abs(y_mp(2)-data.C_true)/data.C_true*100);

%% =====================================================================
%% 8. 绘图
%% =====================================================================

plot_results(results, theta_true, lb, ub, param_labels, param_labels_latex);

%% =====================================================================
%%                         前向模型
%% =====================================================================

%% --- 分段绝热指数 ---
function kT = piecewise_kT(T_g)
    if T_g > 1600
        kT = 1.25;
    elseif T_g > 1400 && T_g <= 1600
        kT = 1.30;
    else
        kT = 1.33;
    end
end

%% --- 分段燃气常数 ---
function RT = piecewise_RT(T_g)
    if T_g > 1600
        RT = 288.6;
    elseif T_g > 1400 && T_g <= 1600
        RT = 288.0;
    else
        RT = 287.6;
    end
end

%% --- 冷却引气系数 ---
function d = delta_cooling(T_g)
    d = max(0.0, min(0.15, 0.02 + (T_g - 1200)/100*0.02));
end

%% --- 前向模型 ---
function [y, aux] = engine_forward(theta_s, theta_f, cond)

    % 13个待反演参数全部从 theta_s 中读取
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

    y   = [NaN, NaN];
    aux = struct();

    try
        % (1) 基本常数
        k_air = 1.4;
        R_air = 287.3;
        a = sqrt(k_air * R_air * T_H);
        if ~isfinite(a) || a <= 0
            return;
        end
        V_flight = a * M_flight;

        % (2) 分段函数值
        kT = piecewise_kT(T_g);
        RT = piecewise_RT(T_g);
        d  = delta_cooling(T_g);

        % (3) 进口总压比与压气机入口温度
        inner = 1 + V_flight^2 / (2 * (k_air/(k_air-1)) * R_air * T_H);
        if inner <= 0
            return;
        end
        tau_v = inner^(k_air / (k_air - 1));
        T_B   = T_H * (inner^k_air);
        if ~isfinite(T_B) || T_B <= 0
            return;
        end

        % (4) 压气机出口温度
        pi_k_ratio = pi_k^((k_air-1)/k_air);
        if ~isfinite(pi_k_ratio) || pi_k_ratio < 1
            return;
        end
        T_k = T_B * (1 + (pi_k_ratio - 1) / eta_k);
        if ~isfinite(T_k) || T_k <= 0
            return;
        end

        % (5) 相对耗油量
        g_T = 3e-5 * T_g - 2.69e-5 * T_k - 0.003;
        if ~isfinite(g_T) || g_T <= 0
            return;
        end

        % (6) 热恢复系数
        compress_work = (k_air/(k_air-1)) * R_air * T_B * (pi_k_ratio - 1);
        gas_enthalpy  = (kT/(kT-1)) * RT * T_g;
        if abs(gas_enthalpy) < 1e-6
            return;
        end

        num_lambda = 1 - compress_work / (gas_enthalpy * eta_k);
        den_lambda = 1 - compress_work / (gas_enthalpy * eta_k * eta_t);
        if abs(den_lambda) < 1e-10
            return;
        end
        lambda_heat = num_lambda / den_lambda;
        if ~isfinite(lambda_heat)
            return;
        end

        % (7) 进口总压恢复系数
        sigma_bx = sigma_cc * sigma_kan;

        % (8) 单位自由能
        exp_T = (kT - 1) / kT;
        expansion_pr_denom = tau_v * sigma_bx * pi_k * sigma_kask * sigma_ks;
        if expansion_pr_denom <= 0
            return;
        end
        expansion_term = (1.0 / expansion_pr_denom)^exp_T;
        if ~isfinite(expansion_term)
            return;
        end

        term1 = (kT / (kT - 1)) * RT * T_g * (1 - expansion_term);

        compress_work = (k_air / (k_air - 1)) * R_air * T_B * ...
            (pi_k^((k_air - 1) / k_air) - 1);

        denom2 = (1 + g_T) * eta_k * eta_T * eta_t * eta_m * (1 - d);
        if abs(denom2) < 1e-10
            return;
        end
        term2 = compress_work / denom2;

        L_sv = lambda_heat * (term1 - term2);
        if ~isfinite(L_sv) || L_sv <= 0
            return;
        end

        % (9) 最优自由能分配系数
        V2_term = m * V_flight^2;
        num_xpc = 1 + V2_term / (2 * L_sv * eta_tv * eta_v * eta_c2);
        den_xpc = 1 + (m * eta_tv * eta_v * eta_c2) / (eta_c1 * lambda);
        if abs(den_xpc) < 1e-10
            return;
        end
        x_pc = num_xpc / den_xpc;
        if ~isfinite(x_pc) || x_pc <= 0 || x_pc >= 1
            if x_pc <= 0
                return;
            end
        end

        % (10) 比推力
        inner_sq1 = 2 * eta_c1 * lambda * x_pc * L_sv;
        if inner_sq1 < 0
            return;
        end
        V_j1 = (1 + g_T) * sqrt(inner_sq1) - V_flight;

        inner_sq2 = 2 * (1 - x_pc) / m * L_sv * eta_tv * eta_v * eta_c2 + V_flight^2;
        if inner_sq2 < 0
            return;
        end
        V_j2 = sqrt(inner_sq2) - V_flight;

        R_ud = (1/(1+m)) * V_j1 + (m/(1+m)) * V_j2;
        if ~isfinite(R_ud) || R_ud <= 0
            return;
        end

        % (11) 比油耗
        denom_C = R_ud * (1 + m);
        if abs(denom_C) < 1e-10
            return;
        end
        C_ud = 3600 * g_T * (1 - d) / denom_C;
        if ~isfinite(C_ud) || C_ud <= 0
            return;
        end

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

%% --- 生成虚拟观测 ---
function data = generate_virtual_data(theta_true, theta_f, cond, nR, nC)
    [yt, ~] = engine_forward(theta_true, theta_f, cond);
    if ~all(isfinite(yt))
        error('generate_virtual_data: 前向模型失效');
    end

    sR = nR * abs(yt(1));
    sC = nC * abs(yt(2));

    data.R_true  = yt(1);
    data.C_true  = yt(2);
    data.R_obs   = yt(1) + sR * randn();
    data.C_obs   = yt(2) + sC * randn();
    data.sigma_R = sR;
    data.sigma_C = sC;
end

%% --- 对数似然 ---
function ll = log_likelihood(theta, theta_f, data, cond)
    ll = -Inf;
    [yp, ~] = engine_forward(theta, theta_f, cond);
    if ~all(isfinite(yp))
        return;
    end

    rR = (data.R_obs - yp(1)) / data.sigma_R;
    rC = (data.C_obs - yp(2)) / data.sigma_C;

    ll = -0.5 * (rR^2 + rC^2);
    if ~isfinite(ll)
        ll = -Inf;
    end
end

%% --- 求下一个 beta：二分法使权重 COV = target ---
function beta_next = find_next_beta(log_likes, beta_prev, cov_tgt)
    valid = isfinite(log_likes);
    if sum(valid) < 2
        beta_next = 1.0;
        return;
    end
    ll = log_likes(valid);

    if compute_cov(ll, 1.0 - beta_prev) <= cov_tgt
        beta_next = 1.0;
        return;
    end

    lo = beta_prev;
    hi = 1.0;

    for iter_bisect = 1:60
        mid = 0.5 * (lo + hi);
        if compute_cov(ll, mid - beta_prev) <= cov_tgt
            lo = mid;
        else
            hi = mid;
        end
        if hi - lo < 1e-7
            break;
        end
    end

    beta_next = min(max(lo, beta_prev + 1e-6), 1.0);
end

function cv = compute_cov(ll_vec, db)
    if db <= 0
        cv = 0;
        return;
    end

    lw = db * ll_vec;
    lw = lw - max(lw);
    w  = exp(lw);

    mu = mean(w);
    if mu <= 0
        cv = Inf;
        return;
    end

    cv = std(w) / mu;
end

%% --- TMCMC 主算法 ---
function results = run_tmcmc(data, theta_f, cond, lb, ub, opts)

    N       = opts.N;            % 粒子数
    n_p     = length(lb);        % 参数维度
    cov_tgt = opts.COV_target;   % 权重COV目标
    nMCMC   = opts.N_MCMC;       % 每阶段MCMC步数
    sc      = opts.scale;        % 提议协方差缩放因子

    % 均匀分布 U(lb, ub)
    particles = bsxfun(@plus, lb, bsxfun(@times, ub - lb, rand(N, n_p)));

    % 计算每个粒子的 log-likelihood
    log_L = zeros(N, 1);

    fprintf('  初始化粒子似然...');
    for i = 1:N
        log_L(i) = log_likelihood(particles(i,:), theta_f, data, cond);
    end
    fprintf(' 完成（有效粒子 %d/%d）\n', sum(isfinite(log_L)), N);

   
    % 3. 初始化 TMCMC 状态变量
    beta_cur = 0;       % 当前 β（从先验开始）
    betas    = 0;       % 记录每一阶段 β
    log_evid = 0;       % 模型证据（边际似然）
    stage    = 0;       % 阶段计数
    acc_hist = [];      % MCMC接受率记录

  
    % 4. 主循环：逐步从 β=0 → β=1
    while beta_cur < 1.0

        stage = stage + 1;

        % 防止无限循环
        if stage > opts.max_stages
            warning('达到最大阶段数上限 %d', opts.max_stages);
            break;
        end

       
        % 自适应选择 β_next
        beta_new = find_next_beta(log_L, beta_cur, cov_tgt);
        db       = beta_new - beta_cur;   % β增量


        %计算重要性权w_i ∝ L(θ_i)^(db) 
        log_w = db * log_L;

        % 数值稳定处理
        finite_mask = isfinite(log_w);
        lw_max = max(log_w(finite_mask));

        w = exp(log_w - lw_max);
        w(~isfinite(w)) = 0;

        w_sum = sum(w);

        % 异常检查（权重塌缩）
        if w_sum <= 0 || ~isfinite(w_sum)
            warning('权重和无效，提前停止');
            break;
        end

        % 归一化权重
        w_n = w / w_sum;

        % 更新模型证据（边际似然）
        log_evid = log_evid + log(w_sum / N) + lw_max;
        
        % 重采样
        idx   = datasample(1:N, N, 'Weights', w_n);
        pts_r = particles(idx, :);   % 重采样粒子
        lL_r  = log_L(idx);          % 对应似然

        
        % 构造提议分布
        mu_w = (w_n' * particles);

        % 加权协方差
        dif  = bsxfun(@minus, particles, mu_w);
        Sig  = (bsxfun(@times, dif, w_n))' * dif;

        % 提议协方差（缩放）
        Sig_prop = sc^2 * Sig + 1e-10 * eye(n_p);

        % Cholesky 分解（用于采样）
        [L_ch, flg] = chol(Sig_prop, 'lower');
        if flg ~= 0
            % 数值不稳定时用对角近似
            L_ch = diag(sqrt(max(diag(Sig_prop), 1e-12)));
        end

      
        % MCMC 扰动
        pts_new = pts_r;
        lL_new  = lL_r;
        n_acc   = 0;

        for i = 1:N

            th_c = pts_r(i, :);          % 当前粒子
            llc  = beta_new * lL_r(i);   % 当前“温度化”后验

            for iter_mcmc = 1:nMCMC

                % 提议新样本
                z    = randn(n_p, 1);
                th_p = th_c + (L_ch * z)';

                % 边界检查（先验约束）
                if any(th_p < lb) || any(th_p > ub)
                    continue;
                end

                % 新样本似然
                ll_raw = log_likelihood(th_p, theta_f, data, cond);
                llp    = beta_new * ll_raw;

                % MH接受准则
                if log(rand()) < llp - llc
                    th_c = th_p;
                    llc  = llp;
                    n_acc = n_acc + 1;
                end
            end

            % 更新粒子
            pts_new(i, :) = th_c;
            lL_new(i)     = log_likelihood(th_c, theta_f, data, cond);
        end

       
        % 统计接受率
        ar = n_acc / (N * nMCMC);
        acc_hist(end+1) = ar;

        fprintf('  Stage %2d: beta %.4f -> %.4f  |  MCMC 接受率 = %.3f\n', ...
            stage, beta_cur, beta_new, ar);

        particles = pts_new;
        log_L     = lL_new;
        beta_cur  = beta_new;
        betas(end+1) = beta_new;

        % 如果已到达后验
        if beta_cur >= 1.0
            break;
        end
    end

   
    % 均值（后验期望）
    th_mean = mean(particles, 1);

    % 标准差（不确定性）
    th_std  = std(particles, 0, 1);

    % MAP
    [~, bi] = max(log_L);
    th_map  = particles(bi, :);

    % 95%置信区间
    ci95 = zeros(n_p, 2);
    for i = 1:n_p
        ci95(i, :) = quantile(particles(:, i), [0.025, 0.975]);
    end

    results.particles    = particles;
    results.log_L        = log_L;
    results.theta_mean   = th_mean;
    results.theta_std    = th_std;
    results.theta_map    = th_map;
    results.theta_ci95   = ci95;
    results.stage_betas  = betas;
    results.acc_hist     = acc_hist;
    results.log_evidence = log_evid;
    results.n_stages     = stage;
end
%% --- 绘图 ---
function plot_results(results, theta_true, lb, ub, param_labels, param_labels_latex)
    pts = results.particles;
    n_p = size(pts, 2);

    % 颜色
    col_hist    = [0.08, 0.38, 0.65];
    col_prior   = [1.00, 0.75, 0.10];
    col_kde     = [0.15, 0.75, 0.35];
    col_scatter = [0.10, 0.65, 0.60];

    %% ---------------------------------------------------------------
    %% 图1: Corner plot
    %% ---------------------------------------------------------------
    fig2 = figure('Name', 'TMCMC 联合后验 Corner Plot', ...
                  'Position', [60, 40, 960, 900]);

    ng = 45;
    h_hist = [];
    h_prior = [];
    h_kde = [];

    for row = 1:n_p
        for col = 1:n_p
            ax = subplot(n_p, n_p, (row-1)*n_p + col);

            if row == col
                % 对角：边缘分布
                h_h = histogram(pts(:, col), 50, 'Normalization', 'pdf', ...
                    'FaceColor', col_hist, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
                hold on;

                xlims = [lb(col), ub(col)];
                xlim(xlims);

                prior_pdf = 1 / (ub(col) - lb(col));
                xp_vec = linspace(lb(col), ub(col), 200);
                h_p = plot(xp_vec, prior_pdf * ones(1, 200), '-', ...
                    'Color', col_prior, 'LineWidth', 2.0);

                x_eval = linspace(lb(col), ub(col), 200);
                try
                    [f_kde, xi_kde] = ksdensity(pts(:, col), x_eval);
                    h_k = plot(xi_kde, f_kde, '--', ...
                        'Color', col_kde, 'LineWidth', 2.0);
                catch
                    h_k = [];
                end

                if row == 1
                    h_hist  = h_h;
                    h_prior = h_p;
                    if ~isempty(h_k)
                        h_kde = h_k;
                    end
                end

                xline(theta_true(col), '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.2);

                set(ax, 'Box', 'on');
                ylabel('pdf', 'FontSize', 8);

            elseif row > col
                % 下三角：2D KDE 热力图
                xg = linspace(lb(col), ub(col), ng);
                yg = linspace(lb(row), ub(row), ng);
                [Xg, Yg] = meshgrid(xg, yg);
                grid_pts = [Xg(:), Yg(:)];

                try
                    f2d = ksdensity([pts(:, col), pts(:, row)], grid_pts);
                    F2d = reshape(f2d, ng, ng);
                catch
                    F2d = zeros(ng, ng);
                end

                imagesc(ax, xg, yg, F2d);
                colormap(ax, parula);
                set(ax, 'YDir', 'normal');
                hold on;

                plot(theta_true(col), theta_true(row), 'w+', ...
                    'MarkerSize', 9, 'LineWidth', 2.0);

                xlim([lb(col), ub(col)]);
                ylim([lb(row), ub(row)]);
                set(ax, 'Box', 'on');

            else
                % 上三角：散点图
                scatter(pts(:, col), pts(:, row), 3, 'filled', ...
                    'MarkerFaceColor', col_scatter, ...
                    'MarkerFaceAlpha', 0.18, ...
                    'MarkerEdgeAlpha', 0.18);
                hold on;

                plot(theta_true(col), theta_true(row), 'r+', ...
                    'MarkerSize', 9, 'LineWidth', 2.0);

                xlim([lb(col), ub(col)]);
                ylim([lb(row), ub(row)]);
                set(ax, 'Box', 'on');
            end

            % 坐标轴标签
            if row == n_p
                xlabel(param_labels_latex{col}, 'FontSize', 9, 'Interpreter', 'latex');
            end
            if col == 1 && row ~= col
                ylabel(param_labels_latex{row}, 'FontSize', 9, 'Interpreter', 'latex');
            end
            if col == 1 && row == 1
                ylabel(param_labels_latex{1}, 'FontSize', 9, 'Interpreter', 'latex');
            end

            set(ax, 'FontSize', 7, 'TickLength', [0.015 0.015]);
        end
    end

    leg_handles = [];
    leg_labels  = {};
    if ~isempty(h_prior)
        leg_handles(end+1) = h_prior;
        leg_labels{end+1} = '先验分布';
    end
    if ~isempty(h_kde)
        leg_handles(end+1) = h_kde;
        leg_labels{end+1} = '贝叶斯后验';
    end
    if ~isempty(h_hist)
        leg_handles(end+1) = h_hist;
        leg_labels{end+1} = 'TMCMC';
    end

    if ~isempty(leg_handles)
        lgd = legend(leg_handles, leg_labels, ...
            'Orientation', 'horizontal', 'FontSize', 10, ...
            'Location', 'southoutside', 'Box', 'on');
        lgd.Position(2) = 0.01;
    end

    sgtitle('四个修正参数的联合与边际后验分布图', 'FontSize', 12, 'FontWeight', 'bold');

    %% ---------------------------------------------------------------
    %% 图2: 单独边缘后验
    %% ---------------------------------------------------------------
    fig1 = figure('Name', 'TMCMC 边缘后验（详细）', ...
                  'Position', [50, 50, 1100, 700]);

    nc = ceil(n_p / 2);
    for i = 1:n_p
        subplot(2, nc, i);

        histogram(pts(:, i), 60, 'Normalization', 'pdf', ...
            'FaceColor', col_hist, 'EdgeColor', 'none', 'FaceAlpha', 0.80);
        hold on;

        xline(theta_true(i), '-',  'LineWidth', 2.2, ...
            'Color', [0.1 0.7 0.2], 'DisplayName', '真值');
        xline(results.theta_mean(i), '--', 'LineWidth', 1.8, ...
            'Color', [0.85 0.15 0.1], 'DisplayName', '后验均值');
        xline(results.theta_map(i), ':', 'LineWidth', 1.8, ...
            'Color', [0.7 0.1 0.8], 'DisplayName', 'MAP');

        x_eval = linspace(lb(i), ub(i), 200);
        try
            [fk, xk] = ksdensity(pts(:, i), x_eval);
            plot(xk, fk, '--', 'Color', col_kde, 'LineWidth', 1.5, 'DisplayName', '后验KDE');
        catch
        end

        xlim([lb(i), ub(i)]);
        xlabel(param_labels_latex{i}, 'FontSize', 11, 'Interpreter', 'latex');
        ylabel('pdf', 'FontSize', 10);

        title(sprintf('%s  真值=%.4f | 均值=%.4f | MAP=%.4f', ...
            param_labels{i}, theta_true(i), ...
            results.theta_mean(i), results.theta_map(i)), ...
            'FontSize', 8.5, 'Interpreter', 'none');

        grid on;
        grid minor;

        if i == 1
            legend('Location', 'best', 'FontSize', 8);
        end
    end

    sgtitle('边缘后验分布直方图', 'FontSize', 12);

    %% ---------------------------------------------------------------
    %% 图3: beta 演化 + 接受率诊断
    %% ---------------------------------------------------------------
    fig3 = figure('Name', 'TMCMC 过程诊断', ...
                  'Position', [200, 50, 950, 380]);

    ns = results.n_stages;

    subplot(1, 2, 1);
    plot(0:ns, results.stage_betas, 'bo-', ...
        'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', 'b');
    xlabel('阶段序号', 'FontSize', 11);
    ylabel('\beta', 'FontSize', 11);
    title('温度参数 \beta 演化', 'FontSize', 12);
    ylim([0, 1.05]);
    grid on;
    grid minor;

    subplot(1, 2, 2);
    bar(1:ns, results.acc_hist, 0.6, 'FaceColor', [0.30, 0.68, 0.38]);
    hold on;
    xlabel('阶段序号', 'FontSize', 11);
    ylabel('MCMC 接受率', 'FontSize', 11);
    title('各阶段 MCMC 接受率', 'FontSize', 12);
    ylim([0, 1]);
    grid on;
    grid minor;
    legend('FontSize', 8, 'Location', 'best');

    sgtitle('TMCMC 过程诊断', 'FontSize', 12);

end
