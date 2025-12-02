% function loss_table = mse_plot_ucb_trajectories(M, Y0, opts)
% %MSE_PLOT_UCB_TRAJECTORIES UCB-based Gradient Descent trajectories.
% %
% %   UCB selection over K initializations (arms) to minimize a non-convex
% %   loss by gradient descent.
% %
% %   Each arm is a 2n-dimensional vector x = [a; b], where
% %       a, b ∈ R^n.
% %   The loss for one arm is
% %       l(a,b) = sum_{j=1}^n L(a_j, b_j),
% %   where M.L is the scalar loss L(a,b) and M.gradL its scalar gradient
% %   [dL/da; dL/db].  This function builds l and ∇l internally.
% %
% %   Inputs
% %   ------
% %     M    : struct with fields
% %               M.L(a,b)     : scalar loss
% %               M.gradL(a,b) : 2×1 gradient for scalar pair
% %               (optionally) M.true_min_loss or M.Lmin : analytic minimum
% %
% %     Y0   : (2n)×K or K×(2n) matrix of initial conditions (K arms)
% %
% %     opts : struct with optional fields
% %               .eta          : learning rate (default 0.01)
% %               .n_steps      : time horizon T (default 1000)
% %               .linewidth    : line width for trajectories (default 1.5)
% %               .colors       : K×3 color matrix (default: lines(K))
% %               .true_min_loss: analytic minimum of l(a,b) (overrides M)
% %
% %   Figures
% %   -------
% %     1) 2D trajectories of the first coordinate pair (a1,b1) for each arm.
% %     2) Loss vs. global step:
% %          - For each arm, points only when that arm is updated (descent
% %            step), plotted as a line with markers.
% %          - Continuous red horizontal line at the analytic minimum of
% %            the loss function.
% %
% %   Output
% %   ------
% %     loss_table : table with columns
% %                     step, loss_arm_1, ..., loss_arm_K
% %                  where loss_arm_k(t) is l(a,b) for arm k after step t.
% 
%     if nargin < 3, opts = struct(); end
% 
%     % ----- Default options -----
%     if ~isfield(opts, 'eta'),       opts.eta       = 0.01; end
%     if ~isfield(opts, 'n_steps'),   opts.n_steps   = 1000; end
%     if ~isfield(opts, 'linewidth'), opts.linewidth = 1.5;  end
% 
%     % ----- Initialize arms -----
%     % Allow Y0 to be (2n)×K or K×(2n)
%     [dim, K] = size(Y0);
%     if mod(dim, 2) ~= 0
%         % Maybe it's K×(2n)
%         if mod(size(Y0, 2), 2) == 0
%             Y0 = Y0.';
%             [dim, K] = size(Y0);
%         else
%             error('Y0 must be of size (2n)×K or K×(2n).');
%         end
%     end
%     n = dim / 2;  % number of (a,b) pairs per arm
% 
%     if ~isfield(opts, 'colors'), opts.colors = lines(K); end
% 
%     X = Y0;                % current positions of all K arms (2n×K)
%     N = zeros(1, K);       % pull counts N_k(t)
% 
%     % 2D trajectory history (we store full vectors, plot first 2 coords)
%     history = cell(1, K);
%     for k = 1:K
%         history{k} = X(:, k);
%     end
% 
%     % Loss history for all arms at all steps (for table)
%     loss_hist_all = nan(opts.n_steps, K);
%     % Loss only at descent steps (for plotting)
%     loss_hist_selected = nan(opts.n_steps, K);
% 
%     % ----- Analytic minimum of the loss -----
%     if isfield(opts, 'true_min_loss')
%         true_min_loss = opts.true_min_loss;
%     elseif isfield(M, 'true_min_loss')
%         true_min_loss = M.true_min_loss;
%     elseif isfield(M, 'Lmin')
%         true_min_loss = M.Lmin;
%     else
%         % For MSE-type losses this is usually 0; override via opts/M if needed.
%         true_min_loss = 0;
%     end
% 
%     % ----- Helper: full loss l(a,b) for one arm -----
%     function val = full_loss(x)
%         % x is 2n×1: [a; b]
%         a = x(1:n);
%         b = x(n+1:2*n);
%         val = 0;
%         for j = 1:n
%             val = val + M.L(a(j), b(j));
%         end
%     end
% 
%     % ----- Helper: gradient of l(a,b) for one arm -----
%     function g = full_grad(x)
%         a = x(1:n);
%         b = x(n+1:2*n);
%         ga = zeros(n,1);
%         gb = zeros(n,1);
%         for j = 1:n
%             g_j = M.gradL(a(j), b(j));   % 2×1 gradient [dL/da; dL/db]
%             ga(j) = g_j(1);
%             gb(j) = g_j(2);
%         end
%         g = [ga; gb];  % 2n×1
%     end
% 
%     % =========================
%     %  Figure 1: trajectories
%     % =========================
%     figure;
%     hold on;
% 
%     % ----- Main UCB–GD loop -----
%     for t = 1:opts.n_steps
% 
%         % 1. Compute UCB statistic for each arm:
%         %    U_k(t) = tanh(l(x_k(t))) - sqrt( 2 log(1 + t log^2 t) / (N_k(t)+1) )
%         U = zeros(1, K);
% 
%         if t == 1
%             numerator = 0;
%         else
%             term = 1 + t * (log(t))^2;   % natural log
%             numerator = 2 * log(term);
%         end
% 
%         for k = 1:K
%             f_val = full_loss(X(:, k));              % current loss l(a,b)
%             c_explore = 1;
%             exploration = sqrt(c_explore * numerator / (N(k) + 1));
%             U(k) = tanh(f_val) - exploration;        % LCB for minimization
%         end
% 
%         % 2. Select arm
%         [~, A_t] = min(U);
% 
%         % 3. Gradient descent step on selected arm
%         gA = full_grad(X(:, A_t));
%         X(:, A_t) = X(:, A_t) - opts.eta * gA;
% 
%         % 4. Update pull count
%         N(A_t) = N(A_t) + 1;
% 
%         % Store new position for 2D plotting
%         history{A_t} = [history{A_t}, X(:, A_t)];
% 
%         % 5. Record losses AFTER this step
%         for k = 1:K
%             loss_hist_all(t, k) = full_loss(X(:, k));
%         end
%         % Only the selected arm has a descent step at time t
%         loss_hist_selected(t, A_t) = loss_hist_all(t, A_t);
%     end
% 
%     % ----- Draw 2D trajectories (first two coordinates) -----
%     for k = 1:K
%         traj = history{k};
%         if size(traj, 2) > 1
%             plot(traj(1,:), traj(2,:), '.-', ...
%                  'Color',     opts.colors(k,:), ...
%                  'LineWidth', opts.linewidth, ...
%                  'MarkerSize', 8);
%             % Mark final position
%             plot(traj(1,end), traj(2,end), 'o', ...
%                  'Color',           opts.colors(k,:), ...
%                  'MarkerFaceColor', opts.colors(k,:), ...
%                  'MarkerSize',      4);
%         end
%     end
% 
%     % ===============================
%     %  Figure 2: loss vs global step
%     % ===============================
%     figure;
%     hold on;
% 
%     steps = (1:opts.n_steps)';
% 
%     % Plot, for each arm, only at descent steps (where that arm was updated)
%     for k = 1:K
%         idx = find(~isnan(loss_hist_selected(:, k)));
%         if ~isempty(idx)
%             plot(steps(idx), loss_hist_selected(idx, k), '-o', ...
%                  'Color',     opts.colors(k,:), ...
%                  'LineWidth', 1, ...
%                  'MarkerSize', 4);
%         end
%     end
% 
%     % Continuous red horizontal line at analytic minimum of loss
%     plot(steps, true_min_loss * ones(size(steps)), ...
%          'r-', 'LineWidth', 2);
% 
%     xlabel('Step');
%     ylabel('Loss l(a,b)');
%     title('Loss per initialization vs. steps (only when updated)');
%     grid on;
% 
%     % ============================
%     %  Build and return loss table
%     % ============================
%     if nargout >= 1
%         varNames = cell(1, K + 1);
%         varNames{1} = 'step';
%         for k = 1:K
%             varNames{k+1} = sprintf('loss_arm_%d', k);
%         end
%         loss_table = array2table([steps, loss_hist_all], ...
%                                  'VariableNames', varNames);
%     end
% end




function loss_table = mse_plot_ucb_trajectories(M, Y0, n_iters, opts)
%MSE_PLOT_UCB_TRAJECTORIES UCB-based Gradient Descent trajectories.
%
%   Now includes n_iters as a direct function argument.
%
%   Usage:
%       loss_table = mse_plot_ucb_trajectories(M, Y0, 3000, opts);
%
%   UCB selection over K initializations (arms) to minimize a non-convex
%   loss by gradient descent.
%
%   Each arm is a 2n-dimensional vector x = [a; b], where
%       a, b ∈ R^n.
%   The loss for one arm is
%       l(a,b) = sum_{j=1}^n L(a_j, b_j),
%   where M.L is the scalar loss L(a,b) and M.gradL its scalar gradient
%   [dL/da; dL/db].
%
% Inputs:
%   M         : model struct with fields M.L, M.gradL, optionally M.Lmin
%   Y0        : (2n×K) or (K×2n) initializations
%   n_iters   : total number of global UCB iterations
%   opts      : optional struct
%                   .eta
%                   .colors
%                   .linewidth
%                   .true_min_loss
%
% Output:
%   loss_table : table with global loss values at every iteration for all arms

    if nargin < 4, opts = struct(); end

    % ----- Defaults -----
    if ~isfield(opts, 'eta'),       opts.eta       = 0.01; end
    if ~isfield(opts, 'linewidth'), opts.linewidth = 1.5;  end

    % Allow backward compatibility: opts.n_steps is ignored if n_iters is given
    n_steps = n_iters;

    % ----- Initialize arms -----
    [dim, K] = size(Y0);
    if mod(dim, 2) ~= 0
        if mod(size(Y0, 2), 2) == 0
            Y0 = Y0.';
            [dim, K] = size(Y0);
        else
            error('Y0 must be (2n×K) or (K×2n).');
        end
    end
    n = dim/2;

    if ~isfield(opts, 'colors'), opts.colors = lines(K); end

    X = Y0;
    N = zeros(1, K);   % pull counts

    % Trajectory storage (for 2D plotting only)
    history = cell(1, K);
    for k = 1:K
        history{k} = X(:, k);
    end

    % Loss histories
    loss_hist_all      = nan(n_steps, K); % global iteration loss
    loss_hist_selected = nan(n_steps, K); % only when selected
    pull_hist_selected = nan(n_steps, K); % per-arm local pull index

    % ----- True minimum of loss -----
    if isfield(opts, 'true_min_loss')
        true_min_loss = opts.true_min_loss;
    elseif isfield(M, 'true_min_loss')
        true_min_loss = M.true_min_loss;
    elseif isfield(M, 'Lmin')
        true_min_loss = M.Lmin;
    else
        true_min_loss = 0;
    end

    % ----- Helpers -----
    function val = full_loss(x)
        a = x(1:n); b = x(n+1:2*n);
        val = 0;
        for j = 1:n
            val = val + M.L(a(j), b(j));
        end
    end

    function g = full_grad(x)
        a = x(1:n); b = x(n+1:2*n);
        ga = zeros(n,1); gb = zeros(n,1);
        for j = 1:n
            g_j = M.gradL(a(j), b(j));
            ga(j) = g_j(1); gb(j) = g_j(2);
        end
        g = [ga; gb];
    end

    % ===============================
    %   FIGURE 1: 2D TRAJECTORIES
    % ===============================
    figure; hold on;

    for t = 1:n_steps

        % ----- Compute UCB score for each arm -----
        if t == 1
            numerator = 0;
        else
            numerator = 2 * log(1 + t*(log(t))^2);
        end

        U = zeros(1, K);
        for k = 1:K
            f_val = full_loss(X(:, k));
            exploration = sqrt(numerator / (N(k)+1));
            U(k) = tanh(f_val) - exploration;
        end

        % ----- Select Arm -----
        [~, A_t] = min(U);

        % ----- Gradient step -----
        gA = full_grad(X(:, A_t));
        X(:, A_t) = X(:, A_t) - opts.eta * gA;

        % ----- Update pull count -----
        N(A_t) = N(A_t) + 1;

        % ----- Store trajectory -----
        history{A_t} = [history{A_t}, X(:, A_t)];

        % ----- Record losses -----
        for k = 1:K
            loss_hist_all(t, k) = full_loss(X(:, k));
        end
        loss_hist_selected(t, A_t) = loss_hist_all(t, A_t);
        pull_hist_selected(t, A_t) = N(A_t);
    end

    % Plot trajectories (first 2 dimensions)
    for k = 1:K
        traj = history{k};
        if size(traj, 2) > 1
            plot(traj(1,:), traj(2,:), '.-', ...
                'Color', opts.colors(k,:), ...
                'LineWidth', opts.linewidth);
        end
    end

    % ===============================
    %   FIGURE 2: LOSS vs PULL COUNT
    % ===============================
    figure; hold on;

    for k = 1:K
        idx = find(~isnan(loss_hist_selected(:, k)));
        if ~isempty(idx)
            x_local = pull_hist_selected(idx, k);
            y_loss  = loss_hist_selected(idx, k);
            plot(x_local, y_loss, '-o', ...
                'Color', opts.colors(k,:), 'LineWidth', 1);
        end
    end

    max_pulls = max(N);
    x_line = 0:max_pulls;
    plot(x_line, true_min_loss*ones(size(x_line)), ...
         'r-', 'LineWidth', 2);

    xlabel('Pull count of each arm');
    ylabel('Loss l(a,b)');
    title('Loss vs Pull Count for each Arm');
    grid on;

    % ===============================
    %   RETURN TABLE
    % ===============================
    steps = (1:n_steps).';
    varNames = [{'step'}, arrayfun(@(k)sprintf('loss_arm_%d',k),1:K,'UniformOutput',false)];
    loss_table = array2table([steps, loss_hist_all], 'VariableNames', varNames);

end
