function mse_plot_ucb_trajectories(M, Y0, opts)
%MSE_PLOT_UCB_TRAJECTORIES UCB-based Gradient Descent trajectories.
%
%   Implements the algorithm from Kartzman (2025):
%   Selects among K arms (initializations) to minimize f(x) using a
%   Lower Confidence Bound (LCB) strategy to balance exploration/exploitation.
%
%   Inputs:
%       M    : Model struct with M.L (loss) and M.gradL (gradient)
%       Y0   : 2xK matrix of initial conditions (K arms)
%       opts : struct with optional fields:
%           .eta        : learning rate (default 0.01)
%           .n_steps    : total time horizon T (default 1000)
%           .linewidth  : line width for trajectories (default 1.5)
%           .colors     : Kx3 matrix of colors (default: lines map)
%
%   The logic follows:
%     1. Calculate UCB statistic: U_k(t) = tanh(f(x)) - Confidence
%     2. Select arm A_t = argmin U_k(t)
%     3. Update arm A_t via Gradient Descent
%     4. Increment count N_{A_t}

    if nargin < 3, opts = struct(); end

    % --- Default Options ---
    if ~isfield(opts, 'eta'),       opts.eta       = 0.01; end
    if ~isfield(opts, 'n_steps'),   opts.n_steps   = 1000; end
    if ~isfield(opts, 'linewidth'), opts.linewidth = 1.5;  end

    % --- Initialize Arms ---
    % Ensure Y0 is 2xK
    [dim, K] = size(Y0);
    if dim ~= 2
        if size(Y0, 2) == 2
            Y0 = Y0.';
            K = dim;
        else
            error('Y0 must be 2xK matrix of initial conditions.');
        end
    end

    if ~isfield(opts, 'colors'), opts.colors = lines(K); end

    X = Y0;                % Current positions of all K arms (2xK)
    N = zeros(1, K);       % Pull counts N_k(t) for all K arms
    
    % Pre-allocate history for plotting (cell array because lengths vary)
    history = cell(1, K);
    for k = 1:K
        history{k} = X(:, k); 
    end

    hold on;

    % --- Main Optimization Loop (t = 1 to T) ---
    for t = 1:opts.n_steps
        
        % 1. Calculate UCB Statistic for each arm k 
        % U_k(t) = tanh(f(x)) - sqrt( 2*log(1 + t*log^2(t)) / (N_k(t)+1) )
        
        U = zeros(1, K);
        
        % Calculate the time-dependent numerator term
        if t == 1
            % log(1) is 0, avoid any numerical noise
            numerator = 0;
        else
            % Note: log() in MATLAB is natural log (ln)
            term = 1 + t * (log(t))^2;
            numerator = 2 * log(term);
        end
        
        for k = 1:K
            % Evaluate objective f = L(a,b)
            f_val = M.L(X(1,k), X(2,k));
            
            % Calculate exploration bonus
            exploration = sqrt( numerator / (N(k) + 1) );
            
            % Compute UCB (technically LCB for minimization)
            U(k) = tanh(f_val) - exploration;
        end
        
        % 2. Select Arm 
        [~, A_t] = min(U);
        
        % 3. Descent Step on Selected Arm 
        % x(t+1) = x(t) - alpha * grad(x(t))
        grad = M.gradL(X(1, A_t), X(2, A_t));
        X(:, A_t) = X(:, A_t) - opts.eta * grad;
        
        % 4. Update Count 
        N(A_t) = N(A_t) + 1;
        
        % Store new position for plotting
        history{A_t} = [history{A_t}, X(:, A_t)];
    end

    % --- Visualization ---
    for k = 1:K
        traj = history{k};
        
        % Only plot if the arm moved at least once
        if size(traj, 2) > 1
            plot(traj(1,:), traj(2,:), '.-', ...
                 'Color', opts.colors(k,:), ...
                 'LineWidth', opts.linewidth, ...
                 'MarkerSize', 8);
                 
            % Optional: Mark the final position
            plot(traj(1,end), traj(2,end), 'o', ...
                 'Color', opts.colors(k,:), ...
                 'MarkerFaceColor', opts.colors(k,:), ...
                 'MarkerSize', 4);
        end
    end

    % Do not call hold off, to allow overlay on existing phase portrait
end