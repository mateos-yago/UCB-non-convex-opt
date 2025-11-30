


function mse_plot_adamw_trajectories(M, Y0, opts)
%MSE_PLOT_ADAMW_TRAJECTORIES  Overlay AdamW trajectories on an existing phase portrait.
%
%   Same interface and behavior as mse_plot_sgd_trajectories.
%   This uses decoupled weight decay (W → (1 - eta*wd)*W) + Adam moments.
%
%   Uses step size schedules if opts.eta is a function handle:
%       eta_k = opts.eta(k)
%
%   Inputs identical to mse_plot_sgd_trajectories, with additional:
%       .beta1  (default 0.9)
%       .beta2  (default 0.999)
%       .eps    (default 1e-8)
%       .weight_decay (default 0)  % decoupled weight decay


    if nargin < 3, opts = struct(); end

    % defaults
    if ~isfield(opts,'eta'), opts.eta = 0.01; end
    if ~isfield(opts,'beta1'), opts.beta1 = 0.9; end
    if ~isfield(opts,'beta2'), opts.beta2 = 0.999; end
    if ~isfield(opts,'eps'),   opts.eps   = 1e-8; end
    if ~isfield(opts,'weight_decay'), opts.weight_decay = 0; end
    if ~isfield(opts,'noise_std'), opts.noise_std = 0; end
    if ~isfield(opts,'n_steps'),  opts.n_steps = 1000; end
    if ~isfield(opts,'color'),    opts.color = [0 1 1]; end
    if ~isfield(opts,'linewidth'),opts.linewidth = 1.5; end
    if ~isfield(opts,'marker_start'), opts.marker_start = true; end
    if ~isfield(opts,'marker_end'),   opts.marker_end   = false; end

    use_box = isfield(opts,'a_min') && isfield(opts,'a_max') && ...
              isfield(opts,'b_min') && isfield(opts,'b_max');

    % convert eta to a handle if constant
    if ~isa(opts.eta,'function_handle')
        eta0 = opts.eta;
        opts.eta = @(k) eta0;
    end

    [~, n_traj] = size(Y0);
    hold on;

    for j = 1:n_traj
        y = Y0(:,j);
        m = zeros(2,1);
        v = zeros(2,1);

        traj = zeros(2, opts.n_steps+1);
        traj(:,1) = y;
        k_final = opts.n_steps;

        for k = 1:opts.n_steps
            alpha = opts.eta(k);
            g = M.gradL(y(1), y(2));
            if opts.noise_std > 0
                g = g + opts.noise_std * randn(2,1);
            end

            % Adam update
            m = opts.beta1*m + (1-opts.beta1)*g;
            v = opts.beta2*v + (1-opts.beta2)*(g.^2);

            m_hat = m / (1 - opts.beta1^k);
            v_hat = v / (1 - opts.beta2^k);

            step = -alpha * m_hat ./ (sqrt(v_hat) + opts.eps);

            % Decoupled weight decay (AdamW)
            y = (1 - alpha * opts.weight_decay) * y + step;

            traj(:,k+1) = y;

            if use_box
                if y(1)<opts.a_min || y(1)>opts.a_max || ...
                   y(2)<opts.b_min || y(2)>opts.b_max
                    k_final = k;
                    traj = traj(:,1:k_final+1);
                    break;
                end
            end
        end

        plot(traj(1,:), traj(2,:), '-', 'Color',opts.color, 'LineWidth',opts.linewidth);

        if opts.marker_start
            plot(traj(1,1), traj(2,1),'o','MarkerFaceColor',opts.color,'MarkerEdgeColor','none','MarkerSize',4);
        end
        if opts.marker_end
            plot(traj(1,end), traj(2,end),'s','MarkerFaceColor',opts.color,'MarkerEdgeColor','none','MarkerSize',4);
        end
    end
end
