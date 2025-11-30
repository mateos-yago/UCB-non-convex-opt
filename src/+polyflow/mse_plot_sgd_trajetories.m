



function mse_plot_sgd_trajectories(M, Y0, opts)
%MSE_PLOT_SGD_TRAJECTORIES  Plot discrete SGD / SGD+momentum paths on an existing phase portrait.
%
%   polyflow.mse_plot_sgd_trajectories(M, Y0, opts)
%
%   Inputs
%   ------
%   M    : model struct with at least
%              M.gradL(a,b) -> [dL/da; dL/db]
%          (same convention as used in the phase portrait code)
%
%   Y0   : 2 x m matrix of initial conditions.
%          Each column is [a0; b0] for one trajectory.
%
%   opts : struct with optional fields
%
%       % basic SGD / momentum
%       .eta        : step size (default 0.01)
%       .beta       : momentum parameter in [0,1) (default 0)
%                     update:
%                       v_{k+1} = beta * v_k - eta * gradL(x_k)
%                       x_{k+1} = x_k + v_{k+1}
%
%       .n_steps    : max number of iterations per trajectory (default 200)
%
%       % optional "stochastic" noise on the gradient
%       .noise_std  : standard deviation of isotropic Gaussian noise added
%                     to gradL at each step (default 0, i.e. pure GD)
%
%       % plotting
%       .color      : RGB triple or line spec (default [0 0 0])
%       .linewidth  : line width for trajectories (default 1.5)
%       .marker_start : logical, plot a marker at the starting point (default true)
%       .marker_end   : logical, plot a marker at the final point   (default false)
%
%       % optional stopping box (typically the phase portrait window)
%       .a_min, .a_max, .b_min, .b_max : if all present, we stop a
%           trajectory once it leaves this box.
%
%   Behavior
%   --------
%   Assumes an existing figure/axes already contain the phase portrait
%   (log-level contours, backbone, Morse skeleton, etc.).
%   This function just does "hold on" and overlays discrete SGD paths.

    if nargin < 3
        opts = struct();
    end

    % --- defaults ---
    if ~isfield(opts, 'eta'),         opts.eta         = 0.01;     end
    if ~isfield(opts, 'beta'),        opts.beta        = 0.0;      end
    if ~isfield(opts, 'n_steps'),     opts.n_steps     = 200;      end
    if ~isfield(opts, 'noise_std'),   opts.noise_std   = 0.0;      end
    if ~isfield(opts, 'color'),       opts.color       = [1 0 1];  end
    if ~isfield(opts, 'linewidth'),   opts.linewidth   = 1.5;      end
    if ~isfield(opts, 'marker_start'),opts.marker_start = true;    end
    if ~isfield(opts, 'marker_end'),  opts.marker_end   = false;   end

    % optional bounding box
    use_box = isfield(opts, 'a_min') && isfield(opts, 'a_max') && ...
              isfield(opts, 'b_min') && isfield(opts, 'b_max');

    [n_dim, n_traj] = size(Y0);
    if n_dim ~= 2
        error('mse_plot_sgd_trajectories:dimMismatch', ...
              'Y0 must be 2 x m (each column is [a;b]).');
    end

    hold_state = ishold;
    hold on;

    eta  = opts.eta;
    beta = opts.beta;
    sig  = opts.noise_std;

    for j = 1:n_traj
        y = Y0(:, j);           % current (a,b)
        v = zeros(2,1);         % momentum

        % preallocate trajectory (we may truncate if we hit the box edge)
        traj = zeros(2, opts.n_steps + 1);
        traj(:,1) = y;

        k_final = opts.n_steps; % will update if we break early

        for k = 1:opts.n_steps
            % gradient at current point
            g = M.gradL(y(1), y(2));

            % optional "SGD-style" noise
            if sig > 0
                g = g + sig * randn(size(g));
            end

            % momentum update:
            %   v_{k+1} = beta * v_k - eta * gradL(x_k)
            %   x_{k+1} = x_k + v_{k+1}
            v = beta * v - eta * g;
            y = y + v;

            traj(:,k+1) = y;

            % if a bounding box is specified, stop when we leave it
            if use_box
                if y(1) < opts.a_min || y(1) > opts.a_max || ...
                   y(2) < opts.b_min || y(2) > opts.b_max
                    k_final = k;  % last valid index is k
                    traj = traj(:,1:k_final+1);
                    break;
                end
            end
        end

        % plot this trajectory
        plot(traj(1,:), traj(2,:), '-', ...
            'Color', opts.color, ...
            'LineWidth', opts.linewidth);

        % start / end markers
        if opts.marker_start
            plot(traj(1,1), traj(2,1), 'o', ...
                'MarkerFaceColor', opts.color, ...
                'MarkerEdgeColor', 'none', ...
                'MarkerSize', 4);
        end

        if opts.marker_end
            plot(traj(1,end), traj(2,end), 's', ...
                'MarkerFaceColor', opts.color, ...
                'MarkerEdgeColor', 'none', ...
                'MarkerSize', 4);
        end
    end

    if ~hold_state
        hold off;
    end
end
