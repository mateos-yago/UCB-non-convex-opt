


function mse_plot_midpoint_trajectories(M, Y0, opts)
%MSE_PLOT_MIDPOINT_TRAJECTORIES  Continuous-time midpoint descent trajectories.
%
%   mse_plot_midpoint_trajectories(M, Y0, opts)
%
%   Inputs:
%       M    : model struct with analytic calculus:
%              M.gradL(a,b), M.hessL(a,b)
%       Y0   : 2×m or m×2 matrix of initial conditions (a,b) for each traj
%       opts : struct with optional fields:
%           .h          : time-step for midpoint (default 0.02)
%           .n_steps    : max number of steps per trajectory (default 2000)
%           .a_min, .a_max, .b_min, .b_max :
%                       : plotting box; if omitted, uses current axis
%           .color      : [r g b] for trajectories (default [0 0 0])
%           .linewidth  : line width (default 1.5)
%
%   This assumes the phase portrait (level sets, backbone, skeleton) has
%   already been drawn and simply overlays smooth midpoint trajectories.

    if nargin < 3
        opts = struct();
    end

    % --- Normalize Y0 to 2×m ----------------------------------------------
    if size(Y0,1) == 2
        Y0_2 = Y0;
    elseif size(Y0,2) == 2
        Y0_2 = Y0.';   % transpose to 2×m
    else
        error('polyflow:mse_plot_midpoint_trajectories:badY0', ...
              'Y0 must be 2×m or m×2.');
    end
    m = size(Y0_2, 2);

    % --- Defaults for options ---------------------------------------------
    if ~isfield(opts, 'h'),         opts.h         = 0.02;    end
    if ~isfield(opts, 'n_steps'),   opts.n_steps   = 2000;    end
    if ~isfield(opts, 'color'),     opts.color     = [0 0 1]; end
    if ~isfield(opts, 'linewidth'), opts.linewidth = 1.5;     end

    % Bounding box: either from opts or from current axis
    if isfield(opts, 'a_min') && isfield(opts, 'a_max') && ...
       isfield(opts, 'b_min') && isfield(opts, 'b_max')
        a_min = opts.a_min; a_max = opts.a_max;
        b_min = opts.b_min; b_max = opts.b_max;
    else
        ax = axis;   % [xmin xmax ymin ymax] of current figure
        a_min = ax(1); a_max = ax(2);
        b_min = ax(3); b_max = ax(4);
    end

    % --- Define gradient flow and Jacobian for implicit midpoint ----------
    % Gradient descent ODE: y' = F(y) = -∇L(a,b)
    F  = @(y) -M.gradL(y(1), y(2));
    JF = @(y) -M.hessL(y(1), y(2));

    % Midpoint solver options (simple but robust)
    mid_opts = struct();
    mid_opts.max_attempts    = 5;
    mid_opts.max_newton_iter = 20;
    mid_opts.newton_abs_tol  = 1e-10;
    mid_opts.newton_rel_tol  = 1e-8;
    mid_opts.newton_damping  = true;
    mid_opts.theta_stiff     = 0.9;
    mid_opts.N_turns         = 36;
    mid_opts.gamma_kappa     = 1.0;
    mid_opts.eta_shrink      = 0.5;
    mid_opts.eta_accept      = 0.9;
    mid_opts.h_min           = 1e-12;
    mid_opts.h_max           = opts.h;   % don’t grow past requested step

    hold_state = ishold;
    hold on;

    % --- Loop over initial conditions -------------------------------------
    for j = 1:m
        y = Y0_2(:, j);

        % Preallocate for a decent upper bound; we’ll only use 1:n_used
        Y = zeros(2, opts.n_steps + 1);
        Y(:,1) = y;
        n_used = 1;

        for n = 1:opts.n_steps
            % Take one implicit midpoint step in gradient-flow time
            try
                [y_new, ~, info_step] = polyflow.midpoint_reversible_step(F, JF, y, opts.h, mid_opts);
            catch ME
                % If the midpoint solver underflows or fails to find a step,
                % we just stop this trajectory gracefully.
                if contains(ME.identifier, 'step_underflow') || ...
                   contains(ME.identifier, 'max_attempts')
                    break;
                else
                    rethrow(ME);
                end
            end

            if ~info_step.converged
                % Newton did not converge in the allotted iterations
                break;
            end

            y = y_new;
            n_used = n_used + 1;
            Y(:, n_used) = y;

            % Stop if we leave the plotting window
            if y(1) < a_min || y(1) > a_max || y(2) < b_min || y(2) > b_max
                break;
            end
        end

        % Plot only the filled portion of Y
        plot(Y(1,1:n_used), Y(2,1:n_used), '-', ...
             'Color', opts.color, 'LineWidth', opts.linewidth);
    end

    if ~hold_state
        hold off;
    end
end
