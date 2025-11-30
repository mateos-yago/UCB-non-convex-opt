


function mse_plot_separatrices(M, info, opts)
%MSE_PLOT_SEPARATRICES Trace stable/unstable manifolds of saddles.
%
%   This version:
%       - Uses analytic Hessians/Jacobians only (no numeric Jacobian).
%       - Uses implicit midpoint with a fixed stepsize h_sep.
%       - Does NOT depend on polyflow.sundman_step or curvature warp.
%
%   Convention (gradient flow):
%       xdot = -grad L(a,b)
%       H = ∇²L(a*,b*)
%       λ > 0  => stable direction for descent
%       λ < 0  => unstable direction for descent
%
%   We draw:
%       - UNSTABLE manifolds of the descent flow (green).
%       - STABLE manifolds, computed as UNSTABLE manifolds of the ascent
%         flow xdot = +grad L (red).
%
%   Inputs:
%       M    : model struct with analytic calculus:
%              M.L(a,b), M.gradL(a,b), M.hessL(a,b)
%       info : struct from mse_morse_info, with info.fixed_points(k).type
%              == 'saddle' and fields .a, .b
%       opts : options struct
%              .h_sep      = stepsize for separatrices   (default 0.02)
%              .steps_sep  = max # steps per branch      (default 10000)
%              .sep_eps    = initial offset along eigvec (default 1e-3)
%              .a_min/max, .b_min/max for plot bounds
%              .newton     = struct for implicit_midpoint_step
%
%   Requires:
%       polyflow.implicit_midpoint_step(f, t, y, h, newton_opts, Jfun)

    if nargin < 3, opts = struct(); end

    % --- Defaults --------------------------------------------------------
    if ~isfield(opts, 'h_sep'),      opts.h_sep      = 0.005;    end
    if ~isfield(opts, 'steps_sep'),  opts.steps_sep  = 10000;   end
    if ~isfield(opts, 'sep_eps'),    opts.sep_eps    = 1e-3;    end
    if ~isfield(opts, 'a_min'),      opts.a_min      = -6;      end
    if ~isfield(opts, 'a_max'),      opts.a_max      =  6;      end
    if ~isfield(opts, 'b_min'),      opts.b_min      = -6;      end
    if ~isfield(opts, 'b_max'),      opts.b_max      =  6;      end
    if ~isfield(opts, 'newton'),     opts.newton     = struct(); end

    % Newton options: analytic only
    newton_opts = opts.newton;
    if ~isfield(newton_opts, 'max_iter'),      newton_opts.max_iter      = 12;    end
    if ~isfield(newton_opts, 'tol'),           newton_opts.tol           = 1e-10; end
    % very important: FORCE analytic Jacobian, no numeric fallback
    newton_opts.allow_numeric = false;

    if ~isfield(info, 'fixed_points') || isempty(info.fixed_points)
        return;
    end

    fps = info.fixed_points;

    % Base gradient flow objects
    gradL = @(y) M.gradL(y(1), y(2));     % 2×1 gradient
    hessL = @(y) M.hessL(y(1), y(2));     % 2×2 Hessian

    % descent: xdot = -gradL
    base_descent = @(~,y) -gradL(y);
    % ascent:  xdot = +gradL
    base_ascent  = @(~,y)  gradL(y);

    hold_state = ishold;
    hold on;

    % =====================================================================
    % Loop over saddle points
    % =====================================================================
    for k = 1:numel(fps)
        if ~isfield(fps(k), 'type') || ~strcmp(fps(k).type, 'saddle')
            continue;
        end

        y0 = [fps(k).a; fps(k).b];

        % Hessian and eigendecomposition at the saddle
        H = hessL(y0);
        H = 0.5*(H + H.');   % enforce symmetry numerically
        [V,D]    = eig(H);
        lambdas  = diag(D);

        pos_idx = find(lambdas > 0);
        neg_idx = find(lambdas < 0);

        if numel(pos_idx) ~= 1 || numel(neg_idx) ~= 1
            % Not a simple 2D Morse saddle; skip.
            continue;
        end

        v_stable   = V(:, pos_idx(1));  % λ>0, stable for descent (unstable for ascent)
        v_unstable = V(:, neg_idx(1));  % λ<0, unstable for descent

        % =================================================================
        % UNSTABLE manifolds of descent (green)
        % =================================================================
        eps_unstable = choose_eps_opposite_sides(M, y0, v_unstable, opts.sep_eps);

        for sign_dir = [-1 1]
            if sign_dir > 0
                y = y0 + eps_unstable * v_unstable;
            else
                y = y0 - eps_unstable * v_unstable;
            end

            yy = zeros(2, opts.steps_sep);
            yy(:,1) = y;
            t = 0;

            for n = 2:opts.steps_sep
                % analytic Jacobian for descent: J_f = -H
                Jfun_descent = @(t_mid, y_mid) -hessL(y_mid); %#ok<NASGU,INUSD>
                y = polyflow.implicit_midpoint_step( ...
                        base_descent, t, y, opts.h_sep, newton_opts, Jfun_descent);

                yy(:,n) = y;
                t = t + opts.h_sep;

                % Stop once we leave the plotting window with a small margin.
                if y(1) < opts.a_min-1 || y(1) > opts.a_max+1 || ...
                   y(2) < opts.b_min-1 || y(2) > opts.b_max+1
                    yy = yy(:,1:n);
                    break;
                end
            end

            plot(yy(1,:), yy(2,:), 'g-', 'LineWidth', 2);
            plot(yy(1,1), yy(2,1), 'gv', 'LineWidth', 2);
        end

        % =================================================================
        % STABLE manifolds (ascent unstable) (red)
        % =================================================================
        eps_stable = choose_eps_opposite_sides(M, y0, v_stable, opts.sep_eps);

        for sign_dir = [-1 1]
            if sign_dir > 0
                y = y0 + eps_stable * v_stable;
            else
                y = y0 - eps_stable * v_stable;
            end

            yy = zeros(2, opts.steps_sep);
            yy(:,1) = y;
            t = 0;

            for n = 2:opts.steps_sep
                % analytic Jacobian for ascent: J_f = +H
                Jfun_ascent = @(t_mid, y_mid) hessL(y_mid); %#ok<NASGU,INUSD>
                y = polyflow.implicit_midpoint_step( ...
                        base_ascent, t, y, opts.h_sep, newton_opts, Jfun_ascent);

                yy(:,n) = y;
                t = t + opts.h_sep;

                % Stop once we leave the plotting window with a small margin.
                if y(1) < opts.a_min-1 || y(1) > opts.a_max+1 || ...
                   y(2) < opts.b_min-1 || y(2) > opts.b_max+1
                    yy = yy(:,1:n);
                    break;
                end
            end

            plot(yy(1,:), yy(2,:), 'r-', 'LineWidth', 2);
            plot(yy(1,1), yy(2,1), 'rv', 'LineWidth', 2);
            yy(:, [1 end])
        end
    end

    if ~hold_state
        hold off;
    end
end

% -------------------------------------------------------------------------
function eps_use = choose_eps_opposite_sides(M, y0, v, eps0)
%CHOOSE_EPS_OPPOSITE_SIDES Choose eps so that y0 ± eps*v lie on opposite
% sides of the backbone, as measured by the sign of dL/da.

    eps_use   = eps0;
    max_tries = 6;  % up to 64x original eps

    for attempt = 1:max_tries
        y_plus  = y0 + eps_use * v;
        y_minus = y0 - eps_use * v;

        g_plus  = M.gradL(y_plus(1),  y_plus(2));
        g_minus = M.gradL(y_minus(1), y_minus(2));

        dLa_plus  = g_plus(1);   % ∂L/∂a at y_plus
        dLa_minus = g_minus(1);  % ∂L/∂a at y_minus

        if dLa_plus * dLa_minus < 0
            % They lie on opposite sides of backbone w.r.t. dL/da
            return;
        else
            eps_use = 2*eps_use;  % try a larger offset
        end
    end

    % If we never get opposite signs, we just return the last eps_use.
end
