


function [T, Y, H, stats] = midpoint_reversible_solve(F, JF, y0, tspan, opts)
%MIDPOINT_REVERSIBLE_SOLVE Reversible implicit midpoint integration over tspan.
%
%   [T, Y, H, stats] = midpoint_reversible_solve(F, JF, y0, tspan, opts)
%
%   Integrates the autonomous ODE:
%       y' = F(y)
%   using the reversible adaptive implicit midpoint *single-step* method:
%       polyflow.midpoint_reversible_step
%
%   This driver:
%       - supports forward or backward integration depending on tspan(2)-tspan(1)
%       - keeps step sizes consistent with time direction
%       - does NOT add extra asymmetric adaptivity: the next step's h_init
%         is simply the previous accepted h_used (time-symmetric choice)
%       - preserves anadromicity at the step level; multi-step reversibility
%         is governed by using the same scheme and step logic in reverse.
%
%   Inputs:
%       F     : @(y) -> (n x 1)
%       JF    : @(y) -> (n x n)
%       y0    : initial state (n x 1 or 1 x n)
%       tspan : [t0, tf] (t0 < tf or t0 > tf both allowed)
%       opts  : struct with optional fields:
%           .h_init        (default: (tf - t0)/100 in magnitude)
%           .h_min         (passed to midpoint_reversible_step)
%           .h_max         (passed to midpoint_reversible_step)
%           .max_attempts  (per-step; default 10)
%           .max_newton_iter, .newton_abs_tol, .newton_rel_tol, .newton_damping
%           .theta_stiff, .N_turns, .gamma_kappa
%           .eta_shrink, .eta_accept
%           .max_steps     (default: 100000)
%
%   Outputs:
%       T     : 1 x N time vector (monotone from t0 toward tf)
%       Y     : n x N state matrix
%       H     : 1 x (N-1) vector of accepted step sizes (same sign as tf-t0)
%       stats : struct:
%           .n_steps          : number of accepted steps
%           .t0, .tf          : endpoints
%           .direction        : +1 (forward) or -1 (backward)
%           .total_rejects    : sum of rejected_attempts over all steps
%           .last_step_info   : info struct from last step

    if nargin < 5, opts = struct(); end

    t0 = tspan(1);
    tf = tspan(2);

    if t0 == tf
        T = t0;
        Y = y0(:);
        H = [];
        stats = struct('n_steps', 0, ...
                       't0', t0, 'tf', tf, ...
                       'direction', 0, ...
                       'total_rejects', 0, ...
                       'last_step_info', []);
        return;
    end

    % Time direction and absolute span
    direction = sign(tf - t0);           % +1 forward, -1 backward
    Tspan_abs = abs(tf - t0);

    % Default step size magnitude
    if ~isfield(opts, 'h_init')
        opts.h_init = Tspan_abs / 10;
    end

    % Max steps safeguard
    if ~isfield(opts, 'max_steps')
        opts.max_steps = 1e5;
    end

    % Ensure column initial state
    y = y0(:);
    n = numel(y);

    % Allocate
    T = zeros(1, opts.max_steps);
    Y = zeros(n, opts.max_steps);
    H = zeros(1, opts.max_steps-1);

    T(1) = t0;
    Y(:,1) = y;

    % Initial step with correct sign
    h = direction * abs(opts.h_init);

    k = 1;
    total_rejects = 0;
    last_info = struct();

    t = t0;

    while true
        % Check if we are within floating-point epsilon of tf
        if direction > 0
            if t >= tf - eps(max(1,abs(tf)))
                break;
            end
        else
            if t <= tf + eps(max(1,abs(tf)))
                break;
            end
        end

        if k >= opts.max_steps
            error('polyflow:midpoint_reversible_solve:max_steps', ...
                  'Exceeded max_steps = %d.', opts.max_steps);
        end

        % Adjust h so as not to overshoot tf
        if direction > 0
            if t + h > tf
                h = tf - t;
            end
        else
            if t + h < tf
                h = tf - t;
            end
        end

        % Call the reversible stepper
        [y_next, h_used, info_step] = polyflow.midpoint_reversible_step(F, JF, y, h, opts);

        % Update time and state
		t_new = t + h_used;

        % Record
        k = k + 1;
        T(k)   = t_new;
        Y(:,k) = y_next;
        H(k-1) = h_used;

        % Prepare for next step
        t = t_new;
        y = y_next;
        h = h_used;   % symmetric choice: reuse same step as initial guess

        total_rejects = total_rejects + info_step.rejected_attempts;
        last_info = info_step;
    end

    % Trim outputs
    T = T(1:k);
    Y = Y(:,1:k);
    H = H(1:k-1);

    stats = struct();
    stats.n_steps       = k-1;
    stats.t0            = t0;
    stats.tf            = tf;
    stats.direction     = direction;
    stats.total_rejects = total_rejects;
    stats.last_step_info = last_info;
end
