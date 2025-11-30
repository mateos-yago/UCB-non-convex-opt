


function [y_next, h_used, info] = midpoint_reversible_step(F, JF, y, h_init, opts)
%MIDPOINT_REVERSIBLE_STEP Reversible adaptive implicit midpoint step.
%
%   [y_next, h_used, info] = midpoint_reversible_step(F, JF, y, h_init, opts)
%
%   Autonomous ODE: y' = F(y), y in R^n.
%
%   This performs ONE step using the implicit midpoint method:
%
%       y_{n+1} = y_n + h F( (y_n + y_{n+1}) / 2 )
%
%   with:
%       - strict Newton solve using analytic F and JF
%       - adaptive step size tied to midpoint geometry (curvature + speed)
%       - optional stiffness bound (OFF by default)
%       - accept/reject based ONLY on midpoint quantities
%       - exact time-reversibility for all accepted steps
%
%   Key opts fields:
%       .max_attempts     (default 10)
%       .max_newton_iter  (default 50)
%       .newton_abs_tol   (default 1e-12)
%       .newton_rel_tol   (default sqrt(eps))
%       .newton_damping   (default true)
%       .theta_stiff      (default 0.8)   % only used if use_stiffness_bound = true
%       .use_stiffness_bound (default false)  % *** IMPORTANT ***
%       .N_turns          (default 36)    % steps per 2π of tangent rotation
%       .gamma_kappa      (default 2.0)
%       .eta_shrink       (default 0.5)
%       .eta_accept       (default 0.9)
%       .h_min            (default 1e-12)
%       .h_max            (default Inf)
%       .kappa_floor      (default 1e-8)
%       .min_speed        (default 1e-8)
%       .debug            (default false)

    if nargin < 5, opts = struct(); end

    % --- Defaults ---
    if ~isfield(opts,'max_attempts'),        opts.max_attempts        = 10;         end
    if ~isfield(opts,'max_newton_iter'),     opts.max_newton_iter     = 50;         end
    if ~isfield(opts,'newton_abs_tol'),      opts.newton_abs_tol      = 1e-12;      end
    if ~isfield(opts,'newton_rel_tol'),      opts.newton_rel_tol      = sqrt(eps);  end
    if ~isfield(opts,'newton_damping'),      opts.newton_damping      = true;       end
    if ~isfield(opts,'theta_stiff'),         opts.theta_stiff         = 0.8;        end
    if ~isfield(opts,'use_stiffness_bound'), opts.use_stiffness_bound = false;      end
    if ~isfield(opts,'N_turns'),             opts.N_turns             = 36;         end
    if ~isfield(opts,'gamma_kappa'),         opts.gamma_kappa         = 2.0;        end
    if ~isfield(opts,'eta_shrink'),          opts.eta_shrink          = 0.5;        end
    if ~isfield(opts,'eta_accept'),          opts.eta_accept          = 0.9;        end
    if ~isfield(opts,'h_min'),               opts.h_min               = 1e-12;      end
    if ~isfield(opts,'h_max'),               opts.h_max               = Inf;        end
    if ~isfield(opts,'kappa_floor'),         opts.kappa_floor         = 1e-8;       end
    if ~isfield(opts,'min_speed'),           opts.min_speed           = 1e-8;       end
    if ~isfield(opts,'debug'),               opts.debug               = false;      end

    % Ensure column vector
    y = y(:);
    n = numel(y);
    I = eye(n);

    % Initial trial step (respect max)
    h = h_init;
    if abs(h) > opts.h_max
        h = sign(h) * opts.h_max;
    end

    % Target angle per step (for curvature)
    dtheta_target = 2*pi / opts.N_turns;

    % Info init
    info = struct();
    info.converged         = false;
    info.newton_iter       = NaN;
    info.res_norm          = NaN;
    info.step_norm         = NaN;
    info.attempts          = 0;
    info.rejected_attempts = 0;
    info.h_init            = h_init;
    info.h_final           = NaN;

    last_geom = [];

    % Main attempt loop: we may shrink |h| a few times
    for attempt = 1:opts.max_attempts
        info.attempts = attempt;

        % Guard against too small step
        if abs(h) < opts.h_min
            if opts.debug && ~isempty(last_geom)
                fprintf('[midstep] step_underflow at attempt=%d, |h|=%.3e < h_min=%.3e\n', ...
                        attempt, abs(h), opts.h_min);
                fprintf('[midstep]   last geom: Jnorm=%.3e, kappa_mid=%.3e, speed=%.3e, h_stiff=%.3e, h_kappa=%.3e, h_allowed=%.3e\n', ...
                        last_geom.Jnorm, last_geom.kappa_mid, last_geom.speed, ...
                        last_geom.h_stiff, last_geom.h_kappa, last_geom.h_allowed);
            end
            error('polyflow:midpoint_reversible_step:step_underflow', ...
                  'Step size underflow: |h| = %.3e < h_min = %.3e.', ...
                  abs(h), opts.h_min);
        end

        % --- Newton solve for implicit midpoint with step h ---
        [z, newton_info] = local_midpoint_newton(F, JF, y, h, opts, I);

        if ~newton_info.converged
            % Treat Newton failure as "step too big" – shrink and retry
            info.rejected_attempts = info.rejected_attempts + 1;
            h = opts.eta_shrink * h;
            if opts.debug
                fprintf('[midstep]   newton_fail: shrink -> |h|=%.3e\n', abs(h));
            end
            continue;
        end

        % Candidate next state
        y_cand = z;

        % Midpoint and geometry at midpoint
        m       = 0.5 * (y + y_cand);
        v_mid   = F(m);
        speed   = norm(v_mid);
        speed_eff = max(speed, opts.min_speed);

        A_mid   = JF(m);
        Jnorm   = norm(A_mid);

        % Curvature at midpoint (2D formula)
        kappa_mid = local_flow_curvature(F, JF, m);
        kappa_mid = max(kappa_mid, opts.kappa_floor);

        % --- Stiffness-based bound at midpoint ---
        use_stiff = true;
        if isfield(opts, 'use_stiffness_bound') && ~opts.use_stiffness_bound
            use_stiff = false;
        end
        
        if use_stiff
            if Jnorm > 0
                h_stiff = opts.theta_stiff * 2 / Jnorm;
            else
                h_stiff = opts.h_max;
            end
        else
            % No stiffness constraint: let curvature (and h_max) control h
            h_stiff = opts.h_max;
        end


        % --- Curvature-based bound at midpoint ---
        % allow optional kappa_floor, min_speed from opts
        kappa = kappa_mid;
        spd   = speed;
        if isfield(opts, 'kappa_floor')
            kappa = max(kappa, opts.kappa_floor);
        end
        if isfield(opts, 'min_speed')
            spd = max(spd, opts.min_speed);
        end
        
        if spd > 0 && kappa > 0
            kappa_eff = opts.gamma_kappa * kappa;
            h_kappa   = dtheta_target / (kappa_eff * spd);
        else
            h_kappa = opts.h_max;
        end

        % Allowed magnitude based on midpoint
        h_allowed = min([h_stiff, h_kappa, opts.h_max]);

        if opts.debug
            fprintf('[midstep] attempt=%d, |h|=%.3e, Jnorm=%.3e, kappa_mid=%.3e, speed=%.3e, h_stiff=%.3e, h_kappa=%.3e, h_allowed=%.3e\n', ...
                    attempt, abs(h), Jnorm, kappa_mid, speed_eff, h_stiff, h_kappa, h_allowed);
        end

        last_geom = struct('Jnorm', Jnorm, 'kappa_mid', kappa_mid, ...
                           'speed', speed_eff, 'h_stiff', h_stiff, ...
                           'h_kappa', h_kappa, 'h_allowed', h_allowed);

        % Symmetric acceptance condition: |h| <= eta_accept * h_allowed
        if abs(h) <= opts.eta_accept * h_allowed
            % Accept this step
            y_next           = y_cand;
            h_used           = h;
            info.converged   = true;
            info.newton_iter = newton_info.newton_iter;
            info.res_norm    = newton_info.res_norm;
            info.step_norm   = newton_info.step_norm;
            info.h_final     = h;
            return;
        else
            % Step too large given midpoint geometry; shrink and retry
            info.rejected_attempts = info.rejected_attempts + 1;
            h_old = h;
            h     = sign(h) * opts.eta_shrink * h_allowed;
            if opts.debug
                fprintf('[midstep]   shrink: new |h|=%.3e (from h_allowed=%.3e, old |h|=%.3e)\n', ...
                        abs(h), h_allowed, abs(h_old));
            end
            if abs(h) < opts.h_min
                if opts.debug
                    fprintf('[midstep] step_underflow after midpoint test: |h|=%.3e < h_min=%.3e\n', ...
                            abs(h), opts.h_min);
                    fprintf('[midstep]   last geom: Jnorm=%.3e, kappa_mid=%.3e, speed=%.3e, h_stiff=%.3e, h_kappa=%.3e, h_allowed=%.3e\n', ...
                            last_geom.Jnorm, last_geom.kappa_mid, last_geom.speed, ...
                            last_geom.h_stiff, last_geom.h_kappa, last_geom.h_allowed);
                end
                error('polyflow:midpoint_reversible_step:step_underflow', ...
                      'Step size underflow after midpoint test: |h| = %.3e < h_min = %.3e.', ...
                      abs(h), opts.h_min);
            end
        end
    end

    % If we get here, too many attempts
    error('polyflow:midpoint_reversible_step:max_attempts', ...
          'Exceeded max_attempts = %d without finding an acceptable step.', ...
          opts.max_attempts);
end

%==========================================================================
function [z, info] = local_midpoint_newton(F, JF, y, h, opts, I)
%LOCAL_MIDPOINT_NEWTON Solve z - y - h F((y+z)/2) = 0 by Newton.

    y = y(:);
    n = numel(y);

    % Predictor: explicit Euler
    k1 = F(y);
    z  = y + h * k1;

    base_scale = max(norm(y,2), 1.0);

    info.newton_iter = 0;
    info.res_norm    = NaN;
    info.step_norm   = NaN;
    info.converged   = false;

    for k = 1:opts.max_newton_iter
        % Midpoint
        y_mid = 0.5 * (y + z);
        F_mid = F(y_mid);

        % Residual
        R = z - y - h * F_mid;
        res_norm = norm(R, 2);

        % Dynamic tolerance
        scale = max([base_scale, norm(z,2), 1.0]);
        tol   = opts.newton_abs_tol + opts.newton_rel_tol * scale;

        if res_norm <= tol
            info.newton_iter = k - 1;
            info.res_norm    = res_norm;
            info.step_norm   = 0;
            info.converged   = true;
            return;
        end

        % Jacobian of residual
        JF_mid = JF(y_mid);
        J      = I - 0.5 * h * JF_mid;

        % Newton direction
        delta = -J \ R;
        step_norm = norm(delta, 2);

        if step_norm <= tol
            z = z + delta;
            info.newton_iter = k;
            info.res_norm    = res_norm;
            info.step_norm   = step_norm;
            info.converged   = true;
            return;
        end

        % Optional damping
        if opts.newton_damping
            lambda = 1.0;
            z_trial = z + delta;
            y_mid_trial = 0.5 * (y + z_trial);
            R_trial = z_trial - y - h * F(y_mid_trial);
            res_trial_norm = norm(R_trial, 2);

            damp_count = 0;
            while res_trial_norm > res_norm && damp_count < 10
                lambda = 0.5 * lambda;
                z_trial = z + lambda * delta;
                y_mid_trial = 0.5 * (y + z_trial);
                R_trial = z_trial - y - h * F(y_mid_trial);
                res_trial_norm = norm(R_trial, 2);
                damp_count = damp_count + 1;
            end

            z = z_trial;
            res_norm  = res_trial_norm;
            step_norm = lambda * step_norm;
        else
            z = z + delta;
        end

        base_scale = max([base_scale, norm(z,2)]);
    end

    % No convergence within max_newton_iter
    info.newton_iter = opts.max_newton_iter;
    info.res_norm    = res_norm;
    info.step_norm   = step_norm;
    info.converged   = false;
end

%==========================================================================
function kappa = local_flow_curvature(F, JF, y)
%LOCAL_FLOW_CURVATURE Curvature of y' = F(y) in R^2 at point y.
%
%   kappa = |v x a| / ||v||^3,  where v = F(y), a = JF(y)*v

    y = y(:);
    v = F(y);
    speed = norm(v);

    if speed == 0
        kappa = 0;
        return;
    end

    A = JF(y);
    a = A * v;

    if numel(v) < 2
        kappa = 0;
        return;
    end

    % 2D scalar cross product
    cross2d = v(1)*a(2) - v(2)*a(1);

    kappa = abs(cross2d) / max(speed^3, eps);
end
    