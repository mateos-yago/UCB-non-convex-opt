


function mse_plot_morse_skeleton_reversible(M, info, opts)
%MSE_PLOT_MORSE_SKELETON_REVERSIBLE
%   Debug-focused version:
%       - Trace stable/unstable manifolds of saddles using
%         reversible implicit midpoint.
%       - NO linear/backbone completion (pure integration).
%       - On termination of each branch, prints:
%           * reason
%           * y
%           * gradL(y)
%           * HessL(y)
%           * curvature kappa(F,JF,y)
%
%   Inputs:
%       M    : model struct with analytic calculus:
%              M.L(a,b), M.gradL(a,b), M.hessL(a,b), M.a_star(b)
%       info : struct from mse_morse_info, with info.fixed_points(k)
%              having fields .a, .b, .type (e.g. 'saddle', 'min').
%       opts : options struct (all fields optional):
%
%           Integration:
%             .h_init        (default 0.02)    initial step size
%             .h_min         (default 1e-12)   minimum step size
%             .max_steps_sep (default 4000)    max steps per branch
%             .sep_eps       (default 1e-3)    initial offset from saddle
%
%           Shaping:
%             .use_shaping   (default false)   (off by default for debugging)
%             .shape_K       (default 1)
%
%           Plot window:
%             .a_min, .a_max (default -6, 6)
%             .b_min, .b_max (default -6, 6)
%
%           Fixed-point proximity:
%             .R_fp_min      (default 0.1)     distance to "hit" a fixed point
%
%   Requires:
%       polyflow.midpoint_reversible_step(F, JF, y, h, step_opts)
%       polyflow.flow_curvature(F, JF, y)

    if nargin < 3, opts = struct(); end

    % ---- defaults -------------------------------------------------------
    if ~isfield(opts,'h_init'),        opts.h_init        = 0.02;   end
    if ~isfield(opts,'h_min'),         opts.h_min         = 1e-15;  end
    if ~isfield(opts,'max_steps_sep'), opts.max_steps_sep = 4000;   end
    if ~isfield(opts,'sep_eps'),       opts.sep_eps       = 1e-3;   end

    if ~isfield(opts,'a_min'),         opts.a_min         = -6;     end
    if ~isfield(opts,'a_max'),         opts.a_max         =  6;     end
    if ~isfield(opts,'b_min'),         opts.b_min         = -6;     end
    if ~isfield(opts,'b_max'),         opts.b_max         =  6;     end

    if ~isfield(opts,'use_shaping'),   opts.use_shaping   = false;  end
    if ~isfield(opts,'shape_K'),       opts.shape_K       = 1;      end

    if ~isfield(opts,'R_fp_min'),      opts.R_fp_min      = 0.1;    end

    if ~isfield(info,'fixed_points') || isempty(info.fixed_points)
        return;
    end

    % Base gradient/Hessian
    gradL = @(y) M.gradL(y(1), y(2));
    hessL = @(y) M.hessL(y(1), y(2));

    % Flow for integration: shaped or unshaped
    if opts.use_shaping
        Ms = polyflow.build_shaped_model(M, opts.shape_K);
        F_shape  = @(y) Ms.F(y);
        JF_shape = @(y) Ms.JF(y);

        % Descent and ascent flows (shaped)
        Fd  = @(y) -F_shape(y);
        JFd = @(y) -JF_shape(y);
        Fa  = @(y)  F_shape(y);
        JFa = @(y)  JF_shape(y);
    else
        % Raw gradient flows
        Fd  = @(y) -gradL(y);
        JFd = @(y) -hessL(y);
        Fa  = @(y)  gradL(y);
        JFa = @(y)  hessL(y);
    end

    % Stepper options
    step_opts = struct('h_min', opts.h_min);

    fps  = info.fixed_points;
    n_fp = numel(fps);

    % Precompute fixed point positions and types
    fp_pos  = zeros(2, n_fp);
    fp_type = cell(1, n_fp);
    for k = 1:n_fp
        fp_pos(:,k) = [fps(k).a; fps(k).b];
        if isfield(fps(k),'type')
            fp_type{k} = fps(k).type;
        else
            fp_type{k} = 'unknown';
        end
    end

    hold_state = ishold;
    hold on;

    % =====================================================================
    % Loop over saddle points
    % =====================================================================
    for k = 1:n_fp
        if ~strcmp(fp_type{k}, 'saddle')
            continue;
        end

        y0 = fp_pos(:,k);

        % Hessian eigendecomposition at saddle (unshaped)
        H = hessL(y0);
        H = 0.5*(H + H.');
        [V,D] = eig(H);
        lam   = diag(D);

        pos = find(lam > 0);
        neg = find(lam < 0);
        if numel(pos) ~= 1 || numel(neg) ~= 1
            % Not a simple Morse saddle in 2D; skip.
            continue;
        end

        v_stable   = V(:,pos);   % stable for descent / unstable for ascent
        v_unstable = V(:,neg);   % unstable for descent

        % =================================================================
        % UNSTABLE manifold of descent (green)
        % =================================================================
        eps_u = choose_eps_opposite_sides(M, y0, v_unstable, opts.sep_eps);

        for sgn = [-1 1]
            y_init = y0 + sgn * eps_u * v_unstable;
            [Y_branch, term_reason] = trace_branch_with_debug(Fd, JFd, M, ...
                                          y0, k, y_init, ...
                                          fp_pos, fp_type, opts, step_opts); %#ok<ASGLU>
            if size(Y_branch,2) > 1
                plot(Y_branch(1,:), Y_branch(2,:), 'g-','LineWidth',1.8);
                plot(Y_branch(1,1), Y_branch(2,1), 'gv','MarkerFaceColor','g');
            end
        end

        % =================================================================
        % STABLE manifold of descent = UNSTABLE manifold of ascent (red)
        % =================================================================
        eps_s = choose_eps_opposite_sides(M, y0, v_stable, opts.sep_eps);

        for sgn = [-1 1]
            y_init = y0 + sgn * eps_s * v_stable;
            [Y_branch, term_reason] = trace_branch_with_debug(Fa, JFa, M, ...
                                          y0, k, y_init, ...
                                          fp_pos, fp_type, opts, step_opts); %#ok<ASGLU>
            if size(Y_branch,2) > 1
                plot(Y_branch(1,:), Y_branch(2,:), 'r-','LineWidth',1.8);
                plot(Y_branch(1,1), Y_branch(2,1), 'rv','MarkerFaceColor','r');
            end
        end
    end

    if ~hold_state
        hold off;
    end
end

% -------------------------------------------------------------------------
function [Y, term_reason] = trace_branch_with_debug(F, JF, M, ...
                                        y_sad, idx_sad, y_init, ...
                                        fp_pos, fp_type, opts, step_opts)
%TRACE_BRANCH_WITH_DEBUG
%   Step-by-step reversible midpoint integration of one branch.
%   Termination reasons:
%       - 'step_underflow'
%       - 'left_plot_box'
%       - 'near_fixed_point'
%       - 'max_steps'
%
%   On termination, prints debug info.

    max_steps = opts.max_steps_sep;

    Y = zeros(2, max_steps);
    Y(:,1) = y_init;
    y = y_init;
    h = opts.h_init;
    n = 1;

    term_reason = 'max_steps';  % default; overwritten on early exit

    while n < max_steps
        % One reversible midpoint step
        try
            [y_next, h_used, info_step] = polyflow.midpoint_reversible_step(F, JF, y, h, step_opts); %#ok<NASGU>
        catch ME
            if contains(ME.identifier, 'step_underflow') || ...
               contains(ME.message, 'Step size underflow')
                Y = Y(:,1:n);
                term_reason = 'step_underflow';
                report_termination(term_reason, F, JF, M, y);
                return;
            else
                rethrow(ME);
            end
        end

        y = y_next;
        h = h_used;
        n = n + 1;
        Y(:,n) = y;

        a = y(1); b = y(2);

        % ---- Outside plotting box with margin? --------------------------
        if a < opts.a_min-1 || a > opts.a_max+1 || ...
           b < opts.b_min-1 || b > opts.b_max+1
            Y = Y(:,1:n);
            term_reason = 'left_plot_box';
            report_termination(term_reason, F, JF, M, y);
            return;
        end

        % ---- Proximity to ANY fixed point (except the originating saddle)
        [idx_fp, dist_e] = nearest_fixed_point(y, fp_pos);
        if dist_e < opts.R_fp_min && idx_fp ~= idx_sad
            Y = Y(:,1:n);
            term_reason = sprintf('near_fixed_point[%s]', fp_type{idx_fp});
            report_termination(term_reason, F, JF, M, y);
            return;
        end
    end

    Y = Y(:,1:n);
    term_reason = 'max_steps';
    report_termination(term_reason, F, JF, M, y);
end

% -------------------------------------------------------------------------
function report_termination(reason, F, JF, M, y)
%REPORT_TERMINATION  Debug print for branch termination.
%
%   Prints:
%       - reason
%       - location y
%       - grad L(y)
%       - Hessian H(y)
%       - curvature kappa of the flow at y

    a = y(1);
    b = y(2);

    g = M.gradL(a,b);
    H = M.hessL(a,b);

    % Curvature using helper; if missing, returns NaN.
    try
        kappa = polyflow.flow_curvature(F, JF, y);
    catch
        kappa = NaN;
    end

    fprintf('\n[Skel] Branch terminated: %s\n', reason);
    fprintf('  y      = [%.16g, %.16g]\n', a, b);
    fprintf('  gradL  = [%.16g, %.16g]\n', g(1), g(2));
    fprintf('  HessL  = [%.16g, %.16g; %.16g, %.16g]\n', ...
        H(1,1), H(1,2), H(2,1), H(2,2));
    fprintf('  kappa  = %.16g\n', kappa);
end

% -------------------------------------------------------------------------
function [idx, dmin] = nearest_fixed_point(y, fp_pos)
    diffs = fp_pos - y;
    dists = sqrt(sum(diffs.^2,1));
    [dmin, idx] = min(dists);
end

% -------------------------------------------------------------------------
function eps_use = choose_eps_opposite_sides(M, y0, v, eps0)
%CHOOSE_EPS_OPPOSITE_SIDES
%   Choose eps so that y0 ± eps*v lie on opposite sides of the backbone,
%   as detected by the sign of dL/da.

    eps_use = eps0;
    for k = 1:8
        y_plus  = y0 + eps_use*v;
        y_minus = y0 - eps_use*v;

        g_plus  = M.gradL(y_plus(1),  y_plus(2));
        g_minus = M.gradL(y_minus(1), y_minus(2));

        if g_plus(1)*g_minus(1) < 0
            return;
        end
        eps_use = 2*eps_use;
    end
end
