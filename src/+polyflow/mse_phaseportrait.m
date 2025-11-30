


function mse_phaseportrait(M, opts)
%MSE_PHASEPORTRAIT Phase portrait for MSE gradient flow in (a,b)-plane.
%
% Uses:
%   - mse_morse_info(M, opts) for critical points and Hessians.
%   - mse_plot_backbone(M, opts) to draw the dL/da = 0 nullcline.
%   - mse_plot_morse_skeleton_geometric(M, info, sep_opts) for the
%     geometric Morse skeleton (stable/unstable manifolds).
%
% Plot consists of:
%   - log(L) contours in light gray on a box that contains ALL finite
%     critical points (expanded by a margin) UNION the user's requested box.
%   - backbone.
%   - critical points (min, saddle, etc).
%   - Morse skeleton (green: unstable for descent, red: stable for descent).

    if nargin < 2, opts = struct(); end

    % --- user defaults (may be overridden by auto-box union) ---
    if ~isfield(opts, 'a_min'),      opts.a_min      = -6; end
    if ~isfield(opts, 'a_max'),      opts.a_max      =  6; end
    if ~isfield(opts, 'b_min'),      opts.b_min      = -6; end
    if ~isfield(opts, 'b_max'),      opts.b_max      =  6; end
    if ~isfield(opts, 'MeshDensity'),opts.MeshDensity = 200; end
    if ~isfield(opts, 'newton'),     opts.newton     = struct(); end

    % --- Morse / fixed point info (does NOT depend on plotting box) ---
    info = polyflow.mse_morse_info(M, opts);

    % Light diagnostics if fields exist
    if isfield(info, 'ok_real')        && ...
       isfield(info, 'root_supported') && ...
       isfield(info, 'morse_backbone') && ...
       isfield(info, 'morse_global')
        fprintf('MSE Morse diagnostics: ok_real=%d, root_supported=%d, morse_backbone=%d, morse_global=%d\n', ...
            info.ok_real, info.root_supported, info.morse_backbone, info.morse_global);
    end

    % --- Auto bounding box from finite fixed points ---
    if isfield(info, 'fixed_points') && ~isempty(info.fixed_points)
        fps = info.fixed_points;
        a_fp = [fps.a];
        b_fp = [fps.b];

        a_min_fp = min(a_fp);
        a_max_fp = max(a_fp);
        b_min_fp = min(b_fp);
        b_max_fp = max(b_fp);

        % Simple margin: 20% of the span, at least 0.5
        span_a = max(a_max_fp - a_min_fp, 1e-6);
        span_b = max(b_max_fp - b_min_fp, 1e-6);
        margin_a = max(0.2 * span_a, 0.5);
        margin_b = max(0.2 * span_b, 0.5);

        a_min_auto = a_min_fp - margin_a;
        a_max_auto = a_max_fp + margin_a;
        b_min_auto = b_min_fp - margin_b;
        b_max_auto = b_max_fp + margin_b;
    else
        % Fallback: keep user box if we somehow have no fixed points
        a_min_auto = opts.a_min;
        a_max_auto = opts.a_max;
        b_min_auto = opts.b_min;
        b_max_auto = opts.b_max;
    end

    % --- Union of user box and auto box ---
    opts.a_min = min(opts.a_min, a_min_auto);
    opts.a_max = max(opts.a_max, a_max_auto);
    opts.b_min = min(opts.b_min, b_min_auto);
    opts.b_max = max(opts.b_max, b_max_auto);

    % --- Background: log(L) level sets in light gray on the union box ---
    LogL = @(a,b) log(M.L(a,b));  % assume L>0 on this box for now

    figure; clf; hold on;

    fcontour(LogL, ...
        [opts.a_min, opts.a_max, opts.b_min, opts.b_max], ...
        'LineColor', 0.6*[1 1 1], ...
        'LevelList', (-5:0.5:25), ...
        'MeshDensity', opts.MeshDensity);

    xlabel('a');
    ylabel('b');
    axis([opts.a_min opts.a_max opts.b_min opts.b_max]);
    % axis equal;

    % --- Backbone (nullcline dL/da = 0) ---
    if exist('polyflow.mse_plot_backbone', 'file') == 2 || ...
       exist('+polyflow/mse_plot_backbone.m', 'file') == 2
        try
            polyflow.mse_plot_backbone(M, opts);
        catch ME
            warning('mse_phaseportrait:plotBackboneFailed', ...
                'mse_plot_backbone failed: %s', ME.message);
        end
    end

    % --- Critical points (from info.fixed_points) ---
    if isfield(info, 'fixed_points') && ~isempty(info.fixed_points)
        fps = info.fixed_points;
        for k = 1:numel(fps)
            a = fps(k).a;
            b = fps(k).b;

            if ~isfield(fps(k), 'type')
                % Fallback if no type is present
                plot(a, b, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
                continue;
            end

            switch lower(fps(k).type)
                case 'min'
                    % global / local minima: blue filled circles
                    plot(a, b, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
                case 'max'
                    % maxima (if any): red filled circles
                    plot(a, b, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
                case 'saddle'
                    % saddles: blue crosses
                    plot(a, b, 'bx', 'LineWidth', 2, 'MarkerSize', 8);
                otherwise
                    % degenerate / other: black diamond
                    plot(a, b, 'kd', 'MarkerSize', 7, 'LineWidth', 1.5);
            end
        end
    end

    % --- Geometric Morse skeleton (separatrices) ---
    sep_opts = struct();
    sep_opts.ds_sep        = 0.02;
    sep_opts.max_steps_sep = 1e3;        % arc-length based; generous but finite
    sep_opts.sep_eps       = 1e-4;       % initial offset along eigendirections

    sep_opts.a_min = opts.a_min;
    sep_opts.a_max = opts.a_max;
    sep_opts.b_min = opts.b_min;
    sep_opts.b_max = opts.b_max;

    % geometric integrator settings
    sep_opts.min_speed           = 1e-10;
    sep_opts.use_curvature_adapt = true;
    sep_opts.N_turns             = 24;    % steps per 2π of tangent rotation
    sep_opts.kappa_floor         = 1e-6;  % avoid crazy ds from tiny curvature

    % capture near minima
    sep_opts.capture_radius_min  = 0.1;
    sep_opts.grad_tol_min        = 1e-4;

    % newton settings reserved (not used here, but kept for consistency)
    sep_opts.newton = struct('max_iter', 50, ...
                             'tol',      1e-12, ...
                             'allow_numeric', false);

    % set to false by default; you can flip this on in experiments
    sep_opts.debug = false;

    polyflow.mse_plot_morse_skeleton_geometric(M, info, sep_opts);

    % polyflow.mse_plot_morse_skeleton_reversible(M, info, sep_opts);

    hold off;
    axis([opts.a_min opts.a_max opts.b_min opts.b_max]);  % ensure final box
end
