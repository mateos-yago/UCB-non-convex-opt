function fps = mse_fixed_points(M, opts)
%MSE_FIXED_POINTS Find fixed points from reduced loss u(b), then refine in 2D.
%
%   Uses:
%       - 1D root finding on u'(b) to locate candidates.
%       - a*(b) = B(b)/A(b) for an initial backbone point.
%       - 2D Newton refinement on grad L(a,b) = 0 for high accuracy.
%
%   Output: struct array fps(k) with fields:
%       .a        refined a*
%       .b        refined b*
%       .type     'min' | 'saddle' | 'degenerate'
%       .u_second u''(b*) evaluated at refined b*

    if nargin < 2, opts = struct(); end
    if ~isfield(opts, 'b_min'),          opts.b_min          = -5;   end
    if ~isfield(opts, 'b_max'),          opts.b_max          =  5;   end
    if ~isfield(opts, 'b_samples'),      opts.b_samples      = 400;  end
    if ~isfield(opts, 'tol_root'),       opts.tol_root       = 1e-8; end
    if ~isfield(opts, 'tol_newton'),     opts.tol_newton     = 1e-12; end
    if ~isfield(opts, 'newton_max_iter'),opts.newton_max_iter = 20;  end

    b_min = opts.b_min;
    b_max = opts.b_max;

    % --- 1. Coarse search for sign changes of u'(b) ---
    b_grid  = linspace(b_min, b_max, opts.b_samples);
    up_vals = M.u_prime(b_grid);

    signs = sign(up_vals);
    signs(signs == 0) = 1;  % treat exact zeros as positive for sign-change detection
    changes = find(diff(signs) ~= 0);

    b_roots = [];

    for idx = changes
        a_int = b_grid(idx);
        c_int = b_grid(idx+1);

        % Skip intervals where A(b) hits zero (degenerate / non-Morse)
        if M.A(a_int) == 0 || M.A(c_int) == 0
            continue;
        end

        f = @(bb) M.u_prime(bb);

        opts_fzero = optimset('TolX', 1e-12, 'TolFun', 1e-12, ...
                              'MaxIter', 100, 'MaxFunEvals', 1000);

        try
            br = fzero(f, [a_int, c_int], opts_fzero);
            if br > b_min - 1e-6 && br < b_max + 1e-6
                b_roots(end+1) = br; %#ok<AGROW>
            end
        catch
            % fzero may fail; just skip that interval
        end
    end

    if ~isempty(b_roots)
        % Merge near-duplicates
        b_roots = unique(round(b_roots / opts.tol_root) * opts.tol_root);
    end

    n = numel(b_roots);
    fps = struct('a', cell(1,n), 'b', [], 'type', [], 'u_second', []);

    % --- 2. For each 1D root, refine the full 2D fixed point ---
    for k = 1:n
        b_star0 = b_roots(k);
        a_star0 = M.a_star(b_star0);  % backbone projection

        [a_ref, b_ref] = refine_fixed_point(M, a_star0, b_star0, ...
                                            opts.tol_newton, opts.newton_max_iter);

        % Classify via u'' at refined b*
        u2 = M.u_second(b_ref);
        if u2 > 0
            ttype = 'min';
        elseif u2 < 0
            ttype = 'saddle';
        else
            ttype = 'degenerate';
        end

        fps(k).a        = a_ref;
        fps(k).b        = b_ref;
        fps(k).type     = ttype;
        fps(k).u_second = u2;
    end
end

% -------------------------------------------------------------------------
function [a_ref, b_ref] = refine_fixed_point(M, a0, b0, tol, max_iter)
%REFINE_FIXED_POINT 2D Newton refinement of a fixed point of the gradient flow.
%
%   Solves grad L(a,b) = 0 using Newton iterations:
%       [a;b]_{k+1} = [a;b]_k - H^{-1} gradL
%   where H is the Hessian of L.

    if nargin < 4, tol = 1e-12; end
    if nargin < 5, max_iter = 20; end

    y = [a0; b0];

    for k = 1:max_iter
        g = M.gradL(y(1), y(2));  % 2x1 gradient (dL/da, dL/db)
        if norm(g, 2) < tol
            break;
        end

        H = M.hessL(y(1), y(2));  % 2x2 Hessian
        % Symmetrize for numerical sanity:
        H = 0.5 * (H + H.');

        % Solve H * delta = -g
        delta = - H \ g;

        y = y + delta;

        if norm(delta, 2) < tol
            break;
        end
    end

    a_ref = y(1);
    b_ref = y(2);
end
