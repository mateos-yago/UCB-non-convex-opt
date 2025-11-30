function traj = mse_muon_descent(M, w0, opts)
%MSE_MUON_DESCENT Muon-flavoured momentum descent for 2D MSE models.
%
%   This is a pedagogical heavy-ball / Muon-style descent:
%
%       v_{k+1} = mu * v_k - eta * g_k / max(norm(g_k), g_floor)
%       w_{k+1} = w_k + v_{k+1}
%
%   with optional "energy clipping" to keep ||v|| from blowing up.
%
%   traj = mse_muon_descent(M, w0, opts)
%
%   M   : model struct from mse_build_model, with M.gradL(a,b)
%   w0  : initial [a; b]
%   opts: struct with fields
%       .eta        (default 0.05)
%       .mu         (default 0.9)     momentum
%       .g_floor    (default 1e-3)   gradient norm floor for scaling
%       .v_max      (default 0.5)    max allowed velocity norm
%       .max_steps  (default 1000)
%       .grad_tol   (default 1e-6)
%
%   traj: (2 x T) matrix of iterates [a; b]

    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'eta'),       opts.eta       = 0.05;   end
    if ~isfield(opts, 'mu'),        opts.mu        = 0.9;    end
    if ~isfield(opts, 'g_floor'),   opts.g_floor   = 1e-3;   end
    if ~isfield(opts, 'v_max'),     opts.v_max     = 0.5;    end
    if ~isfield(opts, 'max_steps'), opts.max_steps = 1000;   end
    if ~isfield(opts, 'grad_tol'),  opts.grad_tol  = 1e-6;   end

    w = w0(:);
    v = zeros(2,1);

    traj = zeros(2, opts.max_steps+1);
    traj(:,1) = w;

    for k = 1:opts.max_steps
        a = w(1); b = w(2);
        g = M.gradL(a,b);

        gn = norm(g);
        if gn < opts.grad_tol
            traj = traj(:,1:k);
            return;
        end

        % Normalised gradient with floor
        scale = max(gn, opts.g_floor);
        g_hat = g / scale;

        % Muon / heavy-ball update
        v = opts.mu * v - opts.eta * g_hat;

        % Optional velocity clipping to keep motion "reasonable" in the plot
        vn = norm(v);
        if vn > opts.v_max
            v = (opts.v_max / vn) * v;
        end

        w = w + v;
        traj(:,k+1) = w;
    end

    traj = traj(:,1:opts.max_steps+1);
end
