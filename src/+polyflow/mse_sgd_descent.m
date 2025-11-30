function traj = mse_sgd_descent(M, w0, opts)
%MSE_SGD_DESCENT Simple SGD-style descent on M.L with analytic gradient.
%
%   traj = mse_sgd_descent(M, w0, opts)
%
%   M   : model struct from mse_build_model, with M.gradL(a,b)
%   w0  : initial [a; b]
%   opts: struct with optional fields
%       .eta0       (default 0.05)  step size
%       .eta_decay  (default 0)     decay per step: eta_k = eta0 / (1 + decay*k)
%       .max_steps  (default 1000)
%       .grad_tol   (default 1e-6)
%
%   traj: (2 x T) matrix of iterates [a; b]
%
%   Designed purely for 2D teaching phase portraits.

    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'eta0'),      opts.eta0      = 0.05;  end
    if ~isfield(opts, 'eta_decay'), opts.eta_decay = 0.0;   end
    if ~isfield(opts, 'max_steps'), opts.max_steps = 1000;  end
    if ~isfield(opts, 'grad_tol'),  opts.grad_tol  = 1e-6;  end

    w = w0(:);
    traj = zeros(2, opts.max_steps+1);
    traj(:,1) = w;

    for k = 1:opts.max_steps
        a = w(1); b = w(2);
        g = M.gradL(a,b);     % column gradient [dL/da; dL/db]

        if norm(g) < opts.grad_tol
            traj = traj(:,1:k);
            return;
        end

        eta = opts.eta0 / (1 + opts.eta_decay * (k-1));
        w   = w - eta * g;
        traj(:,k+1) = w;
    end

    % No early stop
    traj = traj(:,1:opts.max_steps+1);
end
