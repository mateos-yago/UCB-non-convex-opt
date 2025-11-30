function traj = mse_adamw_descent(M, w0, opts)
%MSE_ADAMW_DESCENT AdamW-style optimizer for 2D MSE models.
%
%   traj = mse_adamw_descent(M, w0, opts)
%
%   M   : model struct from mse_build_model, with M.gradL(a,b)
%   w0  : initial [a; b]
%   opts: struct with optional fields
%       .eta       (default 0.01)   base learning rate
%       .beta1     (default 0.9)
%       .beta2     (default 0.999)
%       .eps       (default 1e-8)
%       .weight_decay (default 1e-2)  L2 decay on parameters
%       .max_steps (default 1000)
%       .grad_tol  (default 1e-6)
%
%   traj: (2 x T) matrix of iterates [a; b]

    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'eta'),          opts.eta          = 0.01;   end
    if ~isfield(opts, 'beta1'),        opts.beta1        = 0.9;    end
    if ~isfield(opts, 'beta2'),        opts.beta2        = 0.999;  end
    if ~isfield(opts, 'eps'),          opts.eps          = 1e-8;   end
    if ~isfield(opts, 'weight_decay'), opts.weight_decay = 1e-2;   end
    if ~isfield(opts, 'max_steps'),    opts.max_steps    = 1000;   end
    if ~isfield(opts, 'grad_tol'),     opts.grad_tol     = 1e-6;   end

    w = w0(:);
    m = zeros(2,1);
    v = zeros(2,1);

    traj = zeros(2, opts.max_steps+1);
    traj(:,1) = w;

    beta1 = opts.beta1;
    beta2 = opts.beta2;
    eta   = opts.eta;
    eps   = opts.eps;
    wd    = opts.weight_decay;

    for k = 1:opts.max_steps
        a = w(1); b = w(2);
        g = M.gradL(a,b);

        if norm(g) < opts.grad_tol
            traj = traj(:,1:k);
            return;
        end

        % Apply decoupled weight decay (AdamW-style)
        g_decayed = g + wd * w;

        % Adam moments
        m = beta1 * m + (1-beta1) * g_decayed;
        v = beta2 * v + (1-beta2) * (g_decayed.^2);

        m_hat = m / (1 - beta1^k);
        v_hat = v / (1 - beta2^k);

        w = w - eta * (m_hat ./ (sqrt(v_hat) + eps));
        traj(:,k+1) = w;
    end

    traj = traj(:,1:opts.max_steps+1);
end
