function y_next = implicit_midpoint_step(f, t, y, h, newton_opts, Jfun)
%IMPLICIT_MIDPOINT_STEP Single implicit midpoint step for 2D ODE.
%
%   y_next = implicit_midpoint_step(f, t, y, h, newton_opts)
%   y_next = implicit_midpoint_step(f, t, y, h, newton_opts, Jfun)
%
%   f        : vector field handle f(t,y) (column vector)
%   newton_opts.max_iter : maximum Newton iterations (default 10)
%   newton_opts.tol      : Newton residual/step tolerance (default 1e-8)
%   newton_opts.allow_numeric : if true, fall back to numerical Jacobian
%                               when Jfun is empty (default false)
%   Jfun     : optional Jacobian handle J(t,y); if omitted or empty and
%              allow_numeric is true, numerical_jacobian is used.
%
%   This implementation is tailored for the polynomial-flow project:
%   - prefers analytic Jacobians where available;
%   - keeps a numerical Jacobian only as a diagnostic fallback.

    if nargin < 5 || isempty(newton_opts), newton_opts = struct(); end
    if ~isfield(newton_opts, 'max_iter'),        newton_opts.max_iter        = 10;    end
    if ~isfield(newton_opts, 'tol'),             newton_opts.tol             = 1e-8;  end
    if ~isfield(newton_opts, 'allow_numeric'),   newton_opts.allow_numeric   = false; end

    if nargin < 6
        Jfun = [];
    end

    y = y(:);
    y_next = y;

    for k = 1:newton_opts.max_iter
        t_mid = t + 0.5*h;
        y_mid = 0.5*(y + y_next);

        F = y_next - y - h * f(t_mid, y_mid);
        if norm(F, 2) < newton_opts.tol
            return;
        end

        if ~isempty(Jfun)
            % Analytic Jacobian for the midpoint residual:
            % G(z) = z - y - h f(t_mid, (y+z)/2)
            % J_G = I - (h/2) * J_f(t_mid, y_mid)
            Jf_mid = Jfun(t_mid, y_mid);
            J = eye(numel(y)) - 0.5*h * Jf_mid;
        elseif newton_opts.allow_numeric
            % Diagnostic fallback: numerical Jacobian of the residual.
            G = @(z) z - y - h * f(t_mid, 0.5*(y + z));
            J = polyflow.numerical_jacobian(G, y_next);
        else
            error('implicit_midpoint_step:NoJacobian', ...
                  ['Analytic Jacobian handle Jfun is required unless ', ...
                   'newton_opts.allow_numeric is set to true.']);
        end

        delta = - J \ F;
        y_next = y_next + delta;

        if norm(delta, 2) < newton_opts.tol
            return;
        end
    end
end
