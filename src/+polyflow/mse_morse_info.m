function info = mse_morse_info(M, opts)
%MSE_MORSE_INFO Aggregate diagnostics: Hamburger, root support, fixed points.
if nargin < 2, opts = struct(); end
if ~isfield(opts, 'moment_tol'), opts.moment_tol = 1e-10; end
if ~isfield(opts, 'b_min'),      opts.b_min      = -5; end
if ~isfield(opts, 'b_max'),      opts.b_max      =  5; end
d = max(M.deg_f, M.deg_g);
[ok_real, ham_details] = polyflow.mse_check_hamburger(M.mu, d, opts.moment_tol);
[is_root_supp, w, resid] = polyflow.mse_check_root_support(M.g_coeffs, M.mu, 1e-6);
fp_opts = opts;
fps = polyflow.mse_fixed_points(M, fp_opts);
u2 = [fps.u_second];
is_morse_backbone = ~isempty(fps) && all(abs(u2) > 1e-6);
info.ok_real           = ok_real;
info.ham_details       = ham_details;
info.root_supported    = is_root_supp;
info.root_weights      = w;
info.root_resid        = resid;
info.fixed_points      = fps;
info.is_morse_backbone = is_morse_backbone;
info.is_morse_global   = ok_real && ~is_root_supp && is_morse_backbone;
end
