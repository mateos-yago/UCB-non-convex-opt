function [is_supported, w, resid] = mse_check_root_support(g_coeffs, mu, tol)
%MSE_CHECK_ROOT_SUPPORT Approximate check: are moments supported on roots of g?
if nargin < 3, tol = 1e-6; end
mu = mu(:);
K = numel(mu) - 1;
g_high2low = fliplr(g_coeffs(:).');
r = roots(g_high2low);
r = r( abs(imag(r)) < 1e-10 );
r = real(r);
m = numel(r);
if m == 0
    is_supported = false;
    w = [];
    resid = inf;
    return;
end
V = zeros(K+1, m);
for k = 0:K
    V(k+1,:) = r.'.^k;
end
w = V \ mu;
resid = norm(V*w - mu, 2);
if resid < tol && all(w >= -tol) && abs(sum(w) - 1) < sqrt(tol)
    is_supported = true;
else
    is_supported = false;
end
end
