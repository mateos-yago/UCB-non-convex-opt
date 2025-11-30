function [ok, details] = mse_check_hamburger(mu, d, tol)
%MSE_CHECK_HAMBURGER Hamburger moment sanity check (finite segment).
if nargin < 3, tol = 1e-10; end
mu = mu(:).';
ok = true;
details.H = cell(d+1, 1);
details.eigs = cell(d+1, 1);
for k = 0:d
    Hk = hankel(mu(1:k+1), mu(k+1:2*k+1));
    details.H{k+1} = Hk;
    ev = eig((Hk + Hk')/2);
    details.eigs{k+1} = ev;
    if min(ev) < -tol
        ok = false;
    end
end
end
