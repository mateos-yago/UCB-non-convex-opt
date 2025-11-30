function M = mse_build_model(f_coeffs, g_coeffs, mu)
%MSE_BUILD_MODEL Build MSE model from f,g coefficients and moments.
f_coeffs = f_coeffs(:).';
g_coeffs = g_coeffs(:).';
mu      = mu(:).';
deg_f = numel(f_coeffs) - 1;
deg_g = numel(g_coeffs) - 1;
if numel(mu) < 2*max(deg_f, deg_g) + 1
    error('Need at least mu_0..mu_{2d} with d >= max(deg f, deg g).');
end
% C = E[f^2]
C = 0;
for i = 0:deg_f
    for k = 0:deg_f
        C = C + f_coeffs(i+1)*f_coeffs(k+1)*mu(i+k+1);
    end
end
% A(b) coefficients
maxAdeg = 2*deg_g;
alpha = zeros(1, maxAdeg+1);
for i = 0:deg_g
    for j = 0:deg_g
        m = i + j;
        alpha(m+1) = alpha(m+1) + g_coeffs(i+1)*g_coeffs(j+1)*mu(i+j+1);
    end
end
% B(b) coefficients
maxBdeg = deg_g;
beta = zeros(1, maxBdeg+1);
for j = 0:deg_g
    s = 0;
    for i = 0:deg_f
        s = s + f_coeffs(i+1)*g_coeffs(j+1)*mu(i+j+1);
    end
    beta(j+1) = s;
end
A_poly       = @(b) polyval(fliplr(alpha), b);
Aprime_poly  = @(b) polyflow.polyder_eval(alpha, b);
B_poly       = @(b) polyval(fliplr(beta), b);
Bprime_poly  = @(b) polyflow.polyder_eval(beta, b);
Lfun = @(a,b) C - 2*a.*B_poly(b) + (a.^2).*A_poly(b);
gradL = @(a,b) polyflow.gradL_fun(a,b,A_poly,B_poly,Aprime_poly,Bprime_poly);
hessL = @(a,b) polyflow.hessL_fun(a,b,A_poly,B_poly,Aprime_poly,Bprime_poly);
M.f_coeffs = f_coeffs;
M.g_coeffs = g_coeffs;
M.mu       = mu;
M.deg_f    = deg_f;
M.deg_g    = deg_g;
M.C        = C;
M.alpha    = alpha;
M.beta     = beta;
M.A        = A_poly;
M.Aprime   = Aprime_poly;
M.B        = B_poly;
M.Bprime   = Bprime_poly;
M.L        = @(a,b) Lfun(a,b);
M.gradL    = @(a,b) gradL(a,b);
M.hessL    = @(a,b) hessL(a,b);
M.a_star   = @(b) polyflow.a_star_fun(b, A_poly, B_poly);
M.u        = @(b) polyflow.u_fun(b, C, A_poly, B_poly);
M.u_prime  = @(b) polyflow.u_prime_fun(b, C, A_poly, B_poly, Aprime_poly, Bprime_poly);
M.u_second = @(b) polyflow.u_second_fd(b, M.u);
end
