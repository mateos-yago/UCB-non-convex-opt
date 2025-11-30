function J = numerical_jacobian(F, y)
%NUMERICAL_JACOBIAN Approximate Jacobian of F at y by central differences.
eps0 = 1e-6;
y = y(:);
n = numel(y);
Fy = F(y); %#ok<NASGU>
m = numel(Fy);
J = zeros(m, n);
for j = 1:n
    h = eps0 * max(1, abs(y(j)));
    e = zeros(n,1); e(j) = 1;
    Fp = F(y + h*e);
    Fm = F(y - h*e);
    J(:,j) = (Fp - Fm) / (2*h);
end
end
