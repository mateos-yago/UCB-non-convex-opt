function u = u_fun(b, C, A, B)
%U_FUN Reduced loss u(b) = min_a L(a,b) = C - B(b)^2/A(b).
A_b = A(b);
B_b = B(b);
u = C - (B_b.^2) ./ A_b;
end
