function H = hessL_fun(a,b,A,B,Aprime,Bprime)
%HESSL_FUN Hessian of L(a,b) for the MSE model (approx L_bb numerically).
A_b  = A(b);
Ap_b = Aprime(b);
B_b  = B(b);
Bp_b = Bprime(b);
L_aa = 2*A_b;
L_ab = -2*Bp_b + 2*a.*Ap_b;
epsb = 1e-5;
dLdb = @(bb) -2*a.*Bprime(bb) + (a.^2).*Aprime(bb);
L_bb = (dLdb(b+epsb) - dLdb(b-epsb)) / (2*epsb);
H = [L_aa, L_ab; L_ab, L_bb];
end
