function up = u_prime_fun(b, C, A, B, Aprime, Bprime) %#ok<INUSD>
%U_PRIME_FUN Derivative u'(b) for the reduced loss u(b).
A_b  = A(b);
Ap_b = Aprime(b);
B_b  = B(b);
Bp_b = Bprime(b);
num = -2*B_b.*Bp_b.*A_b + (B_b.^2).*Ap_b;
den = A_b.^2;
up  = num ./ den;
end
