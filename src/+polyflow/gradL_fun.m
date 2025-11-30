function g = gradL_fun(a,b,A,B,Aprime,Bprime)
%GRADL_FUN Gradient of L(a,b) for the MSE model.
A_b = A(b);
B_b = B(b);
dLda = -2*B_b + 2*a.*A_b;
dLdb = -2*a.*Bprime(b) + (a.^2).*Aprime(b);
g = [dLda; dLdb];
end
