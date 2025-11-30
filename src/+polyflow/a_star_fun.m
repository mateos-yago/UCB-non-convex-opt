function a = a_star_fun(b,A,B)
%A_STAR_FUN Backbone a*(b) = B(b)/A(b).
A_b = A(b);
B_b = B(b);
a = B_b ./ A_b;
end
