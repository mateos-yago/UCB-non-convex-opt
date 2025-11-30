function u2 = u_second_fd(b, u)
%U_SECOND_FD Numeric second derivative of scalar u(b) at b.
h = 1e-4 * max(1, abs(b));
u2 = (u(b+h) - 2*u(b) + u(b-h)) / (h^2);
end
