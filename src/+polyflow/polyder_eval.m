function val = polyder_eval(coeffs, b)
%POLYDER_EVAL Evaluate derivative of polynomial at b.
n = numel(coeffs)-1;
if n <= 0
    val = 0;
    return;
end
d = coeffs(2:end) .* (1:n);
val = polyval(fliplr(d), b);
end
