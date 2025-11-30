function w = warp_descent(Lval, grad, opts)
%WARP_DESCENT Sundman warp factor for descent/ascent flows.
if nargin < 3, opts = struct(); end
if ~isfield(opts, 'L_epsilon'), opts.L_epsilon = 1e-8; end
Ltilde = log(Lval + opts.L_epsilon);
g_norm = norm(grad, 2) + 1e-12;
% Emphasize acceleration near minima by attenuating gradient term:
w = 1 / (1 + abs(Ltilde) + 0.5*g_norm);
end
