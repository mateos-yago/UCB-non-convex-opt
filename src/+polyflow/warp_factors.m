function w = warp_descent(Lval, grad, opts)
%WARP_DESCENT Sundman warp factor for descent/ascent flows.
%
%   w = warp_descent(Lval, grad, opts)
%
% Default: log shaping + scaling with gradient norm to control stiffness.

if nargin < 3, opts = struct(); end
if ~isfield(opts, 'L_epsilon'), opts.L_epsilon = 1e-8; end

Ltilde = log(Lval + opts.L_epsilon);  % shaping
g_norm = norm(grad, 2) + 1e-12;

% Simple choice: slow down when gradient huge or L large
w = 1 / (1 + abs(Ltilde) + g_norm);
end

function w = warp_levelset(Lval, grad, opts)
%WARP_LEVELSET Warp factor for level-set traversal.
%
% Default: normalize speed to ~1 along curve.

if nargin < 3, opts = struct(); end

g_norm = norm(grad, 2) + 1e-12;

% Make the level set vector field roughly unit-speed
w = 1 / g_norm;
end
