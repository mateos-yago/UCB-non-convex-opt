function y_next = sundman_step(base_vf, Lfun, mode, t, y, h, opts)
%SUNDMAN_STEP Implicit midpoint + Sundman warp wrapper.
if nargin < 7, opts = struct(); end
if ~isfield(opts, 'newton')
    opts.newton = struct();
end
vf = @(tt, yy) wrapped_vf(base_vf, Lfun, mode, tt, yy, opts);
y_next = polyflow.implicit_midpoint_step(vf, t, y, h, opts.newton);
end

function v = wrapped_vf(base_vf, Lfun, mode, t, y, opts)
g = base_vf(t, y);
Lval = Lfun(y);
switch mode
    case {'descent','ascent'}
        w = polyflow.warp_descent(Lval, g, opts);
    case 'level'
        w = polyflow.warp_levelset(Lval, g, opts);
    otherwise
        error('Unknown mode: %s', mode);
end
v = w * g;
end
