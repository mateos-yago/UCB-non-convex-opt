function w = warp_levelset(Lval, grad, opts) %#ok<INUSD>
%WARP_LEVELSET Warp factor for level-set traversal (unit-ish speed).
g_norm = norm(grad, 2) + 1e-12;
w = 1 / g_norm;
end
