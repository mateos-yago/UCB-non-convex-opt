function backbone = mse_compute_backbone(M, b_grid)
%MSE_COMPUTE_BACKBONE Sample backbone a*(b) and u(b) on given grid.
backbone.b = b_grid(:).';
backbone.a = M.a_star(backbone.b);
backbone.u = M.u(backbone.b);
end
