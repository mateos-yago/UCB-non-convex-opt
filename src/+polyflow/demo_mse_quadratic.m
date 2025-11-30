function demo_mse_quadratic(f_coeffs, g_coeffs, mu)
%DEMO_MSE_QUADRATIC Simple MSE phase portrait demo.
M = polyflow.mse_build_model(f_coeffs, g_coeffs, mu);
opts = struct();
opts.a_min = -6;
opts.a_max =  6;
opts.b_min = -6;
opts.b_max =  6;
opts.h      = 0.001;
opts.grid_n = 120;
opts.traj_steps = 400;
opts.n_traj_a = 8;
opts.n_traj_b = 8;
polyflow.mse_phaseportrait(M, opts);
end
