
% random initial conditions:

rng(138)

n = 16; % number of vectors
K = 16*n; % number of arms
dim = 2*n;

Y0 = -6 + 12*rand(dim, K);  % (2n)xK initializations
    
d = 2;
f = randn(1, d+1);
g = randn(1, d+1);
mu = 1./(1:(2*d+1));

M = polyflow.mse_build_model(f, g, mu);

figure(1);

polyflow.mse_phaseportrait(M, []);  % this draws the log L, backbone, skeleton, etc.

opts = struct();

%visualizer for 2D case
%polyflow.mse_plot_midpoint_trajectories(M, Y0, opts);


opts.eta = 0.0005; opts.n_steps = 1e4;
loss_table = polyflow.mse_plot_ucb_trajectories_multiple_points(M, Y0, 50000, opts);

%opts.eta = 1e-4; opts.n_steps = 10/opts.eta;
%polyflow.mse_plot_sgd_trajectories(M, Y0, opts);

%opts.eta = 1e-2; opts.n_steps = 10/opts.eta;
%polyflow.mse_plot_adamw_trajectories(M, Y0, opts);

%opts.eta = 1e-3; opts.n_steps = 10/opts.eta;
%polyflow.mse_plot_muon_trajectories(M, Y0, opts);