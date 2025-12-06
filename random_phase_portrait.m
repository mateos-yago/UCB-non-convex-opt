
% random initial conditions:

rng(139)

n = 32; % number of vectors
K = 16*n; % number of arms
dim = 2*n;

Y0 = -6 + 12*rand(dim, K);  % (2n)xK initializations
    
d = 2;
f = randn(1, d+1);
g = randn(1, d+1);
mu = 1./(1:(2*d+1));

M = polyflow.mse_build_model(f, g, mu);



opts = struct();
opts.eta = 0.0005;

polyflow.mse_plot_ucb_trajectories_multiple_points(M, Y0, n*10000, opts);