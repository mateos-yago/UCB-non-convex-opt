function mse_plot_backbone(M, opts)
%MSE_PLOT_BACKBONE Plot backbone a = a*(b) with a on x-axis, b on y-axis.
if nargin < 2, opts = struct(); end
if ~isfield(opts, 'b_min'), opts.b_min = -5; end
if ~isfield(opts, 'b_max'), opts.b_max =  5; end
if ~isfield(opts, 'b_samples'), opts.b_samples = 8000; end
b = linspace(opts.b_min, opts.b_max, opts.b_samples);
a = M.a_star(b);
plot(a, b, 'k-', 'LineWidth', 1.5);
xlabel('a'); ylabel('b');
end
