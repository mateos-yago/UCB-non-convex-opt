


function mse_plot_muon_trajectories(M, Y0, opts)
%MSE_PLOT_MUON_TRAJECTORIES  Plot Muon optimizer trajectories.
%
%   Same interface as SGD/AdamW.
%   Uses momentum of *normalized* gradient:
%
%      v_{k+1} = beta * v_k + (1-beta) * g/||g||
%      x_{k+1} = x_k - eta_k * v_{k+1}
%
%   Supports learning rate schedules via opts.eta(k).

    if nargin<3, opts=struct(); end

    if ~isfield(opts,'eta'), opts.eta = 0.01; end
    if ~isfield(opts,'beta'), opts.beta = 0.9; end
    if ~isfield(opts,'eps'), opts.eps = 1e-8; end
    if ~isfield(opts,'noise_std'), opts.noise_std=0; end
    if ~isfield(opts,'n_steps'), opts.n_steps=1000; end

    if ~isfield(opts,'color'), opts.color=[1 1 0]; end
    if ~isfield(opts,'linewidth'), opts.linewidth=1.5; end
    if ~isfield(opts,'marker_start'), opts.marker_start=true; end
    if ~isfield(opts,'marker_end'), opts.marker_end=false; end

    use_box = isfield(opts,'a_min') && isfield(opts,'a_max') && ...
              isfield(opts,'b_min') && isfield(opts,'b_max');

    if ~isa(opts.eta,'function_handle')
        eta0 = opts.eta;
        opts.eta = @(k) eta0;
    end

    [~, n_traj] = size(Y0);
    hold on;

    for j=1:n_traj
        y = Y0(:,j);
        v = zeros(2,1);

        traj = zeros(2,opts.n_steps+1);
        traj(:,1)=y;

        k_final = opts.n_steps;

        for k=1:opts.n_steps
            alpha = opts.eta(k);

            g = M.gradL(y(1),y(2));
            if opts.noise_std>0
                g = g + opts.noise_std*randn(2,1);
            end

            ng = g / (norm(g)+opts.eps);
            v = opts.beta * v + (1-opts.beta) * ng;

            y = y - alpha * v;
            traj(:,k+1)=y;

            if use_box
                if y(1)<opts.a_min || y(1)>opts.a_max || ...
                   y(2)<opts.b_min || y(2)>opts.b_max
                    k_final=k;
                    traj = traj(:,1:k_final+1);
                    break;
                end
            end
        end

        plot(traj(1,:), traj(2,:), '-', 'Color',opts.color, 'LineWidth',opts.linewidth);

        if opts.marker_start
            plot(traj(1,1), traj(2,1),'o','MarkerSize',4,'MarkerFaceColor',opts.color,'MarkerEdgeColor','none');
        end
        if opts.marker_end
            plot(traj(1,end), traj(2,end),'s','MarkerSize',4,'MarkerFaceColor',opts.color,'MarkerEdgeColor','none');
        end
    end
end
