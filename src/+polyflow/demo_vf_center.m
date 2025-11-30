function demo_vf_center()
%DEMO_VF_CENTER Simple linear center using implicit midpoint.

f = @(t,y) [y(2); -y(1)];  % rotation (center)

h = 0.05;
steps = 400;
t = 0;
y = [1; 0];

yy = zeros(2, steps);
yy(:,1) = y;

opts = struct();
for n = 2:steps
    y = polyflow.implicit_midpoint_step(f, t, y, h, opts);
    yy(:,n) = y;
    t = t + h;
end

figure; plot(yy(1,:), yy(2,:), 'b-'); axis equal;
title('Linear center via implicit midpoint');
xlabel('x'); ylabel('y');
end
