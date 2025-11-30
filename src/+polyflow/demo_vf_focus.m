function demo_vf_focus()
%DEMO_VF_FOCUS Spiral focus test.

A = [-0.2 -1.0; 1.0 -0.2];
f = @(t,y) A*y;

h = 0.05;
steps = 400;
t = 0;
y = [2; 0];

yy = zeros(2, steps);
yy(:,1) = y;

opts = struct();
for n = 2:steps
    y = polyflow.implicit_midpoint_step(f, t, y, h, opts);
    yy(:,n) = y;
    t = t + h;
end

figure; plot(yy(1,:), yy(2,:), 'b-'); axis equal;
title('Spiral focus via implicit midpoint');
xlabel('x'); ylabel('y');
end
