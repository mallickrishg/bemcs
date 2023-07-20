close all;
clear variables;


% Declare symbols and build functions to integrate
syms x0 y0 x y C r g gx gy gxy gxx gyy ux uy sxx syy sxy nu mu xoffset yoffset fx fy

x = x0 - xoffset;
y = y0 - yoffset;
C = 1 / (4 * pi * (1 - nu));
r = sqrt(x^2 + y^2);
g = -C * log(r);
gx = -C * x / (x^2 + y^2);
gy = -C * y / (x^2 + y^2);
gxy = C * 2 * x * y / (x^2 + y^2)^2;
gxx = C * (x^2 - y^2) / (x^2 + y^2)^2;
gyy = -gxx;
ux = fx / (2 * mu) * ((3 - 4 * nu) * g - x * gx) + fy / (2 * mu) * (-y * gx);
uy = fx / (2 * mu) * (-x * gy) + fy / (2 * mu) * ((3 - 4 * nu) * g - y * gy);
sxx = fx * (2 * (1 - nu) * gx - x * gxx) + fy * (2 * nu * gy - y * gxx);
syy = fx * (2 * nu * gx - x * gyy) + fy * (2 * (1 - nu) * gy - y * gyy);
sxy = fx * ((1 - 2 * nu) * gy - x * gxy) + fy * ((1 - 2 * nu) * gx - y * gxy);

% Try integrating the xx component of stress along a line
sxx_definite = int(sxx, x0, [-1.0 1.0]);
ux_definite = int(ux, x0, [-1.0 1.0]);

% Convert to function
sxx_function_handle = matlabFunction(sxx_definite);
ux_function_handle = matlabFunction(ux_definite);

% Plot result over a grid
n_pts = 50;
x_vec = linspace(-1.5, 1.5, n_pts);
y_vec = linspace(-0.5, 0.5, n_pts);
[x_mat, y_mat] = meshgrid(x_vec, y_vec);
sxx_mat = zeros(size(x_mat));
ux_mat = zeros(size(x_mat));

for i=1:n_pts
    for j=1:n_pts
        sxx_mat(i, j) = sxx_function_handle(1, 0, 0.25, x_mat(i, j), y_mat(i, j), 0);
        ux_mat(i, j) = ux_function_handle(1, 0, 1, 0.25, x_mat(i, j), y_mat(i, j), 0);
    end
end
sxx_mat = -1 * sxx_mat; % Sign convetion...dunno

% Uniform numeric integration
ux_numeric_mat_total = zeros(size(x_mat));
ux_numeric_mat = zeros(size(x_mat));
sxx_numeric_mat_total = zeros(size(x_mat));
sxx_numeric_mat = zeros(size(x_mat));
point_vec = linspace(-1.0, 1.0, 10);
for k=1:numel(point_vec)
    for i=1:n_pts
        for j=1:n_pts
            [n_ux, n_sxx] = kelvin_point(x_mat(i, j), y_mat(i, j), point_vec(k), 0, 1, 0, 1, 0.25);
            ux_numeric_mat(i, j) = n_ux;
            sxx_numeric_mat(i, j) = n_sxx;
        end
    end
    ux_numeric_mat_total = ux_numeric_mat_total + ux_numeric_mat;
    sxx_numeric_mat_total = sxx_numeric_mat_total + sxx_numeric_mat;
end
ux_numeric_mat_total = ux_numeric_mat_total / numel(point_vec);
sxx_numeric_mat_total = sxx_numeric_mat_total / numel(point_vec);


% Gauss-Legendre numeric
% Nodes and weights from:
% https://pomax.github.io/bezierinfo/legendre-gauss.html#n10
ux_gl_mat_total = zeros(size(x_mat));
ux_gl_mat = zeros(size(x_mat));
sxx_gl_mat_total = zeros(size(x_mat));
sxx_gl_mat = zeros(size(x_mat));

point_weight = [
0.2955242247147529
0.2955242247147529
0.2692667193099963
0.2692667193099963
0.2190863625159820
0.2190863625159820
0.1494513491505806
0.1494513491505806
0.0666713443086881
0.0666713443086881
];

point_vec = [
-0.1488743389816312
0.1488743389816312
-0.4333953941292472
0.4333953941292472
-0.6794095682990244
0.6794095682990244
-0.8650633666889845
0.8650633666889845
-0.9739065285171717
0.9739065285171717
];

for k=1:numel(point_vec)
    for i=1:n_pts
        for j=1:n_pts
            [n_ux, n_sxx] = kelvin_point(x_mat(i, j), y_mat(i, j), point_vec(k), 0, 1, 0, 1, 0.25);
            ux_numeric_mat(i, j) = point_weight(k) * n_ux;
            sxx_numeric_mat(i, j) = point_weight(k) * n_sxx;
        end
    end
    ux_gl_mat_total = ux_gl_mat_total + ux_numeric_mat;
    sxx_gl_mat_total = sxx_gl_mat_total + sxx_numeric_mat;
end
ux_gl_mat_total = ux_gl_mat_total / numel(point_vec);
sxx_gl_mat_total = sxx_gl_mat_total / numel(point_vec);


% Plot summary (ux)
figure("Position", [0, 0, 1600, 600])

subplot(2, 3, 1);
contourf(x_mat, y_mat, ux_mat)
colorbar;
axis("equal")
title("exact")

subplot(2, 3, 2);
contourf(x_mat, y_mat, ux_numeric_mat_total)
colorbar;
axis("equal")
title("uniform numerical integration")

subplot(2, 3, 3);
contourf(x_mat, y_mat, ux_gl_mat_total)
colorbar;
axis("equal")
title("10 point Gauss-Legendre numerical integration")

subplot(2, 3, 5);
contourf(x_mat, y_mat, 100 * (ux_numeric_mat_total + ux_mat) ./ ux_mat)
colorbar;
axis("equal")
title("uniform numerical integration (% error)")

subplot(2, 3, 6);
contourf(x_mat, y_mat, 100 * (ux_gl_mat_total + ux_mat) ./ ux_mat)
colorbar;
axis("equal")
title("10 point Gauss-Legendre numerical integration (% error)")




% Plot summary (sxx)
figure("Position", [0, 0, 1600, 600])

subplot(2, 3, 1);
contourf(x_mat, y_mat, sxx_mat)
colorbar;
axis("equal")
title("exact")

subplot(2, 3, 2);
contourf(x_mat, y_mat, sxx_numeric_mat_total)
colorbar;
axis("equal")
title("uniform numerical integration")

subplot(2, 3, 3);
contourf(x_mat, y_mat, sxx_gl_mat_total)
colorbar;
axis("equal")
title("10 point Gauss-Legendre numerical integration")

subplot(2, 3, 5);
contourf(x_mat, y_mat, 100 * (sxx_numeric_mat_total + sxx_mat) ./ sxx_mat)
colorbar;
axis("equal")
title("uniform numerical integration (% error)")

subplot(2, 3, 6);
contourf(x_mat, y_mat, 100 * (sxx_gl_mat_total + sxx_mat) ./ sxx_mat)
colorbar;
axis("equal")
title("10 point Gauss-Legendre numerical integration (% error)")


function [ux, sxx] = kelvin_point(x0, y0, xoffset, yoffset, fx, fy, mu, nu)
    x = x0 - xoffset;
    y = y0 - yoffset;
    C = 1 / (4 * pi * (1 - nu));
    r = sqrt(x^2 + y^2);
    g = -C * log(r);
    gx = -C * x / (x^2 + y^2);
    gy = -C * y / (x^2 + y^2);
    gxy = C * 2 * x * y / (x^2 + y^2)^2;
    gxx = C * (x^2 - y^2) / (x^2 + y^2)^2;
    gyy = -gxx;
    ux = fx / (2 * mu) * ((3 - 4 * nu) * g - x * gx) + fy / (2 * mu) * (-y * gx);
    uy = fx / (2 * mu) * (-x * gy) + fy / (2 * mu) * ((3 - 4 * nu) * g - y * gy);
    sxx = fx * (2 * (1 - nu) * gx - x * gxx) + fy * (2 * nu * gy - y * gxx);
    syy = fx * (2 * nu * gx - x * gyy) + fy * (2 * (1 - nu) * gy - y * gyy);
    sxy = fx * ((1 - 2 * nu) * gy - x * gxy) + fy * ((1 - 2 * nu) * gx - y * gxy);
end



