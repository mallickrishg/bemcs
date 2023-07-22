close all;
clear variables;


% Declare symbols and build functions to integrate
syms x0 y0 g gx gy gxy gxx gyy nu mu xoffset yoffset fx fy

x = x0 - xoffset;
y = y0 - yoffset;
C = 1 / (4 * pi * (1 - nu));
r = sqrt(x^2 + y^2);
g = -C * log(r);
gx = -C * x / (x^2 + y^2);
gy = -C * y / (x^2 + y^2);
% gxy = C * 2 * x * y / (x^2 + y^2)^2;
% gxx = C * (x^2 - y^2) / (x^2 + y^2)^2;
% gyy = -gxx;
ux = fx / (2 * mu) * ((3 - 4 * nu) * g - x * gx) + fy / (2 * mu) * (-y * gx);
% uy = fx / (2 * mu) * (-x * gy) + fy / (2 * mu) * ((3 - 4 * nu) * g - y * gy);
% sxx = fx * (2 * (1 - nu) * gx - x * gxx) + fy * (2 * nu * gy - y * gxx);
% syy = fx * (2 * nu * gx - x * gyy) + fy * (2 * (1 - nu) * gy - y * gyy);
% sxy = fx * ((1 - 2 * nu) * gy - x * gxy) + fy * ((1 - 2 * nu) * gx - y * gxy);

% Try integrating the xx component of stress along a line
% sxx_definite = int(sxx, x0, [-1.0 1.0]);
ux_definite = int(ux, x0, [-1.0 1.0]);

% Convert to function
% sxx_function_handle = matlabFunction(sxx_definite);
ux_function_handle = matlabFunction(ux_definite);

%% Plot result over a grid
n_pts = 100;
x_vec = linspace(-2, 2, n_pts);
y_vec = linspace(-1.5, 1.5, n_pts);
[x_mat, y_mat] = meshgrid(x_vec, y_vec);
sxx_mat = zeros(size(x_mat));
ux_mat = zeros(size(x_mat));

for i=1:n_pts
    for j=1:n_pts
        % sxx_mat(i, j) = sxx_function_handle(1, 0, 0.25, x_mat(i, j), y_mat(i, j), 0);
        ux_mat(i, j) = ux_function_handle(1, 0, 1, 0.25, x_mat(i, j), y_mat(i, j), 0);
    end
end

% plot results
figure(1),clf
contourf(x_mat, y_mat, ux_mat)
colorbar;
clim([-1,1].*max(abs(ux_mat(:))))
axis("equal")
colormap(parula(10))
title("analytical solution")

%% numerical integration (GL quadrature)

ux_numeric_mat_total = zeros(size(x_mat));
ux_numeric_mat = zeros(size(x_mat));
% point_vec = linspace(-1.0, 1.0, 100);
y0 = 0;

N_gl = 39;
[xk,wk] = calc_gausslegendre_weights(N_gl);
tic
for k=1:numel(xk)
    for i=1:n_pts
        for j=1:n_pts
            [n_ux, ~] = kelvin_point(x_mat(i, j), y_mat(i, j), xk(k), y0, 1, 0, 1, 0.25);
            ux_numeric_mat(i, j) = n_ux*wk(k);
        end
    end
    ux_numeric_mat_total = ux_numeric_mat_total + ux_numeric_mat;
end
toc

figure(2),clf

subplot(3,1,1)
contourf(x_mat, y_mat, ux_mat)
colorbar;
clim([-1,1].*max(abs(ux_mat(:))))
axis("equal")
colormap(parula(10))
title("analytical solution")

subplot(3,1,2)
contourf(x_mat, y_mat, ux_numeric_mat_total)
colorbar;
clim([-1,1].*max(abs(ux_mat(:))))
axis("equal")
colormap(parula(10))
title(['Gauss-Legendre numerical integration of order ' num2str(N_gl)])

% plot residuals as % of analytical solution
subplot(3,1,3)
contourf(x_mat, y_mat, 100.*(ux_mat - ux_numeric_mat_total)./ux_mat)
cb=colorbar;cb.Label.String = '% residuals';
clim([-1,1].*10)
axis("equal")
colormap(parula(10))
set(gca,'Fontsize',15)

%% define kelvin point source function
function [ux, sxx] = kelvin_point(x0, y0, xoffset, yoffset, fx, fy, mu, nu)
    x = x0 - xoffset;
    y = y0 - yoffset;
    C = 1 / (4 * pi * (1 - nu));
    r = sqrt(x^2 + y^2);
    g = -C * log(r);
    gx = -C * x / (x^2 + y^2);
    % gy = -C * y / (x^2 + y^2);
    % gxy = C * 2 * x * y / (x^2 + y^2)^2;
    % gxx = C * (x^2 - y^2) / (x^2 + y^2)^2;
    % gyy = -gxx;
    ux = fx / (2 * mu) * ((3 - 4 * nu) * g - x * gx) + fy / (2 * mu) * (-y * gx);
    sxx = 0;
    % uy = fx / (2 * mu) * (-x * gy) + fy / (2 * mu) * ((3 - 4 * nu) * g - y * gy);
    % sxx = fx * (2 * (1 - nu) * gx - x * gxx) + fy * (2 * nu * gy - y * gxx);
    % syy = fx * (2 * nu * gx - x * gyy) + fy * (2 * (1 - nu) * gy - y * gyy);
    % sxy = fx * ((1 - 2 * nu) * gy - x * gxy) + fy * ((1 - 2 * nu) * gx - y * gxy);
end