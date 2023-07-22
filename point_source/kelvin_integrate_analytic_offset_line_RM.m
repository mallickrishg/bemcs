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

%% numerical integration
% Uniform numeric integration
ux_numeric_mat_total = zeros(size(x_mat));
ux_numeric_mat = zeros(size(x_mat));
point_vec = linspace(-1.0, 1.0, 100);
y0 = 0;

tic
for k=1:numel(point_vec)
    for i=1:n_pts
        for j=1:n_pts
            [n_ux, ~] = kelvin_point(x_mat(i, j), y_mat(i, j), point_vec(k), y0, 1, 0, 1, 0.25);
            ux_numeric_mat(i, j) = n_ux;
        end
    end
    ux_numeric_mat_total = ux_numeric_mat_total + ux_numeric_mat;
end
toc
ux_numeric_mat_total = ux_numeric_mat_total / numel(point_vec);

figure(2),clf
contourf(x_mat, y_mat, ux_numeric_mat_total)
colorbar;
clim([-1,1].*max(abs(ux_numeric_mat_total(:))))
axis("equal")
colormap(parula(10))
title("uniform numerical integration")

%% testing
% Define the function f(x, y)
f = @(x, y) 1./(x.^2 + y.^2).^(0.5);

% Define the limits of integration
x_min = -1;  % Minimum x value
x_max = 1;   % Maximum x value
y_min = -1; % Minimum y value (adjust as needed for the non-point source region)
y_max = 1;  % Maximum y value (adjust as needed for the non-point source region)

% Perform the double integral using integral2
result_point_sources = integral2(@(x, y) f(x, y) .* is_within_point_sources(x, y), x_min, x_max, y_min, y_max, 'AbsTol', 1e-8);
result_non_point_sources = integral2(@(x, y) f(x, y) .* ~is_within_point_sources(x, y), x_min, x_max, y_min, y_max, 'AbsTol', 1e-8);

% Combine the results
result = result_point_sources + result_non_point_sources;

disp(['The result of the double integral is: ', num2str(result)]);

% Function to check if the point (x, y) is within the region of the point sources
function inside = is_within_point_sources(x, y)
    % Define the x-range of the point sources
    x_min_sources = -1;
    x_max_sources = 1;
    
    % Define the y-range of the point sources
    y_min_sources = 0;
    y_max_sources = 0;
    
    % Check if the point (x, y) is within the region of the point sources
    inside = (x >= x_min_sources) & (x <= x_max_sources) & (y >= y_min_sources) & (y <= y_max_sources);
end

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