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
n_pts = 1001;
x_vec = linspace(-2, 2, n_pts);
y_vec = linspace(-1.5, 1.5, n_pts);

eval_type = 0;% 0 - line, 1 - xy grid
if eval_type == 1
    % create a box grid of points
    [x_mat, y_mat] = meshgrid(x_vec, y_vec);
else
    % only evluate function along a line at y=0
    x_mat = x_vec;
    y_mat = x_vec.*0;
end

sxx_mat = zeros(size(x_mat));
ux_mat = zeros(size(x_mat));

tic
if eval_type == 1
    for i=1:n_pts
        for j=1:n_pts
            % sxx_mat(i, j) = sxx_function_handle(1, 0, 0.25, x_mat(i, j), y_mat(i, j), 0);
            ux_mat(i, j) = ux_function_handle(1, 0, 1, 0.25, x_mat(i, j), 0, y_mat(i, j));
        end
    end
else
    for i = 1:n_pts
        ux_mat(i) = ux_function_handle(1, 0, 1, 0.25, x_mat(i), 0, y_mat(i));
    end
end

toc

% plot results
figure(1),clf
if eval_type == 1
    contourf(x_mat, y_mat, ux_mat)
    colorbar;
    clim([-1,1].*max(abs(ux_mat(:))))
    axis("equal")
    colormap(parula(40))
    title("analytical solution")
else
    plot(x_mat,ux_mat,'-','LineWidth',3), hold on
    plot([-1,1],[0,0],'k-','Linewidth',2)
    axis tight, grid on
    xlabel('x'), ylabel('f(x)')
    ylim([-1,1].*max(abs(ux_mat(:)))*1.1)
    title("analytical solution")
end
%% numerical integration (GL quadrature)

ux_numeric_GL = zeros(size(x_mat));
ux_numeric_GL2 = zeros(size(x_mat));
ux_numeric_mat = zeros(size(x_mat));
% point_vec = linspace(-1.0, 1.0, 100);
y0 = 0;

N_gl = 11;
[xk,wk] = calc_gausslegendre_weights(N_gl);

tic
for k=1:numel(xk)
    [n_ux, ~] = kelvin_point(x_mat, y_mat, xk(k), y0, 1, 0, 1, 0.25);
    ux_numeric_mat = n_ux.*wk(k);
    ux_numeric_GL = ux_numeric_GL + ux_numeric_mat;
end
toc

N_gl = 39;
[xk,wk] = calc_gausslegendre_weights(N_gl);

tic
for k=1:numel(xk)
    [n_ux, ~] = kelvin_point(x_mat, y_mat, xk(k), y0, 1, 0, 1, 0.25);
    ux_numeric_mat = n_ux.*wk(k);
    ux_numeric_GL2 = ux_numeric_GL2 + ux_numeric_mat;
end
toc

figure(2),clf
if eval_type==1
    subplot(3,1,1)
    contourf(x_mat, y_mat, ux_mat)
    colorbar;
    clim([-1,1].*max(abs(ux_mat(:))))
    axis("equal")
    colormap(parula(10))
    title("analytical solution")
    set(gca,'Fontsize',12)

    subplot(3,1,2)
    contourf(x_mat, y_mat, ux_numeric_GL)
    colorbar;
    clim([-1,1].*max(abs(ux_mat(:))))
    axis("equal")
    colormap(parula(10))
    title(['Gauss-Legendre numerical integration of order ' num2str(N_gl)])
    set(gca,'Fontsize',12)

    % plot residuals as % of analytical solution
    subplot(3,1,3)
    contourf(x_mat, y_mat, 100.*(ux_mat - ux_numeric_GL)./ux_mat)
    cb=colorbar;cb.Label.String = '% residuals';
    clim([-1,1].*10)
    axis("equal")
    colormap(bluewhitered(40))
    title('Residuals %(analytical - numerical)')
    set(gca,'Fontsize',12)
else
    plot(x_mat,ux_mat,'-','LineWidth',4), hold on
    plot(x_mat,ux_numeric_GL,'-','LineWidth',1)
    plot(x_mat,ux_numeric_GL2,'k-','LineWidth',1)
    plot([-1,1],[0,0],'k-','Linewidth',2)
    axis tight, grid on
    xlabel('x'), ylabel('u_x(x)')
    % ylim([-1,1].*max(abs(ux_mat(:)))*1.1)
    legend('analytical','GL-quadrature (small N)','GL-quadrature (large N)')
    set(gca,'FontSize',15)
end

%% define kelvin point source function
function [ux, sxx] = kelvin_point(x0, y0, xoffset, yoffset, fx, fy, mu, nu)
    x = x0 - xoffset;
    y = y0 - yoffset;
    C = 1 / (4 * pi * (1 - nu));
    r = sqrt(x.^2 + y.^2);
    g = -C .* log(r);
    gx = -C .* x ./ (x.^2 + y.^2);
    % gy = -C * y / (x^2 + y^2);
    % gxy = C * 2 * x * y / (x^2 + y^2)^2;
    % gxx = C * (x^2 - y^2) / (x^2 + y^2)^2;
    % gyy = -gxx;
    ux = fx / (2 * mu) * ((3 - 4 * nu) .* g - x .* gx) + fy / (2 * mu) .* (-y .* gx);
    sxx = 0;
    % uy = fx / (2 * mu) * (-x * gy) + fy / (2 * mu) * ((3 - 4 * nu) * g - y * gy);
    % sxx = fx * (2 * (1 - nu) * gx - x * gxx) + fy * (2 * nu * gy - y * gxx);
    % syy = fx * (2 * nu * gx - x * gyy) + fy * (2 * (1 - nu) * gy - y * gyy);
    % sxy = fx * ((1 - 2 * nu) * gy - x * gxy) + fy * ((1 - 2 * nu) * gx - y * gxy);
end