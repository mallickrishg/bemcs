% using matlab's adaptive integral to solve the Kelvin integral over a 2-d
% triangular source
% 
% Rishav Mallick, 2023, Caltech Seismolab

clear 

% need to provide this triangle in terms of a double integral over x and f(x)
% default experiment sets a triangle with a base from -1<=x<=1, and provide
% y(x) as a function
ymax = @(x) (1-abs(x));% defines the shape of a triangle
% In this script I don't use triangles - but am simple playing with various
% size of rectangles

%% Set model parameters 
% Elasticity parameters
mu_val = 1;
nu_val = 0.25;

% Kelvin force vector (for gravity set fx,fy = 0,-1
fx_val = 0;
fy_val = -1;

% provide plotting type (2-d grid or 1-d line)
% 0 - line, 
% 1 - xy grid
eval_type = 1;

n_pts = 100;
x_vec = linspace(-2, 2, n_pts);
y_vec = linspace(-1.5, 1.5, n_pts);

if eval_type == 1
    % create a box grid of points
    [x_mat, y_mat] = meshgrid(x_vec, y_vec);
else
    % only evluate function along a line at y=0
    x_mat = x_vec;
    y_mat = x_vec.*0;
end
%% rectangle domain
n_eval = 4; % must be an even number

rectangle_x = 2*ones(n_eval,1);
rectangle_y = logspace(-1,0,n_eval)';

%% numerical integration (with matlab integral)

ux_vals = zeros(length(x_mat(:,1)),length(x_mat(1,:)),n_eval);
uy_vals = zeros(length(x_mat(:,1)),length(x_mat(1,:)),n_eval);
sxx_vals = zeros(length(x_mat(:,1)),length(x_mat(1,:)),n_eval);
syy_vals = zeros(length(x_mat(:,1)),length(x_mat(1,:)),n_eval);

ux_numeric_int = zeros(size(x_mat));
uy_numeric_int = zeros(size(x_mat));
sxx_numeric_int = zeros(size(x_mat));
syy_numeric_int = zeros(size(x_mat));

tic
for i = 1:n_eval
    area_source = rectangle_x(i)*rectangle_y(i);
    Rx = rectangle_x(i);
    Ry = rectangle_y(i);
    parfor k=1:numel(x_mat)
        fun_ux = @(x0,y0) gf_ux(x_mat(k),y_mat(k),x0, y0, fx_val, fy_val, mu_val, nu_val);
        fun_uy = @(x0,y0) gf_uy(x_mat(k),y_mat(k),x0, y0, fx_val, fy_val, mu_val, nu_val);
        fun_sxx = @(x0,y0) gf_sxx(x_mat(k),y_mat(k),x0, y0, fx_val, fy_val, mu_val, nu_val);
        fun_syy = @(x0,y0) gf_syy(x_mat(k),y_mat(k),x0, y0, fx_val, fy_val, mu_val, nu_val);

        ux_numeric_int(k) = integral2(fun_ux,-Rx/2,Rx/2,-Ry/2,Ry/2)./area_source;
        uy_numeric_int(k) = integral2(fun_uy,-Rx/2,Rx/2,-Ry/2,Ry/2)./area_source;
        % sxx_numeric_int(k) = integral2(fun_sxx,-Rx/2,Rx/2,-Ry/2,Ry/2)./area_source;
        % syy_numeric_int(k) = integral2(fun_syy,-Rx/2,Rx/2,-Ry/2,Ry/2)./area_source;
    end

    ux_vals(:,:,i) = ux_numeric_int;
    uy_vals(:,:,i) = uy_numeric_int;
    % sxx_vals(:,:,i) = sxx_numeric_int;
    % syy_vals(:,:,i) = syy_numeric_int;
end
toc


%% plot solutions
figure(11),clf
figure(12),clf
if eval_type==1
    for i = 1:n_eval
        figure(11)
        subplot(n_eval/2,2,i)
        n_skip = 23;
        
        ux = squeeze(ux_vals(:,:,i));
        uy = squeeze(uy_vals(:,:,i));

        toplot_n = sqrt(ux.^2 + uy.^2);
        contourf(x_mat, y_mat, toplot_n,5), hold on
        quiver(x_mat(1:n_skip:end), y_mat(1:n_skip:end),ux(1:n_skip:end),uy(1:n_skip:end),'r','Linewidth',1)
        cb=colorbar;cb.Label.String = 'Displacement |U|';
        clim([0,1].*0.1)
        axis("equal")
        title(['Source area = ' num2str(rectangle_x(i)*rectangle_y(i))])
        colormap(sky(10))
        xlabel('x'), ylabel('y')
        set(gca,'Fontsize',15)

        figure(12)
        subplot(n_eval/2,2,i)

        toplot_n = squeeze(sxx_vals(:,:,i));
        contourf(x_mat, y_mat, toplot_n,5), hold on
        cb=colorbar;cb.Label.String = '\sigma_{xx}';
        clim([-1,1].*0.02)
        axis("equal")
        title(['Source area = ' num2str(rectangle_x(i)*rectangle_y(i))])
        colormap(parula(100))
        xlabel('x'), ylabel('y')
        set(gca,'Fontsize',15)
    end
else
    cspec = cool(n_eval);
    for i = 1:n_eval        
        ux = squeeze(ux_vals(:,:,i));
        uy = squeeze(uy_vals(:,:,i));
        subplot(2,1,1)
        plot(x_mat,ux,'-','LineWidth',2,'Color',cspec(i,:)), hold on
        axis tight, grid on
        xlabel('x'), ylabel('u_x')
        set(gca,'FontSize',15)

        subplot(2,1,2)
        plot(x_mat,uy,'-','LineWidth',2,'Color',cspec(i,:)), hold on
        axis tight, grid on
        xlabel('x'), ylabel('u_y')
        set(gca,'FontSize',15)
    end
end

% solution from last run only
figure(21),clf
if eval_type==1
    n_skip = 13;
    toplot_n = sqrt(ux_numeric_int.^2 + uy_numeric_int.^2);
    contourf(x_mat, y_mat, toplot_n), hold on
    quiver(x_mat(1:n_skip:end), y_mat(1:n_skip:end),ux_numeric_int(1:n_skip:end),uy_numeric_int(1:n_skip:end),'r','Linewidth',1)
    colorbar;
    clim([0,1].*max(abs(toplot_n(:))))
    axis("equal")
    title('Adaptive quadrature')
    set(gca,'Fontsize',12)

else
    subplot(2,1,1)
    plot(x_mat,ux_numeric_int,'-','LineWidth',2)
    axis tight, grid on
    xlabel('x'), ylabel('u_x')
    set(gca,'FontSize',15)

    subplot(2,1,2)
    plot(x_mat,uy_numeric_int,'-','LineWidth',2)
    axis tight, grid on
    xlabel('x'), ylabel('u_y')
    set(gca,'FontSize',15)
end

%% define kelvin point source function
function ux = gf_ux(x0, y0, xoffset, yoffset, fx, fy, mu, nu)
[ux, ~, ~, ~, ~] = kelvin_point(x0, y0, xoffset, yoffset, fx, fy, mu, nu);
end
function uy = gf_uy(x0, y0, xoffset, yoffset, fx, fy, mu, nu)
[~, uy, ~, ~, ~] = kelvin_point(x0, y0, xoffset, yoffset, fx, fy, mu, nu);
end
function sxx = gf_sxx(x0, y0, xoffset, yoffset, fx, fy, mu, nu)
[~, ~, sxx, ~, ~] = kelvin_point(x0, y0, xoffset, yoffset, fx, fy, mu, nu);
end
function syy = gf_syy(x0, y0, xoffset, yoffset, fx, fy, mu, nu)
[~, ~, ~, syy, ~] = kelvin_point(x0, y0, xoffset, yoffset, fx, fy, mu, nu);
end
function sxy = gf_sxy(x0, y0, xoffset, yoffset, fx, fy, mu, nu)
[~, ~, ~, ~, sxy] = kelvin_point(x0, y0, xoffset, yoffset, fx, fy, mu, nu);
end

function [ux, uy, sxx, syy, sxy] = kelvin_point(x0, y0, xoffset, yoffset, fx, fy, mu, nu)
    x = x0 - xoffset;
    y = y0 - yoffset;
    C = 1 / (4 * pi * (1 - nu));
    r = sqrt(x.^2 + y.^2);
    g = -C .* log(r);
    gx = -C .* x ./ (x.^2 + y.^2);
    gy = -C .* y ./ (x.^2 + y.^2);
    gxy = C .* 2 .* x .* y ./ (x.^2 + y.^2).^2;
    gxx = C .* (x.^2 - y.^2) ./ (x.^2 + y.^2).^2;
    gyy = -gxx;
    ux = fx / (2 * mu) * ((3 - 4 * nu) .* g - x .* gx) + fy / (2 * mu) .* (-y .* gx);
    uy = fx / (2 * mu) * (-x .* gy) + fy / (2 * mu) * ((3 - 4 * nu) .* g - y .* gy);
    sxx = fx .* (2 * (1 - nu) .* gx - x .* gxx) + fy .* (2 * nu .* gy - y .* gxx);
    syy = fx .* (2 * nu .* gx - x .* gyy) + fy .* (2 * (1 - nu) .* gy - y .* gyy);
    sxy = fx .* ((1 - 2 * nu) .* gy - x .* gxy) + fy .* ((1 - 2 * nu) .* gx - y .* gxy);
end