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

fx_val = 0;
fy_val = -1;

% provide plotting type (2-d grid or 1-d line)
% 0 - line, 
% 1 - xy grid
eval_type = 1;

n_pts = 101;
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
n_eval = 6; % must be an even number

rectangle_x = 2*ones(n_eval,1);
rectangle_y = linspace(0.1,1,n_eval)';

%% numerical integration (with matlab integral)

ux_vals = zeros(n_pts,n_pts,n_eval);
uy_vals = zeros(n_pts,n_pts,n_eval);

ux_numeric_int = zeros(size(x_mat));
uy_numeric_int = zeros(size(x_mat));

tic
for i = 1:n_eval
    area_source = rectangle_x(i)*rectangle_y(i);
    Rx = rectangle_x(i);
    Ry = rectangle_y(i);
    parfor k=1:numel(x_mat)
        fun_x = @(x0,y0) gf_x(x_mat(k),y_mat(k),x0, y0, fx_val, fy_val, mu_val, nu_val);
        fun_y = @(x0,y0) gf_y(x_mat(k),y_mat(k),x0, y0, fx_val, fy_val, mu_val, nu_val);
                
        ux_numeric_int(k) = integral2(fun_x,-Rx/2,Rx/2,-Ry/2,Ry/2)./area_source;
        uy_numeric_int(k) = integral2(fun_y,-Rx/2,Rx/2,-Ry/2,Ry/2)./area_source;
    end

    ux_vals(:,:,i) = ux_numeric_int;
    uy_vals(:,:,i) = uy_numeric_int;
end
toc


%% plot solutions
figure(1),clf
if eval_type==1
    for i = 1:n_eval
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
    end
end

% solution from last run only
figure(2),clf
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
function ux = gf_x(x0, y0, xoffset, yoffset, fx, fy, mu, nu)
[ux, ~] = kelvin_point(x0, y0, xoffset, yoffset, fx, fy, mu, nu);
end
function uy = gf_y(x0, y0, xoffset, yoffset, fx, fy, mu, nu)
[~, uy] = kelvin_point(x0, y0, xoffset, yoffset, fx, fy, mu, nu);
end

function [ux, uy] = kelvin_point(x0, y0, xoffset, yoffset, fx, fy, mu, nu)
    x = x0 - xoffset;
    y = y0 - yoffset;
    C = 1 / (4 * pi * (1 - nu));
    r = sqrt(x.^2 + y.^2);
    g = -C .* log(r);
    gx = -C .* x ./ (x.^2 + y.^2);
    gy = -C .* y ./ (x.^2 + y.^2);
    ux = fx / (2 * mu) * ((3 - 4 * nu) .* g - x .* gx) + fy / (2 * mu) .* (-y .* gx);
    uy = fx / (2 * mu) * (-x .* gy) + fy / (2 * mu) * ((3 - 4 * nu) .* g - y .* gy);
end