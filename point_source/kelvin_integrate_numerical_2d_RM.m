clear 

% construct the source
A = [-1,0];
B = [1,0];
C = [0,1];

% unit vectors
nA = [C(2)-B(2);
      B(1)-C(1)]/norm(C-B);
nB = [C(2)-A(2);
      A(1)-C(1)]/norm(C-A);
nC = [B(2)-A(2);
      A(1)-B(1)]/norm(B-A);
  
% check that unit vectors are pointing outward
if (nA'*(A(:)-(B(:)+C(:))/2))>0
    nA=-nA;
end
if (nB'*(B(:)-(A(:)+C(:))/2))>0
    nB=-nB;
end
if (nC'*(C(:)-(A(:)+B(:))/2))>0
    nC=-nC;
end

% parameterized line integral
y2=@(t,A,B) (A(1)+B(1))/2+t*(B(1)-A(1))/2;
y3=@(t,A,B) (A(2)+B(2))/2+t*(B(2)-A(2))/2;

%% Set model parameters 
% Elasticity parameters
mu_val = 1;
nu_val = 0.25;

fx_val = 0;
fy_val = -1;
y0_val = 0;

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

%% need to construct this function correctly
% construct function for integration
Int_func_x = @(t) ...
    (fx_val*nC(1)+fy_val*nC(2)) * norm(B-A) .* Greenfunc_x(y2(t,A,B),y3(t,A,B)) ...
   +(fx_val*nA(1)+fy_val*nA(2)) * norm(C-B) .* Greenfunc_x(y2(t,B,C),y3(t,B,C)) ...
   +(fx_val*nB(1)+fy_val*nB(2)) * norm(A-C) .* Greenfunc_x(y2(t,C,A),y3(t,C,A));

Int_func_y = @(t) ...
    (fx_val*nC(1)+fy_val*nC(2)) * norm(B-A) .* Greenfunc_y(y2(t,A,B),y3(t,A,B)) ...
   +(fx_val*nA(1)+fy_val*nA(2)) * norm(C-B) .* Greenfunc_y(y2(t,B,C),y3(t,B,C)) ...
   +(fx_val*nB(1)+fy_val*nB(2)) * norm(A-C) .* Greenfunc_y(y2(t,C,A),y3(t,C,A));

%% numerical integration (GL quadrature)

ux_numeric_GL = zeros(size(x_mat));
uy_numeric_GL = zeros(size(x_mat));

N_gl = 39;
[xk,wk] = calc_gausslegendre_weights(N_gl);

tic
for k=1:numel(xk)
    [n_ux, n_uy] = kelvin_point(x_mat, y_mat, xk(k), y0_val, fx_val, fy_val, mu_val, nu_val);
    ux_numeric_GL = ux_numeric_GL + n_ux.*wk(k);
    uy_numeric_GL = uy_numeric_GL + n_uy.*wk(k);
end
toc


%% plot comparison of solutions
figure(2),clf
if eval_type==1
    n_skip = 13;
    toplot_n = sqrt(ux_numeric_GL.^2 + uy_numeric_GL.^2);
    contourf(x_mat, y_mat, toplot_n), hold on
    quiver(x_mat(1:n_skip:end), y_mat(1:n_skip:end),ux_numeric_GL(1:n_skip:end),uy_numeric_GL(1:n_skip:end),'r','Linewidth',1)
    colorbar;
    clim([0,1].*max(abs(toplot_n(:))))
    axis("equal")
    title(['Gauss-Legendre numerical integration of order ' num2str(N_gl)])
    set(gca,'Fontsize',12)

else
    subplot(2,1,1)
    plot(x_mat,ux_numeric_GL,'-','LineWidth',2)
    plot([-1,1],[0,0],'k-','Linewidth',2)
    axis tight, grid on
    xlabel('x'), ylabel('u_x')
    set(gca,'FontSize',15)

    subplot(2,1,2)
    plot(x_mat,uy_numeric_GL,'-','LineWidth',2)
    plot([-1,1],[0,0],'k-','Linewidth',2)
    axis tight, grid on
    xlabel('x'), ylabel('u_y')
    set(gca,'FontSize',15)
end

%% define kelvin point source function
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