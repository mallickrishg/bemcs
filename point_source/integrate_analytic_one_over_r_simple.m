close all;
clear all;

% Declare symbols and build functions to integrate
syms xsrc ysrc xobs yobs
x = xsrc - xobs;
y = ysrc - yobs;
r = sqrt(x^2 + y^2);
f = 1 / r;

% Inner integral over y
s_inner_indef = int(f, yobs);
s_inner_lower_limit = subs(s_inner_indef, yobs, 0);
s_inner_upper_limit = subs(s_inner_indef, yobs, 1 - xsrc);
s_inner_def = s_inner_upper_limit - s_inner_lower_limit;

% Outer integral over x
s_outer_indef = int(s_inner_def, xobs);
s_outer_lower_limit = subs(s_outer_indef, xobs, 0);
s_outer_upper_limit = subs(s_outer_indef, xobs, 1);
s_outer_def = s_outer_upper_limit - s_outer_lower_limit;

% Convert to symbolic integral to function
s_function_handle = matlabFunction(s_outer_def);

% Evaluate on a grid
x_vec = linspace(-1.0, 2.0, 201);
y_vec = linspace(-1.0, 2.0, 201);
[x_mat, y_mat] = meshgrid(x_vec, y_vec);
f_mat = 1 ./ sqrt(x_mat.^2 + y_mat.^2);
s_mat = real(s_function_handle(x_mat, y_mat));

% Plot single kernel and integrated kernel
figure("Position", [0 0 1000 500]);
n_contours = 20;
subplot(1, 2, 1)
contourf(x_mat, y_mat, f_mat, n_contours);
colorbar;
axis("equal")
title("kernel to integrate at origin")
subplot(1, 2, 2)
contourf(x_mat, y_mat, s_mat, n_contours);
colorbar;
axis("equal")
title("kernel integrated over unit square")

