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

% Try integrating the xx component of stress

% This is the definite integral I want but it doesn't evaluate the
% integration does not return a solution but rather only the expression
% that is passed.
% int(int(sxx, yoffset, [0 1-xoffset]), xoffset, [0 1])

% Trying this as a set of smaller steps
% Indefinite integral over yoffset (inner integral)
sxx_indefinite_yoffset = int(sxx, yoffset);
sxx_indefinite_yoffset_lower_limit = subs(sxx_indefinite_yoffset, yoffset, 0);
sxx_indefinite_yoffset_upper_limit = subs(sxx_indefinite_yoffset, yoffset, 1-xoffset);
sxx_definite_yoffset = sxx_indefinite_yoffset_upper_limit - sxx_indefinite_yoffset_lower_limit;

% Indefinite integral over xoffset (outer integral)
sxx_definite_yoffset_indefinite_xoffset = int(sxx_definite_yoffset, xoffset);
sxx_definite_yoffset_indefinite_xoffset_lower_limit = subs(sxx_definite_yoffset_indefinite_xoffset, xoffset, 0);
sxx_definite_yoffset_indefinite_xoffset_upper_limit = subs(sxx_definite_yoffset_indefinite_xoffset, xoffset, 1);
sxx_definite_yoffset_definite_xoffset = sxx_definite_yoffset_indefinite_xoffset_upper_limit - sxx_definite_yoffset_indefinite_xoffset_lower_limit;

% Convert to function and plot result over a grid
sxx_function_handle = matlabFunction(sxx_definite_yoffset_definite_xoffset);

% Convert to string and replace matlab notation with python notation
