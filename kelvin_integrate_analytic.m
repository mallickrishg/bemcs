syms x y C r g gx gy gxy gxx gyy ux uy sxx syy sxy nu mu xoffset yoffset fx fy

% x = x - xoffset;
% y = y - yoffset;
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

% Try integrating one of the stresses
% This isn't what we really want but it's conceptually similar
int(int(sxx, y), x)