clear; clc; close all;
d = [ ...
  1.00E+00	1.00E+00	1.00E+00	1.00E+00	1.00E+00	1.00E+00	1.00E+00	1.00E+00	1.00E+00	1.00E+00	1.00E+00	1.00E+00
9.25E-01	9.68E-01	9.50E-01	9.52E-01	8.77E-01	9.71E-01	9.78E-01	9.67E-01	9.88E-01	9.52E-01	9.52E-01	9.93E-01
7.74E-01	8.53E-01	8.88E-01	8.84E-01	8.28E-01	9.06E-01	9.38E-01	9.22E-01	8.14E-01	9.23E-01	9.42E-01	8.73E-01
];

ncore = [1 2 4];
npts = [ 4.0E+03	8.0E+03	1.6E+04	3.2E+04	2.0E+04	4.0E+04	8.0E+04	1.6E+05	2.0E+05	4.0E+05	8.0E+05	1.6E+06 ];

NC=repmat(ncore',1,12);
NP=repmat(npts,3,1);

surf(NP,NC,d);
shading interp;
colorbar;
axis tight;
xlabel('DOF'); ylabel('# Cores'); zlabel('Parallel Efficiency');