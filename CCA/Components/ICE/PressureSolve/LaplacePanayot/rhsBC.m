function g = rhsBC(x)
% For Dirichlet B.C. : u = g at domain boundaries
%[xmat1,xmat2] = ndgrid(x,y);

g = zeros(size(x{1}));
%g = xmat1;
