function g = rhsBC(x)
% For Dirichlet B.C. : u = g at domain boundaries
%[xmat1,xmat2] = ndgrid(x,y);

g = zeros(size(x{1}));
%g = xmat1;

sigma = 0.25;
g = g + exp(-((x{1}-0.5).^2 + (x{2}-0.5).^2)/sigma^2);

%g = log(sqrt((x{1}-0.5).^2 + (x{2}-0.5).^2));

