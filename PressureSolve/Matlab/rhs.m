function f = rhs(x,y)

[xmat1,xmat2] = ndgrid(x,y);

f = 2*pi*pi*sin(pi*xmat1).*sin(pi*xmat2);
