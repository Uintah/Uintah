function [H,R] = LShapedGrid(lambda)
fprintf('Generating static mesh refinement for L-shaped problem, lambda = %e\n',lambda);
alpha       = 2/3;
beta        = 1-alpha/2;
R0          = 0.5;       % Size of domain
level       = 0;
h           = .125;    % Coarsest level meshsize
r           = R0;
work        = 0;
H           = [];
R           = [];
while ((level == 0) || (r > h))
    Rtent       = lambda^(-1/(4*beta))*h^(1/beta);
    r           = min(R0,2*h*round(Rtent/(2*h)));
    if (r == 0)
        break;
    end
    numPoints   = (r/h)^2;
    work        = work+numPoints;
    H = [H; h];
    R = [R; r];
    fprintf('   Level = %2d, R = %e, h = %e, #points = %d\n',...
        level+1,r,h,numPoints);
    level       = level+1;
    h           = h/2;
end
fprintf('Generated grid, #levels = %d, work = %d\n',level,work);
