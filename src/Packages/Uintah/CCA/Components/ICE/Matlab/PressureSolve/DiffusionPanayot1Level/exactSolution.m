function u = exactSolution(diffusionType,x,y)
u = zeros(size(x));

switch (diffusionType)
    case 'smooth',
        % For constant diffusion and smooth force term function.
        u = sin(2*pi*x).*sin(2*pi*y);

    case 'jump',
        % Piecewise constant diffusion coefficient with a big jump at x=0.5.
        left = find(x < 0.5);
        right = find(x > 0.5);
        u = sin(2*pi*x).*sin(2*pi*y);
        u(left) = u(left);
        u(right) = u(right)/1e+6;
end
