function a = diffusion(diffusion,x,y)
a = zeros(size(x));

switch (diffusionType)
    case 'smooth',
        % Constant diffusion
        a = repmat(1,size(x));

    case 'jump',
        % Piecewise constant diffusion coefficient with a big jump at x=0.5.
        left = find(x < 0.5);
        right = find(x > 0.5);
        a(left) = 1.0;
        a(right) = 1.0e+6;
end
