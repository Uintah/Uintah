function r = densityIntegral(diffusionType,x1,x2,y,dim)
% Integral of 1/a along the curve [x1,x2]. Here x1,x2 are x-coordinates if
% dim=1, and y is the fixed y-coordinate along the curve. If dim=2, x1,x2
% are y-coordinates and y is the fixed x-coordinate along the curve.

r = zeros(size(x1));

switch (diffusionType)
    case 'smooth',

        % Constant diffusion
        r = x2-x1;

    case 'jump',
        % Piecewise constant diffusion coefficient with a big jump at x=0.5.
        switch (dim)
            case 1,
                bothLeft        = find((x1 < 0.5) & (x2 < 0.5));
                r(bothLeft)     = x2(bothLeft)-x1(bothLeft);

                crossJump       = find((x1 < 0.5) & (x2 > 0.5));
                r(crossJump)    = (0.5-x1(crossJump)) + (x2(crossJump)-0.5)/1e+6;
                
                bothRight       = find(x1 > 0.5);
                r(bothRight)    = (x2(bothRight)-x1(bothRight))/1e+6;
            case 2,
                r = x2-x1;
        end
end
