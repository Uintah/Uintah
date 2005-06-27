%=================================================================
% PANAYOT 1991 Discretization Scheme for Diffusion Equation in 2D
%=================================================================

diffusionType = 'jump';
nRange  = 32;%2.^[2:8];
discErr = zeros(size(nRange));

count   = 0;
for n = nRange,
    count   = count+1;
    h       = 1.0/n;

    % Define gridpoint subscripts and indices including ghost cells
    % Vector sub/inds are indicated by capital symbols. Scalar quantities
    % are usually denoted by lowercase symbols.
    I1      = [1:n+2];
    I2      = [1:n+2];
    sz      = [length(I1) length(I2)];
    I       = cell(2,1);
    [I{:}]  = ndgrid(I1,I2);
    indI    = sub2ind(sz,I{:});
    numVars = length(indI(:));

    % Gridpoint physical coordinates
    X1      = (I1-1.5)*h;
    X1(1)   = 0.0;                  % Ghost point
    X1(end) = 1.0;                  % Ghost point
    X2      = (I2-1.5)*h;
    X2(1)   = 0.0;                  % Ghost point
    X2(end) = 1.0;                  % Ghost point
    X       = cell(2,1);
    [X{:}]  = ndgrid(X1,X2);

    % Allocate arrays for solution, RHS
    u = zeros(sz);
    f = zeros(sz);

    %------------------------------------------------------------------
    % Assemble the matrix and RHS of the linear system
    %------------------------------------------------------------------
    Alist   = zeros(0,3);
    b       = zeros(numVars,1);
    unused  = [1:numVars];

    % Assemble the FV equations at interior points
    J1      = [2:n+1];
    J2      = [2:n+1];
    J       = cell(2,1);
    [J{:}]  = ndgrid(J1,J2);
    indJ    = indI(J1,J2);
    indJ    = indJ(:);
    unused  = setdiff(unused,indJ);

    % Control volume is [x1-h1Left,x1+h1Right] x [x2-h2Left,x2+h2Right].
    % Define useful coordinates
    Y1      = (J{1}-1.5)*h;
    Y2      = (J{2}-1.5)*h;
    Y1Left  = Y1 - h;
    Y1Right = Y1 + h;
    Y2Left  = Y2 - h;
    Y2Right = Y2 + h;
    % Adjust nbhr locations near Dirichlet boundaries
    faceXLeft           = find(J{1} == 2);
    Y1Left(faceXLeft)   = Y1Left(faceXLeft) + 0.5*h;
    faceXRight          = find(J{1} == n+1);
    Y1Right(faceXRight) = Y1Right(faceXRight) - 0.5*h;
    faceYLeft           = find(J{2} == 2);
    Y2Left(faceYLeft)   = Y2Left(faceYLeft) + 0.5*h;
    faceYRight          = find(J{2} == n+1);
    Y2Right(faceYRight) = Y2Right(faceYRight) - 0.5*h;

    % Define useful sizes
    h1Left              = Y1        - Y1Left;
    h1Right             = Y1Right   - Y1;
    h2Left              = Y2        - Y2Left;
    h2Right             = Y2Right   - Y2;

    K1                  = 1./(densityIntegral(diffusionType,Y1Left,Y1 ,Y2,1)./(0.5*(h2Left+h2Right)));
    K1Plus              = 1./(densityIntegral(diffusionType,Y1,Y1Right,Y2,1)./(0.5*(h2Left+h2Right)));
    K2                  = 1./(densityIntegral(diffusionType,Y2Left,Y2 ,Y1,2)./(0.5*(h1Left+h1Right)));
    K2Plus              = 1./(densityIntegral(diffusionType,Y2,Y2Right,Y1,2)./(0.5*(h1Left+h1Right)));
    Alist               = [Alist; ...
        % W1Plus flux
        [indJ reshape(indI(J1+1,J2  ),size(indJ)) reshape(-K1Plus,size(indJ))]; ...
        [indJ reshape(indI(J1  ,J2  ),size(indJ)) reshape( K1Plus,size(indJ))]; ...
        % W1 flux
        [indJ reshape(indI(J1  ,J2  ),size(indJ)) reshape( K1    ,size(indJ))]; ...
        [indJ reshape(indI(J1-1,J2  ),size(indJ)) reshape(-K1    ,size(indJ))]; ...
        % W2Plus flux
        [indJ reshape(indI(J1  ,J2+1),size(indJ)) reshape(-K2Plus,size(indJ))]; ...
        [indJ reshape(indI(J1  ,J2  ),size(indJ)) reshape( K2Plus,size(indJ))]; ...
        % W2 flux
        [indJ reshape(indI(J1  ,J2  ),size(indJ)) reshape( K2    ,size(indJ))]; ...
        [indJ reshape(indI(J1  ,J2-1),size(indJ)) reshape(-K2    ,size(indJ))]; ...
        ];
    b(indJ) = 0.25*(h1Left+h1Right).*(h2Left+h2Right).*...
        rhs((J{1}-1.5)*h,(J{2}-1.5)*h);

    % Set Dirichlet boundary conditions
    J1      = [1 n+2];
    J2      = [2:n+1];
    J       = cell(2,1);
    [J{:}]  = ndgrid(J1,J2);
    indJ    = indI(J{:});
    indJ    = indJ(:);
    unused  = setdiff(unused,indJ);
    Alist   = [Alist; ...
        [indJ indJ repmat(1.0,size(indJ))]; ...
        ];
    b(indJ) = 0.0;

    J2      = [1 n+2];
    J1      = [2:n+1];
    J       = cell(2,1);
    [J{:}]  = ndgrid(J1,J2);
    indJ    = indI(J{:});
    indJ    = indJ(:);
    unused  = setdiff(unused,indJ);
    Alist   = [Alist; ...
        [indJ indJ repmat(1.0,size(indJ))]; ...
        ];
    b(indJ) = 0.0;

    % Set the identity operator + zero RHS at unused gridpoints
    Alist   = [Alist; ...
        [unused' unused' repmat(1.0,size(unused'))]; ...
        ];
    b(unused) = 0.0;


    %------------------------------------------------------------------
    % Solve the linear system
    %------------------------------------------------------------------
    A = spconvert(Alist);
    x = A\b;
    x(find(isnan(x))) = 0.0;
    u = reshape(x,sz);
    f = reshape(b,sz);

    %------------------------------------------------------------------
    % Display results
    %------------------------------------------------------------------
    uExact = zeros(sz);
    uExact = exactSolution(diffusionType,X{:});
    discErr(count) = Lpnorm(uExact-u);
    fprintf('n = %3d    L2 error norm = %e\n',n,discErr(count));

    figure(1);
    surf(X1,X2,u);
    title('Discrete Solution u');

%     figure(2);
%     surf(X1,X2,f);
%     title('Right Hand Side f');

    figure(3);
    surf(X1,X2,uExact-u);
    title('Discretization error uExact-u');
end

if (length(nRange) > 1)
    f1  = fac(discErr');
    p1  = log2(f1(end));
    if (length(nRange) > 2)
        f2  = fac(diff(fac(discErr)));
        p2  = p1 + log2(f2(end));
        fprintf('discErr asymptotic = O(h^(%.3f) + h^(%.3f))\n',...
            p1,p2);
    else
        fprintf('discErr asymptotic = O(h^(%.3f))\n',...
            p1);
    end
end
