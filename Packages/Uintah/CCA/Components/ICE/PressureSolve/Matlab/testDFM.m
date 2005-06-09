%TESTDFM  Test Distributive Flux Matching (DFM) discretization.
%   We test DFM discretization error for a simple 2D Poisson problem with a
%   known solution. We prepare an AMR grid with two levels: a global level
%   1, and a local level 2 patch around the center of the domain, where the
%   solution u has more variations. DFM is a cell-centered, finite volume,
%   symmetric discretization on the composite AMR grid.
%
%   See also: ?.

%-------------------------------------------------------------------------
% Set up grid (AMR levels, patches)
%-------------------------------------------------------------------------
fprintf('-------------------------------------------------------------------------\n');
fprintf(' Set up grid\n');
fprintf('-------------------------------------------------------------------------\n');

grid.maxLevels  	= 5;
grid.maxPatches  	= 5;
grid.level          = cell(grid.maxLevels,1);
grid.numLevels      = 0;

%--------------- Level 1: global coarse grid -----------------
dim         = 2;
domainSize  = [1.0 1.0];
resolution  = [256 256];

[grid,k] = addGridLevel(grid,'meshsize',domainSize./resolution);
[grid,q1] = addGridPatch(grid,k,ones(1,dim),resolution,-1);     % One global patch

% %--------------- Level 2: local fine grid around center of domain -----------------
% 
% [grid,k] = addGridLevel(grid,'refineRatio',[2 2]);
% [grid,q2] = addGridPatch(grid,k,[6 6],[13 13],q1);              % Local patch around the domain center

%-------------------------------------------------------------------------
% Set up stencils
%-------------------------------------------------------------------------

for k = 1:grid.numLevels,
    grid.level{k}.stencilOffsets = [...                      % -Laplacian stencil non-zero structure
        [ 0 0 ]; ...
        [-1  0]; ...
        [ 1  0]; ...
        [ 0 -1]; ...
        [ 0  1]...
        ];
end

%-------------------------------------------------------------------------
% Set up the the matrix
%-------------------------------------------------------------------------
fprintf('-------------------------------------------------------------------------\n');
fprintf(' Set up the the matrix\n');
fprintf('-------------------------------------------------------------------------\n');

[A,b] = setupOperator(grid);        % Structured part (stencils on each patch)

%-------------------------------------------------------------------------
% Solve the linear system
%-------------------------------------------------------------------------
fprintf('-------------------------------------------------------------------------\n');
fprintf(' Solve the linear system\n');
fprintf('-------------------------------------------------------------------------\n');

x = A\b;                            % Direct solver
u = sparseToAMR(x,grid);           % Translate the solution vector to patch-based

%-------------------------------------------------------------------------
% Computed exact solution vector, patch-based
%-------------------------------------------------------------------------
fprintf('-------------------------------------------------------------------------\n');
fprintf(' Compute exact solution, plot\n');
fprintf('-------------------------------------------------------------------------\n');

uExact = exactSolutionAMR(grid);

% Plot discretization error

k=1;
q=1;

figure(1);
clf;
surf(u{k}{q});
title('Discrete solution');

figure(2);
clf;
surf(uExact{k}{q});
title('Exact solution');

figure(3);
clf;
surf(u{k}{q}-uExact{k}{q});
title('Discretization error');
shg;

fprintf('L2 discretization error = %e\n',Lpnorm(u{k}{q}-uExact{k}{q}));

