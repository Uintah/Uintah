%TESTDFM  Test Distributive Flux Matching (DFM) discretization.
%   We test DFM discretization error for a simple 2D Poisson problem with a
%   known solution. We prepare an AMR grid with two levels: a global level
%   1, and a local level 2 patch around the center of the domain, where the
%   solution u has more variations. DFM is a cell-centered, finite volume,
%   symmetric discretization on the composite AMR grid.
%
%   See also: ?.

setupGrid           = 1;
solveSystem         = 1;
plotSolution        = 1;

%-------------------------------------------------------------------------
% Set up grid (AMR levels, patches)
%-------------------------------------------------------------------------
if (setupGrid)
    fprintf('-------------------------------------------------------------------------\n');
    fprintf(' Set up grid & system\n');
    fprintf('-------------------------------------------------------------------------\n');
    tStartCPU           = cputime;
    tStartElapsed       = clock;

    grid.maxLevels  	= 5;
    grid.maxPatches  	= 5;
    grid.level          = cell(grid.maxLevels,1);
    grid.numLevels      = 0;
    grid.domainSize     = [1.0 1.0];                                % Domain is from [0.,0.] to [1.,1.]
    grid.dim            = length(grid.domainSize);
    A                   = [];
    b                   = [];
    
    %--------------- Level 1: global coarse grid -----------------

    resolution          = [4 4];
    [grid,k]            = addGridLevel(grid,'meshsize',grid.domainSize./resolution);
    
    [grid,q1]           = addGridPatch(grid,k,ones(1,grid.dim),resolution,-1);     % One global patch
    [A,b]               = updateSystem(grid,k,q1,A,b);

    %--------------- Level 2: local fine grid around center of domain -----------------

    [grid,k]            = addGridLevel(grid,'refineRatio',[2 2]);
    [grid,q2]           = addGridPatch(grid,k,[3 3],[6 6],q1);              % Local patch around the domain center
    [A,b]               = updateSystem(grid,k,q2,A,b);


    tCPU        = cputime - tStartCPU;
    tElapsed    = etime(clock,tStartElapsed);
    fprintf('CPU time     = %f\n',tCPU);
    fprintf('Elapsed time = %f\n',tElapsed);

    printGrid(grid);
end

%-------------------------------------------------------------------------
% Solve the linear system
%-------------------------------------------------------------------------
if (solveSystem)
    fprintf('-------------------------------------------------------------------------\n');
    fprintf(' Solve the linear system\n');
    fprintf('-------------------------------------------------------------------------\n');
    tStartCPU        = cputime;
    tStartElapsed    = clock;
    x = A\b;                            % Direct solver
    u = sparseToAMR(x,grid);           % Translate the solution vector to patch-based
    tCPU        = cputime - tStartCPU;
    tElapsed    = etime(clock,tStartElapsed);
    fprintf('CPU time     = %f\n',tCPU);
    fprintf('Elapsed time = %f\n',tElapsed);
end

%-------------------------------------------------------------------------
% Computed exact solution vector, patch-based
%-------------------------------------------------------------------------
if (plotSolution)
    fprintf('-------------------------------------------------------------------------\n');
    fprintf(' Compute exact solution, plot\n');
    fprintf('-------------------------------------------------------------------------\n');
    tStartCPU        = cputime;
    tStartElapsed    = clock;

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

    tCPU        = cputime - tStartCPU;
    tElapsed    = etime(clock,tStartElapsed);
    fprintf('CPU time     = %f\n',tCPU);
    fprintf('Elapsed time = %f\n',tElapsed);

    fprintf('L2 discretization error = %e\n',Lpnorm(u{k}{q}-uExact{k}{q}));
end
