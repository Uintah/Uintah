%function [grid,A,b,T,TI,u] = testDisc(twoLevel)
%TESTDISC  Test pressure equation discretization.
%   We test pressure equation discretization error for a
%   simple 2D Poisson problem with a known solution.
%   We prepare an AMR grid with two levels: a global level
%   1, and a local level 2 patch around the center of the domain, where the
%   solution u has more variations. The scheme is a cell-centered,
%   finite volume, symmetric discretization on the composite AMR grid.
%   We study the discretization error vs. meshsize on a sequence of
%   increasingly finer composite grids with the same refinement "pattern".
%
%   See also: ADDGRIDLEVEL, ADDGRIDPATCH, TESTADAPTIVE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

globalParams;

initParam;                  % Initialize parameters structure

if (param.profile)
    profile on -detail builtin;                                % Enable profiling
end
totalStartCPU           = cputime;
totalStartElapsed       = clock;

out(1,'=========================================================================\n');
out(1,' Testing discretization accuracy on increasingly finer grids\n');
out(1,'=========================================================================\n');

%=========================================================================
% Run discretization on a sequence of successively finer grids
%=========================================================================
success                     = mkdir('.',param.outputDir);
errNorm                     = zeros(length(param.numCellsRange),5);

for count = 1:length(param.numCellsRange)
    %pause
    pack;
    numCells = param.numCellsRange(count);
    out(1,'#### nCells = %d ####\n',numCells);
    %-------------------------------------------------------------------------
    % Set up grid (AMR levels, patches)
    %-------------------------------------------------------------------------
    if (param.setupGrid)
        out(2,'-------------------------------------------------------------------------\n');
        out(2,' Set up grid & system\n');
        out(2,'-------------------------------------------------------------------------\n');
        out(1,'Setting up grid\n');
        tStartCPU           = cputime;
        tStartElapsed       = clock;

        grid                = [];
        grid.dim            = length(param.domainSize);
        grid.domainSize     = param.domainSize;
        grid.maxLevels  	= param.maxLevels;
        grid.maxPatches  	= param.maxPatches;
        grid.numLevels      = 0;
        grid.level          = cell(grid.maxLevels,1);
        A                   = [];
        b                   = [];
        T                   = [];
        TI                  = [];

        %--------------- Level 1: global coarse grid -----------------

        resolution          = [numCells numCells];
        [grid,k]            = addGridLevel(grid,'meshsize',grid.domainSize./resolution);
        [grid,q1]           = addGridPatch(grid,k,ones(1,grid.dim),resolution,-1);     % One global patch
        for q = 1:grid.level{k}.numPatches,
            [grid,A,b,T,TI]      = updateSystem(grid,k,q,A,b,T,TI);
        end

        %--------------- Level 2: local fine grid around center of domain -----------------

        if (param.twoLevel)
            [grid,k]            = addGridLevel(grid,'refineRatio',[2 2]);
            switch (param.twoLevelType)
                case 'global',
                    % Cover the entire domain
                    [grid,q2]  = addGridPatch(grid,k,ones(1,grid.dim),2*resolution,q1);              % Local patch around the domain center
                case 'centralHalf',
                    % Cover central half of the domain
                    [grid,q2]  = addGridPatch(grid,k,resolution/2 + 1,3*resolution/2,q1);              % Local patch around the domain center
                case 'centralQuarter',
                    % Cover central quarter of the domain
                    [grid,q2]  = addGridPatch(grid,k,3*resolution/4 + 1,5*resolution/4,q1);              % Local patch around the domain center
                case 'leftHalf',
                    % Cover left half of the domain in x1
                    ilower      = ones(size(resolution));
                    iupper      = 2*resolution;
                    iupper(1)   = resolution(1);
                    [grid,q2]  = addGridPatch(grid,k,ilower,iupper,q1);
                case 'rightHalf',
                    % Cover right half of the domain in x1
                    ilower      = ones(size(resolution));
                    ilower(1)   = resolution(1) + 1;
                    iupper      = 2*resolution;
                    [grid,q2]  = addGridPatch(grid,k,ilower,iupper,q1);
                case 'nearXMinus',
                    % A patch next to x-minus boundary, covers the central
                    % half of it, and extends to half of the domain in x.
                    ilower      	= ones(size(resolution));
                    ilower(2:end)	= resolution(2:end)/2 + 1;
                    iupper          = ilower + resolution - 1;
                    iupper(1)       = ilower(1) + resolution(1)/2 - 1;
                    [grid,q2]  = addGridPatch(grid,k,ilower,iupper,q1);
                case 'centralHalf2Patches',
                    % Two fine patches next to each other at the center of the
                    % domain
                    ilower = resolution/2 + 1;
                    iupper = 3*resolution/2;
                    iupper(1) = resolution(1);
                    [grid,q2]  = addGridPatch(grid,k,ilower,iupper,q1);
                    ilower = resolution/2 + 1;
                    iupper = 3*resolution/2;
                    ilower(1) = resolution(1)+1;
                    [grid,q3]  = addGridPatch(grid,k,ilower,iupper,q1);
                case 'centralQuarter2Patches',
                    % Two fine patches next to each other at the central
                    % quarter of the domain
                    ilower = 3*resolution/4 + 1;
                    iupper = 5*resolution/4;
                    iupper(1) = resolution(1);
                    [grid,q2]  = addGridPatch(grid,k,ilower,iupper,q1);
                    ilower = 3*resolution/4 + 1;
                    iupper = 5*resolution/4;
                    ilower(1) = resolution(1)+1;
                    [grid,q3]  = addGridPatch(grid,k,ilower,iupper,q1);
                otherwise,
                    error('Unknown two level type');
            end

            for q = 1:grid.level{k}.numPatches,
                [grid,A,b,T,TI]      = updateSystem(grid,k,q,A,b,T,TI);
            end
        end

        %--------------- Level 3: yet local fine grid around center of domain -----------------
        if ((param.twoLevel) & (param.threeLevel))
            [grid,k]   = addGridLevel(grid,'refineRatio',[2 2]);
            switch (param.threeLevelType)
                case 'centralHalf',
                    % Cover central half of the domain
                    [grid,q3]  = addGridPatch(grid,k,3*resolution/2 + 1,5*resolution/2,q2);              % Local patch around the domain center
                case 'centralHalfOfcentralQuarter',
                    % Cover central half of the central quarter of the domain
                    [grid,q3]  = addGridPatch(grid,k,7*resolution/4 + 1,9*resolution/4,q2);              % Local patch around the domain center
                    %                [grid,q3]  = addGridPatch(grid,k,15*resolution/4 + 1,17*resolution/4,q2);              % Local patch around the domain center
                otherwise,
                    error('Unknown three level type');
            end

            for q = 1:grid.level{k}.numPatches,
                [grid,A,b,T,TI]      = updateSystem(grid,k,q,A,b,T,TI);
            end
        end

        tCPU        = cputime - tStartCPU;
        tElapsed    = etime(clock,tStartElapsed);
        out(2,'CPU time     = %f\n',tCPU);
        out(2,'Elapsed time = %f\n',tElapsed);
        if (param.verboseLevel >= 2)
            printGrid(grid);
        end
    end

    %-------------------------------------------------------------------------
    % Solve the linear system
    %-------------------------------------------------------------------------
    if (param.solveSystem)
        out(2,'-------------------------------------------------------------------------\n');
        out(2,' Solve the linear system\n');
        out(2,'-------------------------------------------------------------------------\n');
        out(1,'Solving system\n');
        u = solveSystem(A,b,grid,TI);
    end

    %-------------------------------------------------------------------------
    % Computed exact solution vector, patch-based
    %-------------------------------------------------------------------------
    out(2,'-------------------------------------------------------------------------\n');
    out(2,' Compute exact solution, plot\n');
    out(2,'-------------------------------------------------------------------------\n');
    tStartCPU        = cputime;
    tStartElapsed    = clock;

    % Plot grid
    if (grid.totalVars <= 200)
        plotGrid(grid,sprintf('%s/grid%d.eps',param.outputDir,numCells),1,0,0,0);
    end

    % Plot and print discretization error at all patches
    uExact = exactSolutionAMR(grid,T,TI);
    tau = sparseToAMR(b-A*AMRToSparse(uExact,grid,T,1),grid,TI,0);
    f = sparseToAMR(b,grid,TI,0);

    % AMR grid norms
    err = cell(size(u));
    for k = 1:grid.numLevels,
        level = grid.level{k};
        for q = 1:grid.level{k}.numPatches,
            err{k}{q} = uExact{k}{q}-u{k}{q};
        end
    end
    temp    = AMRToSparse(err,grid,T,1);
    err     = sparseToAMR(temp,grid,TI,0);
    errNorm(count,:) = [ ...
        numCells ...
        normAMR(grid,err,'L2') ...
        normAMR(grid,err,'max') ...
        normAMR(grid,err,'H1') ...
        normAMR(grid,err,'H1max') ...
        ];
    out(1,'#vars = %7d  L2=%.3e  max=%.3e  H1=%.3e  H1max=%.3e\n',grid.totalVars,errNorm(count,2:end));

    if (param.saveResults)
        saveResults(errNorm(1:count,:));
    end

    if (param.plotResults)
        plotResults(grid,u,uExact,tau,numCells);
    end
    tCPU        = cputime - tStartCPU;
    tElapsed    = etime(clock,tStartElapsed);
    out(2,'CPU time     = %f\n',tCPU);
    out(2,'Elapsed time = %f\n',tElapsed);

end

totalCPU        = cputime - totalStartCPU;
totalElapsed    = etime(clock,totalStartElapsed);
out(1,'CPU time     = %f\n',totalCPU);
out(1,'Elapsed time = %f\n',totalElapsed);
if (param.profile)
    profile report;                             % Generate timing profile report
end
