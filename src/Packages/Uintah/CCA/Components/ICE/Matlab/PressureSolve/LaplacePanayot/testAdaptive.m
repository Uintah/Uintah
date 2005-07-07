%function testAdaptive
%TESTAdaptive  Test pressure equation discretization with adaptive mesh refinement.
%   We test pressure equation discretization error for a simple 2D Poisson problem with a
%   known solution. We prepare an AMR grid with two levels: a global level
%   1, and a local level 2 patch around the center of the domain, where the
%   solution u has more variations. The scheme is a cell-centered, finite volume,
%   symmetric discretization on the composite AMR grid.
%
%   See also: ADDGRIDLEVEL, ADDGRIDPATCH.

globalParams;

%=========================================================================
% Initialize parameters struct
%=========================================================================
param                       = [];

param.problemType           = 'Lshaped'; %'ProblemB'; %'quadratic'; %'Lshaped'; %
param.outputDir             = 'ProblemA_1Level';
param.plotResults           = 0;
param.saveResults           = 1;
param.verboseLevel          = 0;

%=========================================================================
% Create a sequence of adaptive mesh refinement levels
%=========================================================================
success                     = mkdir('.',param.outputDir);

% Create an empty grid
resolution          = [4 4];                                    % Coarsest level resolution
grid.maxLevels  	= 3;
grid.maxPatches  	= 3;
grid.level          = cell(grid.maxLevels,1);
grid.numLevels      = 0;
grid.domainSize     = [1.0 1.0];                                % Domain is from [0.,0.] to [1.,1.]
grid.dim            = length(grid.domainSize);
A                   = [];
b                   = [];
T                   = [];
TI                  = [];

% Prepare quantities to be saved for each refinement stage
uAll                = cell(grid.maxLevels,1);
AAll                = cell(grid.maxLevels,1);
bAll                = cell(grid.maxLevels,1);
TAll                = cell(grid.maxLevels,1);
TIAll               = cell(grid.maxLevels,1);
gridAll             = cell(grid.maxLevels,1);
errNorm             = zeros(grid.maxLevels,4);
patchID             = cell(grid.maxLevels,1);

for numLevels = 1:grid.maxLevels,
    pack;
    fprintf('#### numLevels = %d ####\n',numLevels);
    
    %-------------------------------------------------------------------------
    % Set up grid (AMR levels, patches)
    %-------------------------------------------------------------------------
    if (param.verboseLevel >= 1)
        fprintf('-------------------------------------------------------------------------\n');
        fprintf(' Set up grid & system\n');
        fprintf('-------------------------------------------------------------------------\n');
    end
    tStartCPU           = cputime;
    tStartElapsed       = clock;

    if (numLevels == 1)
        [grid,k]            = addGridLevel(grid,'meshsize',grid.domainSize./resolution);
        [grid,q]            = addGridPatch(grid,k,ones(1,grid.dim),resolution,-1);     % One global patch
        patchID{numLevels}  = q;
    else
        [grid,k]            = addGridLevel(grid,'refineRatio',[2 2]);
        if (numLevels <= 3)
            [grid,q]       = addGridPatch(grid,k,ones(1,grid.dim),2.^(numLevels-1)*resolution,patchID{k-1});
            patchID{numLevels}  = q;
        else
        end
    end

    for q = 1:grid.level{k}.numPatches,
        [grid,A,b,T,TI]      = updateSystem(grid,k,q,A,b,T,TI);
    end

    tCPU        = cputime - tStartCPU;
    tElapsed    = etime(clock,tStartElapsed);
    if (param.verboseLevel >= 1)
        fprintf('CPU time     = %f\n',tCPU);
        fprintf('Elapsed time = %f\n',tElapsed);
        printGrid(grid);
    end

    %-------------------------------------------------------------------------
    % Solve the linear system
    %-------------------------------------------------------------------------
    if (param.verboseLevel >= 1)
        fprintf('-------------------------------------------------------------------------\n');
        fprintf(' Solve the linear system\n');
        fprintf('-------------------------------------------------------------------------\n');
    end
    tStartCPU       = cputime;
    tStartElapsed   = clock;
    x               = A\b;                            % Direct solver
    u               = sparseToAMR(x,grid,TI,1);           % Translate the solution vector to patch-based
    tCPU            = cputime - tStartCPU;
    tElapsed        = etime(clock,tStartElapsed);
    if (param.verboseLevel >= 1)
        fprintf('CPU time     = %f\n',tCPU);
        fprintf('Elapsed time = %f\n',tElapsed);
    end

    %-------------------------------------------------------------------------
    % Computed exact solution vector, patch-based and compute
    % discretization error
    %-------------------------------------------------------------------------
    if (param.verboseLevel >= 1)
        fprintf('-------------------------------------------------------------------------\n');
        fprintf(' Compute exact solution, plot\n');
        fprintf('-------------------------------------------------------------------------\n');
    end
    tStartCPU        = cputime;
    tStartElapsed    = clock;

    % Plot grid
    if (grid.totalVars <= 2000)
        plotGrid(grid,sprintf('%s/grid%d.eps',param.outputDir,numLevels),0,0,0,0);
    end

    % Plot and print discretization error at all patches
    uExact = exactSolutionAMR(grid,T,TI);
    tau = sparseToAMR(b-A*AMRToSparse(uExact,grid,T,1),grid,TI,0);
    f = sparseToAMR(b,grid,TI,0);
    fig = 0;

    % AMR grid norms
    err = cell(size(u));
    for k = 1:grid.numLevels,
        level = grid.level{k};
        for q = 1:grid.level{k}.numPatches,
            err{k}{q} = uExact{k}{q}-u{k}{q};
        end
    end
    temp    = AMRToSparse(err,grid,T,1);
    err     = SparseToAMR(temp,grid,TI,0);
    errNorm(numLevels,:) = [ ...
        normAMR(grid,err,'L2') ...
        normAMR(grid,err,'max') ...
        normAMR(grid,err,'H1') ...
        normAMR(grid,err,'H1max') ...
        ];
    fprintf('L2=%.3e  max=%.3e  H1=%.3e  H1max=%.3e\n',errNorm(numLevels,:));

    tCPU        = cputime - tStartCPU;
    tElapsed    = etime(clock,tStartElapsed);
    if (param.verboseLevel >= 1)
        fprintf('CPU time     = %f\n',tCPU);
        fprintf('Elapsed time = %f\n',tElapsed);

    end
    
    % Save quantities of this refinement stage
    uAll{numLevels} = u;
    TAll{numLevels} = A;
    bAll{numLevels} = b;
    TAll{numLevels} = T;
    TIAll{numLevels} = TI;
    gridAll{numLevels} = grid;
end

if (param.saveResults)
    % Save errors and error factors in latex format
    Label{1}    = 'n';
    Label{2}    = '\|e\|_{L_2}';
    Label{3}    = '{\mbox{factor}}';
    Label{4}    = '\|e\|_{L_{\infty}}';
    Label{5}    = '{\mbox{factor}}';
    Label{6}    = '\|e\|_{H_1}';
    Label{7}    = '{\mbox{factor}}';
    Label{8}    = '\|e\|_{H_1,max}';
    Label{9}    = '{\mbox{factor}}';

    fileName    = sprintf('%s/DiscError',param.outputDir);
    Caption     = sprintf('Discretization error');

    data        = [1:grid.maxLevels]';
    e           = errNorm;
    factors     = fac(e);
    fmt{1}      = '%4d';
    for i = 1:size(e,2)
        data = [data e(:,i) [0; factors(:,i)]];
        fmt{2*i} = '%.3e';
        fmt{2*i+1} = '%.3f';
    end
    latexTableFactors(data,Label,fileName,Caption,fmt{:});
end
