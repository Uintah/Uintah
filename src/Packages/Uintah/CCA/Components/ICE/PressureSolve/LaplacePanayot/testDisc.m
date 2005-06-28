%TESTDISC  Test pressure equation discretization.
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
param                      = [];
param.twoLevel              = 1;
param.setupGrid             = 1;
param.solveSystem           = 1;
param.plotResults           = 1;
param.saveResults           = 0;

param.outputDir             = 'ProblemA_1Level';
param.verboseLevel          = 2;


%=========================================================================
% Run discretization on a sequence of successively finer grids
%=========================================================================
numCellsRange           = 4; %2.^[2:1:7];
success = mkdir('.',param.outputDir);

for count = 1:length(numCellsRange)
    numCells = numCellsRange(count);
    fprintf('#### nCells = %d ####\n',numCells);
    %-------------------------------------------------------------------------
    % Set up grid (AMR levels, patches)
    %-------------------------------------------------------------------------
    if (param.setupGrid)
        if (param.verboseLevel >= 1)
            fprintf('-------------------------------------------------------------------------\n');
            fprintf(' Set up grid & system\n');
            fprintf('-------------------------------------------------------------------------\n');
        end
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

        resolution          = [numCells numCells];
        [grid,k]            = addGridLevel(grid,'meshsize',grid.domainSize./resolution);
        [grid,q1,A,b]       = addGridPatch(grid,k,ones(1,grid.dim),resolution,-1,A,b);     % One global patch

        %--------------- Level 2: local fine grid around center of domain -----------------

        if (param.twoLevel)
            [grid,k]            = addGridLevel(grid,'refineRatio',[2 2]);
            [grid,q2,A,b]       = addGridPatch(grid,k,resolution/2 + 1,3*resolution/2,q1,A,b);              % Local patch around the domain center
        end

        tCPU        = cputime - tStartCPU;
        tElapsed    = etime(clock,tStartElapsed);
        if (param.verboseLevel >= 1)
            fprintf('CPU time     = %f\n',tCPU);
            fprintf('Elapsed time = %f\n',tElapsed);
            printGrid(grid);
        end
    end

    %-------------------------------------------------------------------------
    % Solve the linear system
    %-----------------------tau--------------------------------------------------
    if (param.solveSystem)
        if (param.verboseLevel >= 1)
            fprintf('-------------------------------------------------------------------------\n');
            fprintf(' Solve the linear system\n');
            fprintf('-------------------------------------------------------------------------\n');
        end
        tStartCPU        = cputime;
        tStartElapsed    = clock;
        x = A\b;                            % Direct solver
        u = sparseToAMR(x,grid,1);           % Translate the solution vector to patch-based
        tCPU        = cputime - tStartCPU;
        tElapsed    = etime(clock,tStartElapsed);
        if (param.verboseLevel >= 1)
            fprintf('CPU time     = %f\n',tCPU);
            fprintf('Elapsed time = %f\n',tElapsed);
        end
    end

    %-------------------------------------------------------------------------
    % Computed exact solution vector, patch-based
    %-------------------------------------------------------------------------
    if (param.plotResults)
        if (param.verboseLevel >= 1)
            fprintf('-------------------------------------------------------------------------\n');
            fprintf(' Compute exact solution, plot\n');
            fprintf('-------------------------------------------------------------------------\n');
        end
        tStartCPU        = cputime;
        tStartElapsed    = clock;

        % Plot grid
        if (grid.totalVars <= 200)
            plotGrid(grid,sprintf('%s/grid%d.eps',param.outputDir,numCells),1,0);
        end

        % Plot and print discretization error at all patches
        uExact = exactSolutionAMR(grid);
        tau = sparseToAMR(b-A*AMRToSparse(uExact,grid),grid,0);
        f = sparseToAMR(b,grid,0);
        fig = 0;
        for k = 1:grid.numLevels,
            level = grid.level{k};
            for q = 1:grid.level{k}.numPatches,
                P = level.patch{q};
                e = uExact{k}{q}-u{k}{q};
                e = e(:);
                t = tau{k}{q}(:);
                fprintf('Level %2d, Patch %2d  err (L2=%.3e  max=%.3e  med=%.3e)  tau (L2=%.3e  max=%.3e  med=%.3e)\n',...
                    k,q,...
                    Lpnorm(e),max(abs(e)),median(abs(e)),...
                    Lpnorm(t),max(abs(t)),median(abs(t)));
                err{k}{q}(count,:) = [Lpnorm(e) max(abs(e)) median(abs(e))];
                trunc{k}{q}(count,:) = [Lpnorm(t) max(abs(t)) median(abs(t))];

                fig = fig+1;
                figure(fig);
                clf;
                surf(f{k}{q});
                title(sprintf('Discrete RHS on Level %d, Patch %d',k,q));
                eval(sprintf('print -depsc %s/DiscRHS%d_L%dP%d.eps',param.outputDir,numCells,k,q));

                fig = fig+1;
                figure(fig);
                clf;
                surf(u{k}{q});
                title(sprintf('Discrete solution on Level %d, Patch %d',k,q));
                eval(sprintf('print -depsc %s/DiscSolution%d_L%dP%d.eps',param.outputDir,numCells,k,q));

                fig = fig+1;
                figure(fig);
                clf;
                surf(uExact{k}{q});
                title(sprintf('Exact solution on Level %d, Patch %d',k,q));

                fig = fig+1;
                figure(fig);
                clf;
                surf(u{k}{q}-uExact{k}{q});
                title(sprintf('Discretization error on Level %d, Patch %d',k,q));
                eval(sprintf('print -depsc %s/DiscError%d_L%dP%d.eps',param.outputDir,numCells,k,q));
                shg;

                fig = fig+1;
                figure(fig);
                clf;
                surf(tau{k}{q});
                title(sprintf('Truncation error on Level %d, Patch %d',k,q));
                eval(sprintf('print -depsc %s/TruncError%d_L%dP%d.eps',param.outputDir,numCells,k,q));
                shg;
            end
        end
        tCPU        = cputime - tStartCPU;
        tElapsed    = etime(clock,tStartElapsed);
        if (param.verboseLevel >= 1)
            fprintf('CPU time     = %f\n',tCPU);
            fprintf('Elapsed time = %f\n',tElapsed);
        end
    end

end

for k = 1:grid.numLevels,
    level = grid.level{k};
    for q = 1:grid.level{k}.numPatches,
        P = level.patch{q};
        
        % Save errors and error factors in latex format
        Label{1} = 'n';
        Label{2} = '\|e\|_{L_2}';
        Label{3} = '{\mbox{factor}}';
        Label{4} = '\|e\|_{L_{\infty}}';
        Label{5} = '{\mbox{factor}}';
        Label{6} = '\|e\|_{\mbox{median}}';
        Label{7} = '{\mbox{factor}}';
        fileName = sprintf('%s/DiscErrorL%dP%d',param.outputDir,k,q);
        Caption = sprintf('Discretization error on level %d, patch %d',k,q);
        data    = [numCellsRange'];
        e       = err{k}{q};
        factors = fac(e);
        fmt{1} = '%4d';
        for i = 1:size(e,2)
            data = [data e(:,i) [0; factors(:,i)]];
            fmt{2*i} = '%.3e';
            fmt{2*i+1} = '%.3f';
        end
        %        latexTable(data,Label,fileName,Caption,'%3d','%.3e');
        latexTableFactors(data,Label,fileName,Caption,fmt{:});

        % Save truncation errors and truncation error factors in latex format
        Label{1} = 'n';
        Label{2} = '\|\tau\|_{L_2}';
        Label{3} = '{\mbox{factor}}';
        Label{4} = '\|\tau\|_{L_{\infty}}';
        Label{5} = '{\mbox{factor}}';
        Label{6} = '\|\tau\|_{\mbox{median}}';
        Label{7} = '{\mbox{factor}}';
        fileName = sprintf('%s/TruncErrorL%dP%d',param.outputDir,k,q);
        Caption = sprintf('Truncation error on level %d, patch %d',k,q);
        data    = [numCellsRange'];
        e       = trunc{k}{q};
        factors = fac(e);
        fmt{1} = '%4d';
        for i = 1:size(e,2)
            data = [data e(:,i) [0; factors(:,i)]];
            fmt{2*i} = '%.3e';
            fmt{2*i+1} = '%.3f';
        end
        %        latexTable(data,Label,fileName,Caption,'%3d','%.3e');
        latexTableFactors(data,Label,fileName,Caption,fmt{:});
    end
end
