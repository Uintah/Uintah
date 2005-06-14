%TESTDFM  Test Distributive Flux Matching (DFM) discretization.
%   We test DFM discretization error for a simple 2D Poisson problem with a
%   known solution. We prepare an AMR grid with two levels: a global level
%   1, and a local level 2 patch around the center of the domain, where the
%   solution u has more variations. DFM is a cell-centered, finite volume,
%   symmetric discretization on the composite AMR grid.
%
%   See also: ?.
global verboseLevel

setupGrid           = 1;
solveSystem         = 1;
plotResults         = 1;
saveResults         = 1;

outputDir           = 'ProblemA_1Level';
verboseLevel        = 0;
numCellsRange       = 2.^[2:1:4];

success = mkdir('.',outputDir);

for count = 1:length(numCellsRange)
    numCells = numCellsRange(count);

    %-------------------------------------------------------------------------
    % Set up grid (AMR levels, patches)
    %-------------------------------------------------------------------------
    if (setupGrid)
        if (verboseLevel >= 1)
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

        [grid,q1]           = addGridPatch(grid,k,ones(1,grid.dim),resolution,-1);     % One global patch
        [A,b,grid]          = updateSystem(grid,k,q1,A,b);

        %--------------- Level 2: local fine grid around center of domain -----------------

        if (1)
            [grid,k]            = addGridLevel(grid,'refineRatio',[2 2]);
            [grid,q2]           = addGridPatch(grid,k,resolution/2 + 1,3*resolution/2,q1);              % Local patch around the domain center
            [A,b,grid]          = updateSystem(grid,k,q2,A,b);
        end

        tCPU        = cputime - tStartCPU;
        tElapsed    = etime(clock,tStartElapsed);
        if (verboseLevel >= 1)
            fprintf('CPU time     = %f\n',tCPU);
            fprintf('Elapsed time = %f\n',tElapsed);
            printGrid(grid);
        end
    end

    %-------------------------------------------------------------------------
    % Solve the linear system
    %-------------------------------------------------------------------------
    if (solveSystem)
        if (verboseLevel >= 1)
            fprintf('-------------------------------------------------------------------------\n');
            fprintf(' Solve the linear system\n');
            fprintf('-------------------------------------------------------------------------\n');
        end
        tStartCPU        = cputime;
        tStartElapsed    = clock;
        x = A\b;                            % Direct solver
        u = sparseToAMR(x,grid);           % Translate the solution vector to patch-based
        tCPU        = cputime - tStartCPU;
        tElapsed    = etime(clock,tStartElapsed);
        if (verboseLevel >= 1)
            fprintf('CPU time     = %f\n',tCPU);
            fprintf('Elapsed time = %f\n',tElapsed);
        end
    end

    %-------------------------------------------------------------------------
    % Computed exact solution vector, patch-based
    %-------------------------------------------------------------------------
    if (plotResults)
        if (verboseLevel >= 1)
            fprintf('-------------------------------------------------------------------------\n');
            fprintf(' Compute exact solution, plot\n');
            fprintf('-------------------------------------------------------------------------\n');
        end
        tStartCPU        = cputime;
        tStartElapsed    = clock;

        uExact = exactSolutionAMR(grid);

        % Plot and print discretization error at all patches
        fig = 0;
        for k = 1:grid.numLevels,
            level = grid.level{k};
            for q = 1:grid.level{k}.numPatches,
                P = level.patch{q};
                e = u{k}{q}-uExact{k}{q};
                e = e(:);
                fprintf('Level %2d, Patch %2d  L2_error = %e   max_error = %e   median_error = %e\n',...
                    k,q,Lpnorm(e),max(abs(e)),median(abs(e)));
                err{k}{q}(count,:) = [Lpnorm(e) max(abs(e)) median(abs(e))];

                fig = fig+1;
                figure(fig);
                clf;
                surf(u{k}{q});
                title(sprintf('Discrete solution on Level %d, Patch %d',k,q));
                eval(sprintf('print -depsc %s/DiscSolution%d_L%dP%d.eps',outputDir,numCells,k,q));

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
                eval(sprintf('print -depsc %s/DiscError%d_L%dP%d.eps',outputDir,numCells,k,q));
                shg;
            end
        end
        tCPU        = cputime - tStartCPU;
        tElapsed    = etime(clock,tStartElapsed);
        if (verboseLevel >= 1)
            fprintf('CPU time     = %f\n',tCPU);
            fprintf('Elapsed time = %f\n',tElapsed);
        end

        % Plot grid
        if (grid.totalVars <= 200)
            plotGrid(grid,sprintf('%s/grid%d.eps',outputDir,numCells),1,0);
        end
    end

end

for k = 1:grid.numLevels,
    level = grid.level{k};
    for q = 1:grid.level{k}.numPatches,
        P = level.patch{q};
        Label{1} = 'n';
        Label{2} = '\|e\|_{L_2}';
        Label{3} = '{\mbox{factor}}';
        Label{4} = '\|e\|_{L_{\infty}}';
        Label{5} = '{\mbox{factor}}';
        Label{6} = '\|e\|_{\mbox{median}}';
        Label{7} = '{\mbox{factor}}';
        fileName = sprintf('%s/DiscErrorL%dP%d',outputDir,k,q);
        Caption = sprintf('Discretization error on level %d, patch %d',k,q);
        data = [numCellsRange'];
        e = err{k}{q};
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
