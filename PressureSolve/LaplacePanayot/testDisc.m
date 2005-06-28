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
P                       = [];
P.twoLevel              = 1;
P.setupGrid             = 1;
P.solveSystem           = 1;
P.plotResults           = 1;
P.saveResults           = 0;

P.outputDir             = 'ProblemA_1Level';
P.verboseLevel          = 2;


%=========================================================================
% Run discretization on a sequence of successively finer grids
%=========================================================================
numCellsRange           = 8; %2.^[2:1:6];
success = mkdir('.',P.outputDir);

for count = 1:length(numCellsRange)
    numCells = numCellsRange(count);
    fprintf('#### nCells = %d ####\n',numCells);
    %-------------------------------------------------------------------------
    % Set up grid (AMR levels, patches)
    %-------------------------------------------------------------------------
    if (P.setupGrid)
        if (P.verboseLevel >= 1)
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

        if (twoLevel)
            [grid,k]            = addGridLevel(grid,'refineRatio',[2 2]);
            [grid,q2,A,b]       = addGridPatch(grid,k,resolution/2 + 1,3*resolution/2,q1,A,b);              % Local patch around the domain center
        end

        tCPU        = cputime - tStartCPU;
        tElapsed    = etime(clock,tStartElapsed);
        if (P.verboseLevel >= 1)
            fprintf('CPU time     = %f\n',tCPU);
            fprintf('Elapsed time = %f\n',tElapsed);
            printGrid(grid);
        end
    end

    %-------------------------------------------------------------------------
    % Solve the linear system
    %-------------------------------------------------------------------------
    if (solveSystem)
        if (P.verboseLevel >= 1)
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
        if (P.verboseLevel >= 1)
            fprintf('CPU time     = %f\n',tCPU);
            fprintf('Elapsed time = %f\n',tElapsed);
        end
    end

    %-------------------------------------------------------------------------
    % Computed exact solution vector, patch-based
    %-------------------------------------------------------------------------
    if (plotResults)
        if (P.verboseLevel >= 1)
            fprintf('-------------------------------------------------------------------------\n');
            fprintf(' Compute exact solution, plot\n');
            fprintf('-------------------------------------------------------------------------\n');
        end
        tStartCPU        = cputime;
        tStartElapsed    = clock;

        % Plot grid
        if (grid.totalVars <= 200)
            plotGrid(grid,sprintf('%s/grid%d.eps',outputDir,numCells),1,0);
        end

        % Plot and print discretization error at all patches
        uExact = exactSolutionAMR(grid);
        tau = sparseToAMR(A*AMRToSparse(uExact,grid)-b,grid);
        fig = 0;
        for k = 1:grid.numLevels,
            level = grid.level{k};
            for q = 1:grid.level{k}.numPatches,
                P = level.patch{q};
                e = u{k}{q}-uExact{k}{q};
                e = e(:);
                fprintf('Level %2d, Patch %2d  L2_error = %e   max_error = %e   median_error = %e  max_tau = %e\n',...
                    k,q,Lpnorm(e),max(abs(e)),median(abs(e)),max(abs(tau{k}{q}(:))));
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

                fig = fig+1;
                figure(fig);
                clf;
                surf(tau{k}{q});
                title(sprintf('Truncation error on Level %d, Patch %d',k,q));
                eval(sprintf('print -depsc %s/TruncError%d_L%dP%d.eps',outputDir,numCells,k,q));
                shg;
            end
        end
        tCPU        = cputime - tStartCPU;
        tElapsed    = etime(clock,tStartElapsed);
        if (P.verboseLevel >= 1)
            fprintf('CPU time     = %f\n',tCPU);
            fprintf('Elapsed time = %f\n',tElapsed);
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
        %         for i = 1:size(e,2)
        %             data = [data e(:,i) [0; factors(:,i)]];
        %             fmt{2*i} = '%.3e';
        %             fmt{2*i+1} = '%.3f';
        %         end
        % %        latexTable(data,Label,fileName,Caption,'%3d','%.3e');
        %         latexTableFactors(data,Label,fileName,Caption,fmt{:});
    end
end
