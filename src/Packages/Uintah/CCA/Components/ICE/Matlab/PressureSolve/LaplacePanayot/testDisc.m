%function [grid,A,b,T,TI,u] = testDisc(twoLevel)
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
param                       = [];

param.problemType           = 'Lshaped'; %'ProblemB'; %'quadratic'; %'Lshaped'; %
param.outputDir             = 'ProblemA_1Level';

param.twoLevel              = 0;
param.threeLevel            = 0;
param.setupGrid             = 1;
param.solveSystem           = 1;
param.plotResults           = 0;
param.saveResults           = 0;
param.verboseLevel          = 0;

%=========================================================================
% Run discretization on a sequence of successively finer grids
%=========================================================================
numCellsRange               = 2.^[2:1:6];
success                     = mkdir('.',param.outputDir);
errNorm                     = zeros(length(numCellsRange),4);

for count = 1:length(numCellsRange)
    pack;
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

            if (0)
                % Cover the entire domain
                [grid,q2]  = addGridPatch(grid,k,ones(1,grid.dim),2*resolution,q1);              % Local patch around the domain center
            end

            if (1)
                % Cover central half of the domain
                [grid,q2]  = addGridPatch(grid,k,resolution/2 + 1,3*resolution/2,q1);              % Local patch around the domain center
            end

            if (0)
                % Cover central quarter of the domain
                [grid,q2]  = addGridPatch(grid,k,3*resolution/4 + 1,5*resolution/4,q1);              % Local patch around the domain center
            end

            if (0)
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
            end

            if (0)
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
            end

            for q = 1:grid.level{k}.numPatches,
                [grid,A,b,T,TI]      = updateSystem(grid,k,q,A,b,T,TI);
            end
        end

        %--------------- Level 3: yet local fine grid around center of domain -----------------
        if ((param.twoLevel) & (param.threeLevel))
            [grid,k]   = addGridLevel(grid,'refineRatio',[2 2]);
            
            if (0)
                % Cover central half of the domain
                [grid,q3]  = addGridPatch(grid,k,3*resolution/2 + 1,5*resolution/2,q2);              % Local patch around the domain center
            end
            
            if (1)
                % Cover central half of the central quarter of the domain
                [grid,q3]  = addGridPatch(grid,k,7*resolution/4 + 1,9*resolution/4,q2);              % Local patch around the domain center
%                [grid,q3]  = addGridPatch(grid,k,15*resolution/4 + 1,17*resolution/4,q2);              % Local patch around the domain center
            end
            
            for q = 1:grid.level{k}.numPatches,
                [grid,A,b,T,TI]      = updateSystem(grid,k,q,A,b,T,TI);
            end
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
    %-------------------------------------------------------------------------
    if (param.solveSystem)
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
    end

    %-------------------------------------------------------------------------
    % Computed exact solution vector, patch-based
    %-------------------------------------------------------------------------
    if (param.verboseLevel >= 1)
        fprintf('-------------------------------------------------------------------------\n');
        fprintf(' Compute exact solution, plot\n');
        fprintf('-------------------------------------------------------------------------\n');
    end
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
    errNorm(count,:) = [ ...
        normAMR(grid,err,'L2') ...
        normAMR(grid,err,'max') ...
        normAMR(grid,err,'H1') ...
        normAMR(grid,err,'H1max') ...
        ];
    fprintf('#vars = %5d  L2=%.3e  max=%.3e  H1=%.3e  H1max=%.3e\n',grid.totalVars,errNorm(count,:));

    if (param.plotResults)
        for k = 1:grid.numLevels,
            level = grid.level{k};
            for q = 1:grid.level{k}.numPatches,
                P = level.patch{q};
                %                e = uExact{k}{q}-u{k}{q};
                %                e = e(:);
                %                t = tau{k}{q}(:);
                %                 fprintf('Level %2d, Patch %2d  err (L2=%.3e  max=%.3e  med=%.3e)  tau (L2=%.3e  max=%.3e  med=%.3e)\n',...
                %                     k,q,...
                %                     Lpnorm(e),max(abs(e)),median(abs(e)),...
                %                     Lpnorm(t),max(abs(t)),median(abs(t)));
                %                err{k}{q}(count,:) = [Lpnorm(e) max(abs(e)) median(abs(e))];
                %                trunc{k}{q}(count,:) = [Lpnorm(t) max(abs(t)) median(abs(t))];

                %                 fig = fig+1;
                %                 figure(fig);
                %                 clf;
                %                 surf(f{k}{q});
                %                 title(sprintf('Discrete RHS on Level %d, Patch %d',k,q));
                %                 eval(sprintf('print -depsc %s/DiscRHS%d_L%dP%d.eps',param.outputDir,numCells,k,q));

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

if (param.saveResults)
    % Save errors and error factors in latex format
    Label{1}    = 'n';
    Label{2}    = '\|e\|_{L_2}';
    Label{3}    = '{\mbox{factor}}';
    Label{4}    = '\|e\|_{L_{\infty}}';
    Label{5}    = '{\mbox{factor}}';
    Label{6}    = '\|e\|_{H_1}';
    Label{7}    = '{\mbox{factor}}';

    fileName    = sprintf('%s/DiscError',param.outputDir);
    Caption     = sprintf('Discretization error');

    data        = [numCellsRange'];
    e           = errNorm;
    factors     = fac(e);
    fmt{1}      = '%4d';
    for i = 1:size(e,2)
        data = [data e(:,i) [0; factors(:,i)]];
        fmt{2*i} = '%.3e';
        fmt{2*i+1} = '%.3f';
    end
    %        latexTable(data,Label,fileName,Caption,'%3d','%.3e');
    latexTableFactors(data,Label,fileName,Caption,fmt{:});

    %         % Save truncation errors and truncation error factors in latex format
    %         Label{1} = 'n';
    %         Label{2} = '\|\tau\|_{L_2}';
    %         Label{3} = '{\mbox{factor}}';
    %         Label{4} = '\|\tau\|_{L_{\infty}}';
    %         Label{5} = '{\mbox{factor}}';
    %         Label{6} = '\|\tau\|_{\mbox{median}}';
    %         Label{7} = '{\mbox{factor}}';
    %         fileName = sprintf('%s/TruncErrorL%dP%d',param.outputDir,k,q);
    %         Caption = sprintf('Truncation error on level %d, patch %d',k,q);
    %         data    = [numCellsRange'];
    %         e       = trunc{k}{q};
    %         factors = fac(e);
    %         fmt{1} = '%4d';
    %         for i = 1:size(e,2)
    %             data = [data e(:,i) [0; factors(:,i)]];
    %             fmt{2*i} = '%.3e';
    %             fmt{2*i+1} = '%.3f';
    %         end
    %         %        latexTable(data,Label,fileName,Caption,'%3d','%.3e');
    %         latexTableFactors(data,Label,fileName,Caption,fmt{:});
end
