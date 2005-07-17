function [errNorm,success] = testDisc(p)
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

if (nargin < 1)
    initParam;                                                      % Initialize parameters structure
    if (param.profile)
        profile on -detail builtin;                                 % Enable profiling
    end
else
    param = p;
end

totalStartCPU           = cputime;
totalStartElapsed       = clock;

out(1,'=========================================================================\n');
out(1,' Testing discretization accuracy on increasingly finer grids\n');
out(1,'=========================================================================\n');

%=========================================================================
% Run discretization on a sequence of successively finer grids
%=========================================================================
[successFlag,message,messageid] = mkdir('.',param.outputDir);
errNorm                     = [];

for numCells = param.numCellsRange
    try;
        pack;
        out(1,'#### nCells = %d ####\n',numCells);
        param.baseResolution = numCells;

        if (param.setupGrid)
            out(2,'-------------------------------------------------------------------------\n');
            out(2,' Set up grid & system\n');
            out(2,'-------------------------------------------------------------------------\n');
            out(1,'Setting up grid\n');
            grid = [];
            [grid,A,b,T,TI]     = setupGrid;
        end

        if (param.setupGrid & param.solveSystem)
            out(2,'-------------------------------------------------------------------------\n');
            out(2,' Solve the linear system\n');
            out(2,'-------------------------------------------------------------------------\n');
            out(1,'Solving system\n');
            [u,x] = solveSystem(A,b,grid,TI);

            out(2,'-------------------------------------------------------------------------\n');
            out(2,' Compute exact solution, error norms, plot results\n');
            out(2,'-------------------------------------------------------------------------\n');
            uExact      = exactSolutionAMR(grid,T,TI);
            [y,err,tau]   = computeError(grid,A,b,T,TI,u,x,uExact);
            % Save error norms to latex tables
            if (param.saveResults)
                errNorm = saveResults(grid,A,b,T,TI,'n','%4d',numCells,y,err,tau,errNorm);
            end
            % Plot grid
            if (param.plotGrid & (grid.totalVars <= 200))
                plotGrid(grid,sprintf('%s/grid%d.eps',param.outputDir,numCells),1,0,0,0);
            end
            % Plot errors, solutions and save them to eps files
            if (param.plotResults)
                plotResults(grid,u,uExact,tau,numCells);
            end
        end
        success = 1;
    catch;
        out(1,'Failed in numCells = %d: %s\n',numCells,lasterr);
        success = 0;
    end
end

totalCPU        = cputime - totalStartCPU;
totalElapsed    = etime(clock,totalStartElapsed);
out(1,'CPU time     = %f\n',totalCPU);
out(1,'Elapsed time = %f\n',totalElapsed);
if (nargin < 1)
    if (param.profile)
        profile report;                             % Generate timing profile report
    end
end
