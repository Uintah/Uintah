function plotResults(grid,u,uExact,tau,numCells)
%PLOTRESULTS  Plot discretization error and solution results.
%   PLOTRESULTS(GRID,U,UEXACT,TAU,NUMCELLS) plots several plots with the
%   results of our discretization scheme for scalar diffusion equations.
%   GRID is the AMR grid hierarchy, U is the discrete solution on the AMR
%   grid, UEXACT is the continuous solution inject to the same grid, TAU is
%   the truncation error of UEXACT in the discrete equations, and NUMCELLS
%   is the number of cells in the grid (usually, 4, 8, etc. where the grids
%   are 4x4, 8x8, ... e.g. in 2D).
%
%   See also: PLOTGRID, TESTDISC, TESTADATPIVE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

globalParams;

fig = 0;
for k = 1:grid.numLevels,
    level = grid.level{k};
    for q = 1:grid.level{k}.numPatches,
        P = level.patch{q};
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
        eval(sprintf('print -depsc %s/DiscSolution%d_L%dP%d.eps',param.outputDir,numCells,k,q));
        title(sprintf('Discrete solution on Level %d, Patch %d',k,q));

        %         fig = fig+1;
        %         figure(fig);
        %         clf;
        %         surf(uExact{k}{q});
        %         title(sprintf('Exact solution on Level %d, Patch %d',k,q));

        fig = fig+1;
        figure(fig);
        clf;
        surf(u{k}{q}-uExact{k}{q});
        eval(sprintf('print -depsc %s/DiscError%d_L%dP%d.eps',param.outputDir,numCells,k,q));
        title(sprintf('Discretization error on Level %d, Patch %d',k,q));
        shg;

        %         fig = fig+1;
        %         figure(fig);
        %         clf;
        %         surf(tau{k}{q});
        %         title(sprintf('Truncation error on Level %d, Patch %d',k,q));
        %         eval(sprintf('print -depsc %s/TruncError%d_L%dP%d.eps',param.outputDir,numCells,k,q));
        %         shg;
    end
end
