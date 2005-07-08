function plotResults(grid,u,uExact,tau,numCells)
globalParams;
fig = 0;
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
