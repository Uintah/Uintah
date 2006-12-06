function printGrid(grid)
% Update grid tree:
% - Update ID and base index for each patch.
% - Update index maps of each patch.

globalParams;

fprintf('\n--- Printing Grid Hierarchy ---\n');
for k = 1:grid.numLevels,
    fprintf('Level k=%3d   numPatches = %3d   meshsize = [',...
        k,grid.level{k}.numPatches);
    fprintf(' %f',grid.level{k}.h);
    fprintf(' ]\n');
    for q = 1:grid.level{k}.numPatches,        
        P = grid.level{k}.patch{q};                
        fprintf('  Patch q=%3d   ilower = [',q);
        fprintf(' %3d',P.ilower);
        fprintf(' ]   iupper = [');
        fprintf(' %3d',P.iupper);
        fprintf(' ]\n');
        fprintf('      offsetInd = %5d    parent = %3d\n',...
            P.offsetInd,P.parent);
        fprintf('      Children  = [');
        for i = 1:length(P.children)
            fprintf(' %d',P.children(i));
        end
        fprintf(' ]\n');
        fprintf('      nbhrPatch (left ) = [');
        for i = 1:grid.dim
            fprintf(' %d',P.nbhrPatch(i,1));
        end
        fprintf(' ]\n');
        fprintf('      nbhrPatch (right) = [');
        for i = 1:grid.dim
            fprintf(' %d',P.nbhrPatch(i,2));
        end
        fprintf(' ]\n');        
    end
    fprintf('\n');
end
fprintf('Number of levels     = %d\n',grid.numLevels);
fprintf('Grid Total variables = %d\n\n',grid.totalVars);
