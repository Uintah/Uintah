function printGrid(grid)
% Update grid tree:
% - Update ID and base index for each patch.
% - Update index maps of each patch.

fprintf('\n--- Printing Grid Hierarchy ---\n');
for k = 1:grid.numLevels,
    fprintf('Level k=%3d   numPatches = %3d   meshsize = [%f %f]\n',k,grid.level{k}.numPatches,grid.level{k}.h);
    for q = 1:grid.level{k}.numPatches,        
        P = grid.level{k}.patch{q};                
        fprintf('  Patch q=%3d   ilower = [%3d %3d]   iupper = [%3d %3d]   baseIndex = %5d    parent = %3d',...
            q,P.ilower,P.iupper,P.baseIndex,P.parent);
        fprintf('  children = [');
        for i = 1:length(P.children)
            fprintf(' %d',P.children(i));
        end
        fprintf(' ]\n');        
    end
end
fprintf('Grid Total variables = %d\n\n',grid.totalVars);
