function [grid,q] = addGridPatch(grid,k,ilower,iupper,parentQ)
% Add a patch to level k; patch id returned = q

if (max(ilower > iupper))
    error('Cannot create patch -- ilower > iupper');
end

grid.level{k}.numPatches = grid.level{k}.numPatches+1;
q = grid.level{k}.numPatches;

P.ilower            = ilower;
P.iupper            = iupper;
P.size              = P.iupper - P.ilower + 3;          % Size including ghost cells
P.parent            = parentQ;
P.children          = [];
P.offset            = -P.ilower+2;                      % Add to level-global cell index to get this-patch cell index. Lower left corner (a ghost cell) is (1,1) in patch indices
P.deletedBoxes      = [];

grid.level{k}.patch{q}    = P;
if (k > 1)    
    grid.level{k-1}.patch{parentQ}.children = [grid.level{k-1}.patch{q}.children parentQ];
end
grid = updateGrid(grid);

fprintf('Created level k=%3d patch q=%3d (parentQ = %3d), ilower = [%3d %3d], iupper = [%3d %3d]\n',k,q,parentQ,ilower,iupper);
