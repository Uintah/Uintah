function [grid,q] = addGridPatch(grid,k,ilower,iupper,parentQ)
%ADDGRIDPATCH  Add a patch to the AMR grid.
%   [GRID,Q] = ADDGRIDPATCH(GRID,K,ILOWER,IUPPER,PARENTQ) updates the grid
%   structure with a new patch Q at level K, whose extents
%   (without ghost cells) are ILOWER to IUPPER, under the parent patch
%   PARENTQ.
%
%   See also: ADDGRIDLEVEL, TESTDISC, UPDATESYSTEM.

globalParams;

tStartCPU           = cputime;
tStartElapsed       = clock;

out(2,'--- addGridPatch(k = %d) BEGIN ---\n',k);

if (max(ilower > iupper))
    error('Cannot create patch: ilower > iupper');
end

%==============================================================
% 1. Create an empty patch
%==============================================================

grid.level{k}.numPatches    = grid.level{k}.numPatches+1;
q                           = grid.level{k}.numPatches;
P.ilower                    = ilower;
P.iupper                    = iupper;
P.size                      = P.iupper - P.ilower + 3;          % Size including ghost cells
P.parent                    = parentQ;
P.children                  = [];
P.offsetSub                 = -P.ilower+2;                      % Add to level-global cell index to get this-patch cell index. Lower left corner (a ghost cell) is (1,1) in patch indices
P.nbhrPatch                 = -ones(grid.dim,2);
P.deletedBoxes              = [];
grid.level{k}.patch{q}      = P;
if (k > 1)
    grid.level{k-1}.patch{parentQ}.children = [grid.level{k-1}.patch{parentQ}.children q];
end
out(2,'Created level k=%3d patch q=%3d (parentQ = %3d), ilower = [%3d %3d], iupper = [%3d %3d]\n',...
    k,q,parentQ,ilower,iupper);

%==============================================================
% 2. Update grid hierarchy
%==============================================================

grid                        = updateGrid(grid);

tCPU        = cputime - tStartCPU;
tElapsed    = etime(clock,tStartElapsed);
out(2,'CPU time     = %f\n',tCPU);
out(2,'Elapsed time = %f\n',tElapsed);
out(2,'--- addGridPatch(k = %d, q = %d) END ---\n',k,q);
