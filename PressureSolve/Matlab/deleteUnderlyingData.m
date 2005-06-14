function [Alist,b,grid] = deleteUnderlyingData(grid,k,q,Alist,b)
%DELETEUNDERLYINGDATA  Delete parent data underlying the current patch.
%   [Alist,b] = deleteUnderlyingData(grid,k,q) deletes the discrete
%   equations of the parent patch of patch q at level k, and replaces them
%   with the identity operator. Note that equations at parent patch outside
%   patch-q-at-level-k at also affected. We replace the parent patch matrix
%   by the identity matrix with zero RHS to make the underlying data = 0.
%
%   See also: TESTDFM, SETOPERATOR, SETOPERATORPATCH, ADDPATCH.

level       = grid.level{k};
numPatches  = length(level.numPatches);
P           = grid.level{k}.patch{q};
map         = P.cellIndex;

if (P.parent < 0)                                                   % Base patch at coarsest level, nothing to delete
    fprintf('Nothing to delete\n');
    return;
end
fprintf('--- deleteUnderlyingData(k = %d, q = %d) ---\n',k,q);

% Find box in Q-coordinates underlying P
Q               = grid.level{k-1}.patch{P.parent};                  % Parent patch
underLower      = coarsenIndex(grid,k,P.ilower);
underUpper      = coarsenIndex(grid,k,P.iupper);
under           = cell(grid.dim,1);
% Save box in deletedBoxes list of parent
numDeletesBoxes = length(Q.deletedBoxes);
Box.child       = q;
Box.ilower      = underLower;
Box.iupper      = underUpper;
Q.deletedBoxes{numDeletesBoxes+1} = Box;
grid.level{k-1}.patch{P.parent} = Q;                  % Update parent patch
for dim = 1:grid.dim
    under{dim} = [underLower(dim):underUpper(dim)] + Q.offset(dim); % Patch-based cell indices including ghosts
end

matUnder         = cell(grid.dim,1);
[matUnder{:}]   = ndgrid(under{:});
pindexUnder     = sub2ind(P.size,matUnder{:});                      % Patch-based cell indices - list
pindexUnder     = pindexUnder(:);
mapUnder        = Q.cellIndex(pindexUnder);

% Compute chunk of patch Q (level k-1) equations of the underlying area and
% subtract them from the equations so that they disappear from level k-1 equations.
AlistUnder = setupOperatorPatch(grid,k-1,P.parent,underLower,underUpper,1,1);
AlistUnder(:,3) = -AlistUnder(:,3);
Alist = [Alist; AlistUnder];

% Place an identity operator over the deleted region and ghost cells of it
% that are at domain boundaries (B.C. are now defined at patch P instead).
AlistUnder = setupIdentityPatch(grid,k-1,P.parent,underLower,underUpper);
Alist = [Alist; AlistUnder];
b(mapUnder) = 0.0;
aaa=0;
% At this point, the LHS at the interior of the box in the parent patch
% underlying the fine patch, is the identity operator. Ghost cells of this
% box that are at domain boundaries also have LHS set to identity. Ghost
% cells of this box that are not at domain boundaries (i.e., at C/F
% interface), retain their previously defined LHS equations, except the
% fluxes in those equations directed into the box: these are removed in
% setupOperatorPatch(...,1). The RHS vector is set to 0 at the box and ALL
% its ghost cells. Thus, following this function, we should
% (1) Add the fine fluxes to the equations of the coarse C/F interface cells.
% (2) Modify the right-hand-side of these equations to the correct one.
