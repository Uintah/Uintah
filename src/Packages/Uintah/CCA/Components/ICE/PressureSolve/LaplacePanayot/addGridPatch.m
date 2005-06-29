function [grid,q,A,b] = addGridPatch(grid,k,ilower,iupper,parentQ,A,b)
%ADDGRIDPATCH  Add a patch to the AMR grid.
%   [GRID,Q,A,B] = ADDGRIDPATCH(GRID,K,ILOWER,IUPPER,PARENTQ,A,B) updates
%   the left hand side matrix A and the right hand side B of the composite
%   grid linear system with a new patch Q at level K, whose extents
%   (without ghost cells) are ILOWER to IUPPER, under the parent patch
%   PARENTQ.
%
%   See also: ADDGRIDLEVEL, TESTDISC.

globalParams;

tStartCPU           = cputime;
tStartElapsed       = clock;

if (max(ilower > iupper))
    error('Cannot create patch -- ilower > iupper');
end

%==============================================================
% 1. Create an empty patch
%==============================================================
if (param.verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 1. Create an empty patch\n');
    fprintf('#########################################################################\n');
end

grid.level{k}.numPatches    = grid.level{k}.numPatches+1;
q                           = grid.level{k}.numPatches;
P.ilower                    = ilower;
P.iupper                    = iupper;
P.size                      = P.iupper - P.ilower + 3;          % Size including ghost cells
P.parent                    = parentQ;
P.children                  = [];
P.offsetSub                 = -P.ilower+2;                      % Add to level-global cell index to get this-patch cell index. Lower left corner (a ghost cell) is (1,1) in patch indices
P.deletedBoxes              = [];
grid.level{k}.patch{q}      = P;
if (k > 1)
    grid.level{k-1}.patch{parentQ}.children = [grid.level{k-1}.patch{q}.children parentQ];
end
if (param.verboseLevel >= 1)
    fprintf('Created level k=%3d patch q=%3d (parentQ = %3d), ilower = [%3d %3d], iupper = [%3d %3d]\n',...
        k,q,parentQ,ilower,iupper);
end

grid                        = updateGrid(grid);
P                           = grid.level{k}.patch{q};           % Updated patch
Anew                        = sparse([],[],[],grid.totalVars,grid.totalVars,(2*grid.dim+1)*grid.totalVars);
Anew(1:size(A,1),1:size(A,2)) = A;
A                           = Anew;
b                           = [b; zeros(grid.totalVars-length(b),1)];

%==============================================================
% 2. Create patch interior & BC equations
%==============================================================
if (param.verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 2. Create patch interior & BC equations\n');
    fprintf('#########################################################################\n');
end

[A,b]                   = setupPatchInterior(grid,k,q,A,b);

%==============================================================
% 5. Modify equations near C/F interface on both coarse and fine patches;
% interpolate ghost points.
%==============================================================
if (param.verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 5. Modify equations near C/F interface on both coarse and fine patches;\n');
    fprintf(' interpolate ghost points.\n');
    fprintf('#########################################################################\n');
end

% To align fine cell boundaries with coarse cell boundaries, alpha has to
% be 0.5 here (otherwise near corners, coarse cells at the C/F interface
% have a weird shape).
[A,b]                   = setupInterface(grid,k,q,A,b);

%==============================================================
% 6. Delete unused gridpoints / put identity operator there.
%==============================================================
if (param.verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 6. Delete unused gridpoints / put identity operator there.\n');
    fprintf('#########################################################################\n');
end
patchRange              = P.offsetInd + [1:prod(P.size)];
indUnused               = patchRange(find(abs(diag(A(patchRange,patchRange))) < eps));
A(indUnused,indUnused)  = eye(length(indUnused));
b(indUnused)            = 0.0;
if (param.verboseLevel >= 3)
    indUnused
end

tCPU        = cputime - tStartCPU;
tElapsed    = etime(clock,tStartElapsed);
if (param.verboseLevel >= 2)
    fprintf('CPU time     = %f\n',tCPU);
    fprintf('Elapsed time = %f\n',tElapsed);
end
