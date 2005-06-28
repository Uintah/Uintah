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
% 2. Create patch interior equations
%==============================================================
if (param.verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 2. Create patch interior equations\n');
    fprintf('#########################################################################\n');
end
[A,b,P.indInterior]     = setupPatchInterior(grid,k,q,A,b);

%==============================================================
% 3. Create patch edge equations
%==============================================================
if (param.verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 3. Create patch edge equations\n');
    fprintf('#########################################################################\n');
end

alpha                   = zeros(1,2*grid.dim);
for d = 1:grid.dim,
    for s = [-1 1],
        alpha(2*d-1)    = 0.25;     % Dirichlet boundary on the left in dimension d
        alpha(2*d)      = 0.25;     % Dirichlet boundary on the right in dimension d
    end
end
[A,b,P.indEdge]         = setupPatchEdge(grid,k,q,alpha,A,b);

%==============================================================
% 4. Create patch boundary equations
%==============================================================
if (param.verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 4. Create patch boundary equations\n');
    fprintf('#########################################################################\n');
end
[A,b,indBC]           = setupPatchBC(grid,k,q,alpha,A,b);

%==============================================================
% 5. Modify coarse patch edge equations
%==============================================================
if (param.verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 5. Modify coarse patch edge equations\n');
    fprintf('#########################################################################\n');
end
% Fine cells at C/F interface are the same size as any other fine cell
alphaCF                 = zeros(1,2*grid.dim);
for d = 1:grid.dim,
    for s = [-1 1],
        alphaCF(2*d-1)  = 0.5;
        alphaCF(2*d)    = 0.5;
    end
end
[A,b,indUnder] = setupInterface(grid,k,q,alphaCF,A,b);

%==============================================================
% 6. Delete underlying coarse patch equations and replace them by the identity
% operator (including ghost equations). Delete unused gridpoints.
%==============================================================
if (param.verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 6. Delete underlying coarse patch equations and replace them by the\n');
    fprintf(' identity operator (including ghost equations). Delete unused gridpoints.\n');
    fprintf('#########################################################################\n');
end
patchRange              = P.offsetInd + [1:prod(P.size)];
indUnused               = patchRange(find(abs(diag(A(patchRange,patchRange))) < eps));
indUnused               = union(indUnused,indUnder);
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
