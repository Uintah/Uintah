function [grid,A,b,T,TI] = updateSystem(grid,k,q,A,b,T,TI)
%UPDATESYSTEM  Add a patch to the AMR grid.
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

if (param.verboseLevel >= 1)
    fprintf('--- updateSystem(k = %d, q = %d) BEGIN ---\n',k,q);
end

P                           = grid.level{k}.patch{q};           % Updated patch
Anew                        = sparse([],[],[],grid.totalVars,grid.totalVars,(2*grid.dim+1)*grid.totalVars);
Anew(1:size(A,1),1:size(A,2)) = A;
A                           = Anew;
b                           = [b; zeros(grid.totalVars-length(b),1)];

% Transformation point values -> ghost values (T)
Tnew                                = speye(grid.totalVars);
Tnew(1:size(T,1),1:size(T,2))       = T;
T                                   = Tnew;

%==============================================================
% 2. Create patch interior & BC equations
%==============================================================
if (param.verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 2. Create patch interior & BC equations\n');
    fprintf('#########################################################################\n');
end

[A,b,T]              = setupPatchInterior(grid,k,q,A,b,T);

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
[A,b,T,Alist,Tlist,indDel]          = setupPatchInterface(grid,k,q,A,b,T);
if (k > 1)
    grid.level{k-1}.indUnused   = union(grid.level{k-1}.indUnused,indDel);
end

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

switch (param.problemType)
    
    case 'Lshaped',     % Delete upper-rigt quadrant
        [A,indOut]      = LshapedOut(grid,k,q,A,1);
        indUnused       = union(indUnused,indOut);
end

A(indUnused,:)          = 0.0;
A(:,indUnused)          = 0.0;
A(indUnused,indUnused)  = eye(length(indUnused));
b(indUnused)            = 0.0;
T(indUnused,:)          = 0.0;
T(:,indUnused)          = 0.0;
TI(indUnused,:)         = 0.0;
TI(:,indUnused)         = 0.0;
if (param.verboseLevel >= 2)
    fprintf('# unused gridpoints at this patch = %d\n',length(indUnused));
end
if (param.verboseLevel >= 3)
    indUnused
    A(indUnused,:)
end

% Define inverse of T (except unused points)
patchRange              = [1:grid.totalVars];           % All variables here
indZero                 = patchRange(find(abs(diag(T(patchRange,patchRange))) < eps));
T(indZero,indZero)      = eye(length(indZero));
TI                      = inv(T);
T(indZero,:)            = 0.0;
TI(indZero,:)           = 0.0;

% Save changes to grid structure
grid.level{k}.indUnused   = union(grid.level{k}.indUnused,indUnused);

tCPU        = cputime - tStartCPU;
tElapsed    = etime(clock,tStartElapsed);
if (param.verboseLevel >= 2)
    fprintf('CPU time     = %f\n',tCPU);
    fprintf('Elapsed time = %f\n',tElapsed);
end
if (param.verboseLevel >= 1)
    fprintf('--- updateSystem(k = %d, q = %d) END ---\n',k,q);
end
