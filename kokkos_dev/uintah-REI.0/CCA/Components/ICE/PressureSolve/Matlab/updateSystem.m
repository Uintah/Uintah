function [grid,A,b,T,TI] = updateSystem(grid,k,q,A,b,T,TI)
%UPDATESYSTEM  Add a patch to the AMR grid.
%   [GRID,A,B,T,TI] = UPDATESYSTEM(GRID,K,Q,A,B,T,TI) updates the linear
%   system left hand side matrix A and the right hand side B of the composite
%   grid linear system with a new patch Q at level K, which is already
%   updated in the grid hierarchy GRID. The transformation matrices T,TI
%   are also updated as well as GRID.
%
%   See also: ADDGRIDLEVEL, ADDPGRIDPATCH, TESTDISC, TESTADAPTIVE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

globalParams;

tStartCPU           = cputime;
tStartElapsed       = clock;

out(2,'--- updateSystem(k = %d, q = %d) BEGIN ---\n',k,q);

P                           = grid.level{k}.patch{q};           % Updated patch
indAdded                    = [length(b)+1:grid.totalVars]';
numAdded                    = length(indAdded);
% Anew                        = sparse([],[],[],grid.totalVars,grid.totalVars,(2*grid.dim+1)*grid.totalVars);
% Anew(1:size(A,1),1:size(A,2)) = A;
[i,j,data]                  = find(A);
Anew                        = spconvert([[i j data]; [grid.totalVars grid.totalVars 0]]);
A                           = Anew;
b                           = [b; zeros(numAdded,1)];

% Transformation point values -> ghost values (T)
% Tnew                                = speye(grid.totalVars);
% Tnew(1:size(T,1),1:size(T,2))       = T;
[i,j,data]                  = find(T);
Tnew                        = spconvert([[i j data]; ...
    [indAdded indAdded repmat(1.0,size(indAdded))]]);
T                                   = Tnew;

% Inverse of T
% TInew                               = speye(grid.totalVars);
% TInew(1:size(TI,1),1:size(TI,2))    = TI;
[i,j,data]                  = find(TI);
TInew                        = spconvert([[i j data]; ...
    [indAdded indAdded repmat(1.0,size(indAdded))]]);
TI                                  = TInew;

%==============================================================
% 2. Create patch interior & BC equations
%==============================================================
out(2,'#########################################################################\n');
out(2,' 2. Create patch interior & BC equations\n');
out(2,'#########################################################################\n');

[A,b,T]              = setupPatchInterior(grid,k,q,A,b,T);

%==============================================================
% 5. Modify equations near C/F interface on both coarse and fine patches;
% interpolate ghost points.
%==============================================================
out(2,'#########################################################################\n');
out(2,' 5. Modify equations near C/F interface on both coarse and fine patches;\n');
out(2,' interpolate ghost points.\n');
out(2,'#########################################################################\n');

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
out(2,'#########################################################################\n');
out(2,' 6. Delete unused gridpoints / put identity operator there.\n');
out(2,'#########################################################################\n');

patchRange              = P.offsetInd + [1:prod(P.size)];
indUnused               = patchRange(find(abs(diag(A(patchRange,patchRange))) < eps));

switch (param.problemType)

    case 'Lshaped',     % Delete upper-rigt quadrant
        [A,indOut]      = LshapedOut(grid,k,q,A,1);
        indUnused       = union(indUnused,indOut);
end

A(indUnused,:)          = 0.0;
A(:,indUnused)          = 0.0;
A(indUnused,indUnused)  = -eye(length(indUnused));
b(indUnused)            = 0.0;
T(indUnused,:)          = 0.0;
T(:,indUnused)          = 0.0;
TI(indUnused,:)         = 0.0;
TI(:,indUnused)         = 0.0;
out(2,'# unused gridpoints at this patch = %d\n',length(indUnused));
if (param.verboseLevel >= 3)
    indUnused
    A(indUnused,:)
end

% Define inverse of T (except unused points)
patchRange              = [1:grid.totalVars];           % All variables here
indZero                 = patchRange(find(abs(diag(T(patchRange,patchRange))) < eps));
T                       = deleteRows(T,indZero,indZero,1);
TI                      = inv(T);
T                       = deleteRows(T,indZero,indZero,0);
TI                      = deleteRows(TI,indZero,indZero,0);

% Save changes to grid structure
grid.level{k}.indUnused   = union(grid.level{k}.indUnused,indUnused);

tCPU        = cputime - tStartCPU;
tElapsed    = etime(clock,tStartElapsed);
out(2,'CPU time     = %f\n',tCPU);
out(2,'Elapsed time = %f\n',tElapsed);
out(2,'--- updateSystem(k = %d, q = %d) END ---\n',k,q);
