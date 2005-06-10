function [Anew,bnew] = updateSystem(grid,k,q,A,b)
%==============================================================
% Update Sparse LHS Matrix A and RHS b as a result of adding
% patch q at level k
%==============================================================
fprintf('-------------------------------------------------------------------------\n');
fprintf(' Update System (level = %d, patch = %d)\n',k,q);
fprintf('-------------------------------------------------------------------------\n');
tStartCPU        = cputime;
tStartElapsed    = clock;

level       = grid.level{k};
numPatches  = length(level.numPatches);
P           = grid.level{k}.patch{q};
map         = P.cellIndex;
Alist       = zeros(0,3);
bnew        = [b; zeros(grid.totalVars-length(b),1)];

% Create equations in the interior patch and set boundary
% conditions on appropriate ghost cells.
[AlistPatch,bPatch] = setupOperatorPatch(grid,k,q,P.ilower,P.iupper,0);
Alist = [Alist; AlistPatch];
bnew(map(:)) = bPatch;

% Delete data of parent patch under the current patch
[Alist,bnew] = deleteUnderlyingData(grid,k,q,Alist,bnew);

if (isempty(A))
    Anew = sparse(Alist(:,1),Alist(:,2),Alist(:,3));
else
    B = sparse(Alist(:,1),Alist(:,2),Alist(:,3));
    Anew = sparse(size(B,1),size(B,2));
    Anew(1:size(A,1),1:size(A,2)) = A;
    Anew = Anew + B;
end

tCPU        = cputime - tStartCPU;
tElapsed    = etime(clock,tStartElapsed);
fprintf('CPU time     = %f\n',tCPU);
fprintf('Elapsed time = %f\n',tElapsed);
