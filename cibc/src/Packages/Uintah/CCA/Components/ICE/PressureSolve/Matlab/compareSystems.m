function [compare,A0,b0,x0] = compareSystems(grid,A,b,x,A1,b1,x1)
% Append and compare:
% A,b,x = MATLAB linear system
% A1,b1,x1 = C++ (HypreStandAlone) linear system

% % for offset = 0 in Hierarchy::make()
% matlabPatch = [1,1, 2,2, 3,3, 4,4];
% matlabLevel = [1,2, 1,2, 1,2, 1,2];

% for offset = level in Hierarchy::make()
matlabPatch = [1,4, 2,1, 3,2, 4,3];
matlabLevel = [1,2, 1,2, 1,2, 1,2];

compare = 1;

active = [];
for i = 1:length(matlabPatch),
    k = matlabLevel(i);
    q = matlabPatch(i);
    P = grid.level{k}.patch{q};
    ind = P.cellIndex;        % Global matlab 1D indices of cells in this patch
    [indBox,box,matBox]     = indexBox(P,P.ilower,P.iupper);                % Indices whose equations are created and added to Alist below
    active = [active; indBox];
    fprintf('i = %3d, Matlab patch = %2d, level = %d, interior indices from %d to %d\n',...
        i,q,k,min(indBox),max(indBox));
end

% Schur compliment of active matlab vars
inactive = setdiff([1:size(A,1)],active);
nInactive = length(inactive);
invA11 = spdiags(1./full(diag(A(inactive,inactive))),0,nInactive,nInactive);
A0 = A(active,active) - A(active,inactive)*invA11*A(inactive,active);
b0 = b(active) - A(active,inactive)*invA11*b(inactive);
x0 = x(active);

% Compare sizes
if (    (size(A0,1) ~= size(A1,1)) | ...
        (size(b0,1) ~= size(b1,1)) | ...
        (size(x0,1) ~= size(x1,1)))
    fprintf('Different sizes\n');
    compare = 0;
    return;
end

diffA = full(max(abs(A0(:)-A1(:))));
diffb = full(max(abs(b0(:)-b1(:))));
diffx = full(max(abs(x0(:)-x1(:))));

fprintf('A:  diff = %e\n',diffA);
fprintf('b:  diff = %e\n',diffb);
fprintf('x:  diff = %e\n',diffx);

if (diffA > 1e-5)
    fprintf('Different A\n');
    compare = 0;
    return;
end
if (diffb > 1e-5)
    fprintf('Different b\n');
    compare = 0;
    return;
end
if (diffx > 1e-5)
    fprintf('Different x\n');
    compare = 0;
    return;
end
