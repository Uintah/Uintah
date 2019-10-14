function [A,indOut] = LshapedOut(grid,k,q,A,reallyUpdate)
%LSHAPEDOUT  Indices that are outside an L-shaped domain.
%   [A,INDOUT] = LSHPAEDOUT(GRID,K,Q,A,FLAG) returns the list INDOUT
%   of all indices on patch Q at level K that are in the quadrant
%   [0.5,1] x ... x [0.5,1] that we remove
%   from the original cubic domain to get an L-shaped domain. the LHS
%   matrix A is updated accordingly if FLAG=1, and untouched if FLAG=0.
%
%   See also: TESTDISC, ADDGRIDPATCH, SETUPPATCHINTERIOR.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

globalParams;

out(2,'--- LshapedOut(k = %d, q = %d) BEGIN ---\n',k,q);

if (nargin < 5)
    reallyUpdate = 1;
end

%=====================================================================
% Initialize; set patch "pointers" (in matlab: we actually copy P).
%=====================================================================
level                   = grid.level{k};
numPatches              = length(level.numPatches);
h                       = level.h;
P                       = grid.level{k}.patch{q};
ind                     = P.cellIndex;                              % Global 1D indices of cells

%=====================================================================
% Prepare a list of all cells
%=====================================================================

[indBox,box,matBox]     = indexBox(P,P.ilower-1,P.iupper+1);
x                       = cell(grid.dim,1);
for dim = 1:grid.dim,
    x{dim}              = (box{dim} - P.offsetSub(dim) - 0.5) * h(dim);
end
[x{:}]                  = myndgrid(x{:});

%=====================================================================
% Find all cells in the quadrant to be removed for the L-shaped problem
%=====================================================================

xmin                    = x{1};
xmax                    = x{1};
for dim = 1:grid.dim,
    xmin                = min(xmin,x{dim});
    xmax                = min(xmax,x{dim});
end
[x{:}]                  = myndgrid(x{:});

subOut                  = cell(grid.dim,1);
[subOut{:}]             = find(xmin >= 0.5);

%=====================================================================
% Convert range patch subs to indices
%=====================================================================
indOut                  = ind(mysub2ind(P.size,subOut{:}));
indOut                  = indOut(:);

%=====================================================================
% Delete connections from outside the deleted box to the
% deleted box.
%=====================================================================
[i,j,data]              = find(A(indOut,:));
in2out                  = find(~ismember(j,indOut));
out2in                  = [j(in2out) j(in2out) -data(in2out)];
if (~isempty(out2in))
    indNbhr                 = unique(out2in(:,1));
    Aold                    = spconvert([out2in; [grid.totalVars grid.totalVars 0]]);
    out2inNew               = [out2in(:,1:2) 2*out2in(:,3)];
    Anew                    = spconvert([out2inNew; [grid.totalVars grid.totalVars 0]]);
    if (reallyUpdate)
        A(indNbhr,:)        = A(indNbhr,:) - Aold(indNbhr,:) + Anew(indNbhr,:);
    end
end

out(2,'--- LshapedOut(k = %d, q = %d) END ---\n',k,q);
