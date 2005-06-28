function [A,b,indInterior,indFull,fullList] = setupPatchInterior(grid,k,q,A,b,ilower,iupper,reallyUpdate)
%SETUPPATCCHINTERIOR  Set the discrete operator in a patch's interior.
%   [A,B,INDINTERIOR] = SETUPPATCHINTERIOR(GRID,K,Q,A,B) updates the LHS
%   matrix A and the RHS matrix B, adding to them all the equations at
%   interior nodes (not near patch boundaries). INDINTERIOR is the list of
%   interior cell indices.
%
%   See also: ADDGRIDPATCH, TESTDISC, SETUPPATCHINTERIOR.

globalParams;

if (param.verboseLevel >= 1)
    fprintf('--- setupPatchInterior(k = %d, q = %d) ---\n',k,q);
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
% Prepare a list of all cell indices whose equations are created below.
%=====================================================================
if (nargin < 6)
    ilower                  = P.ilower+1;
end
if (nargin < 7)
    iupper                  = P.iupper-1;
end
boxSize                 = iupper-ilower+1;

if (param.verboseLevel >= 3)
    ilower
    iupper
end
[indInterior,interior]  = indexBox(P,ilower,iupper);
x                       = cell(grid.dim,1);
for dim = 1:grid.dim,
    x{dim}              = (interior{dim} - P.offsetSub(dim) - 0.5) * h(dim);
end
[x{:}]                  = ndgrid(x{:});

%=====================================================================
% Create fluxes and RHS values.
%=====================================================================
flux                    = cell(2*grid.dim,1);
for fluxNum = 1:2*grid.dim
    flux{fluxNum}       = ones(boxSize);                            % Fluxes for interior cells not near boundaries: [1 1 1 1]
end

%=====================================================================
% Create a list of non-zeros to be added to A, consisting of the stencil
% coefficients of all interior cells.
%=====================================================================
Alist                   = zeros(0,3);
fullList                = zeros(0,3);                               % Includes A^T connections of [lower,iupper] box variables
indFull                 = [];

for dim = 1:grid.dim,                                               % Loop over dimensions of patch
    for side = [-1 1]                                               % side=-1 (left) and side=+1 (right) directions in dimension d

        % Direction vector ("normal") from cell to its nbhr
        nbhrNormal      = zeros(1,grid.dim);
        nbhrNormal(dim) = side;
        indNbhr         = indexNbhr(P,indInterior,nbhrNormal);

        % Add fluxes in dimension=dim, direction=side to list of non-zeros Alist and to b
        sideNum         = (side+3)/2;                               % side=-1 ==> 1; side=1 ==> 2
        fluxNum         = 2*dim+sideNum-2;
        indFlux         = flux{fluxNum}(:);
        Alist = [Alist; ...
            [indInterior    indInterior    indFlux]; ...
            [indInterior    indNbhr        -indFlux] ...
            ];
        % Add to the full list of connections also the connections of
        % interior variables outside the box to those in the box
        % [ilower,iupper].
        out = find(~ismember(indNbhr,indInterior));                  % Add only connections from those outside the box to those in the box, so that we don't repeat connections inside the box twice in fullList
        fullList = [fullList; ...
            [indInterior    indInterior    indFlux]; ...
            [indInterior    indNbhr        -indFlux]; ...
            [indNbhr(out)        indInterior(out)    -indFlux(out)]; ...        % Transpose connection
            [indNbhr(out)        indNbhr(out)         indFlux(out)] ...        % Transpose connection
            ];
        indFull = union(indFull,indInterior);
        indFull = union(indFull,indNbhr);
    end
end

if (nargin < 8)
    reallyUpdate = 1;
end
if (reallyUpdate)
    %=====================================================================
    % Set the equations of A of these edge indices to be those specified by
    % Alist.
    %=====================================================================
    Anew                    = spconvert([Alist; [grid.totalVars grid.totalVars 0]]);
    A(indInterior,:)        = Anew(indInterior,:);

    %=====================================================================
    % Update LHS vector b.
    %=====================================================================
    rhsValues               = prod(h) * rhs(x);
    b(indInterior)          = rhsValues(:);
end
