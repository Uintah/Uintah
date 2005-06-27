function [A,b] = setupPatchInterior(grid,k,q,A,b)
%SETUPPATCCHINTERIOR  Set the discrete operator in a patch's interior.
%   [A,B] = SETUPPATCHINTERIOR(GRID,K,Q,A,B) updates the LHS matrix A and
%   the RHS matrix B, adding to them all the equations at interior nodes
%   (not near patch boundaries).
%
%   See also: TESTDFM, UPDATESYSTEM.
global verboseLevel

if (verboseLevel >= 1)
    fprintf('--- setupPatchInterior(k = %d, q = %d) ---\n',k,q);
end
%=====================================================================
% Initialize; set patch "pointers" (in matlab: we actually copy P)
%=====================================================================
level               = grid.level{k};
numPatches          = length(level.numPatches);
h                   = level.h;
P                   = grid.level{k}.patch{q};
ind                 = P.cellIndex;                                          % Global indices of patch cells

Alist               = zeros(0,3);
b                   = zeros(prod(boxSize),1);

%=====================================================================
% Prepare a list of all interior cells
%=====================================================================
interior            = cell(grid.dim,1);
for dim = 1:grid.dim
    interior{dim}   = [P.ilower(dim):P.iupper(dim)] + P.offsetSub(dim);     % Patch-based cell indices including ghosts
end
indInterior         = ind(interior{:});

%=====================================================================
% Construct LHS stencil coefficients from fluxes flux-based, for
% interior cells. Ignore ghost cells that are not near domain
% boundaries. They will have an interpolation stencil later. Stencils are
% stored in Alist and converted to sparse matrix format to update A.
%=====================================================================
for d = 1:grid.dim,                                                 % Loop over dimensions of patch
    for s = 1:2                                                     % s=-1 (left) and s=2 (right) directions in dimension d
        % Outward normal direction
        nbhrOffset      = zeros(1,grid.dim);
        nbhrOffset(d)   = direction;

        % Compute flux in dimension=d, direction=s
        flux = zeros(boxSize);
        flux(interior{:}) = 1;                                      % Laplace operator flux for interior cells not near boundaries: [1 1 1 1]

        % Add fluxes to list of non-zeros Alist and to b
        indNbhr         = indexNbhr(P,indInterior,d,s);
        indFlux         = flux(:);
        Alist = [Alist; ...
            [indInterior    indInterior    indFlux]; ...
            [indInterior    indNbhr        -indFlux] ...
            ];
    end
end
Ainterior           = spconvert(Alist);                             % Size = #interiors x #theirNbhrs
Aold                = A;
A                   = sparse(max(size(Aold),size(Ainterior)));
A(1:size(Aold,1),1:size(Aold,2)) = Aold;
A(indInterior,:)    = Ainterior;

%=====================================================================
% Update LHS vector b
%=====================================================================
x = cell(grid.dim,1);
for dim = 1:grid.dim,
    x{dim} = ([P.ilower(d):P.iupper(d)] - 0.5) * h(d);
end
b(indInterior) = prod(h) * rhs(x{:});
