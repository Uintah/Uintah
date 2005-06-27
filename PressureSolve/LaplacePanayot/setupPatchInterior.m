function [A,b,indInterior] = setupPatchInterior(grid,k,q,A,b)
%SETUPPATCCHINTERIOR  Set the discrete operator in a patch's interior.
%   [A,B,INDINTERIOR] = SETUPPATCHINTERIOR(GRID,K,Q,A,B) updates the LHS
%   matrix A and the RHS matrix B, adding to them all the equations at 
%   interior nodes (not near patch boundaries). INDINTERIOR is the list of
%   interior cell indices.
%
%   See also: TESTDISC, ADDGRIDPATCH.

global verboseLevel

if (verboseLevel >= 1)
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
[indInterior,interior]  = indexBox(P,P.ilower+1,P.iupper-1);

%=====================================================================
% Create a list of non-zeros to be added to A, consisting of the stencil
% coefficients of all interior cells.
%=====================================================================
Alist                   = zeros(0,3);
for d = 1:grid.dim,                                                 % Loop over dimensions of patch
    for s = -1:1                                                    % s=-1 (left) and s=+1 (right) directions in dimension d
        % Direction vector ("normal") from cell to its nbhr
        nbhrNormal      = zeros(1,grid.dim);
        nbhrNormal(d)   = s;

        % Compute flux in dimension=d, direction=s
        flux            = zeros(P.size);
        flux(interior{:}) = 1;                                      % Laplace flux for interior cells = u_{cell} - u_{nbhr}
        indFlux         = flux(interior{:});
        indFlux         = indFlux(:);

        % Add fluxes to list of non-zeros Alist and to b
        indNbhr         = indexNbhr(P,indInterior,nbhrNormal);
        Alist = [Alist; ...
            [indInterior    indInterior    indFlux]; ...
            [indInterior    indNbhr        -indFlux] ...
            ];
    end
end

%=====================================================================
% Merge old matrix with new list of non-zeros into LHS matrix A.
%=====================================================================
[i,j]               = find(A);
data                = A(find(A));
Alist               = [[i j data]; Alist];
A                   = spconvert(Alist);

%=====================================================================
% Update LHS vector b.
%=====================================================================
x = cell(grid.dim,1);
for dim = 1:grid.dim,
    x{dim} = (interior{dim} - P.offsetSub(dim) - 0.5) * h(d);
end
b(indInterior) = prod(h) * rhs(x{:});
