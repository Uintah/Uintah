function [A,b,indDelete] = setupInterface(grid,k,q,alpha,A,b)
%SETUPINTERFACE  Set the discrete operator at coarse-fine interface.
%   [A,B] = SETUPINTERFACE(GRID,K,Q,D,S,ALPHA,A,B) updates the LHS
%   matrix A and the RHS matrix B, adding to them all the equations at
%   coarse-fine interface on the coarse side (subtracting the original
%   coarse flux and adding instead hc/hf fine fluxes that use ghost
%   points). ALPHA is a parameter vector,
%   where each of its entries ALPHA(DIR), DIR=1,...,2*numDims is in (0,1)
%   that specifies the size of the face cells in the DIR direction.
%   ALPHA(DIR)=0.5 is a regular cell, 0.25 is used in Dirichlet B.C.
%   In the modified Panayot method we use ALPHA(DIR) smaller than 0.5
%   (maybe even smaller than 0.25) at C/F interfaces.
%
%   See also: TESTDISC, ADDGRIDPATCH, SETUPPATCHINTERIOR.

globalParams;

if (param.verboseLevel >= 1)
    fprintf('--- setupPatchInterface(k = %d, q = %d) ---\n',k,q);
end

%=====================================================================
% Initialize; set fine patch "pointers" (in matlab: we actually copy P).
%=====================================================================
level                   = grid.level{k};
numPatches              = length(level.numPatches);
h                       = level.h;
P                       = grid.level{k}.patch{q};
ind                     = P.cellIndex;                              % Global 1D indices of cells
edgeDomain              = cell(2,1);                                % Domain edges
edgeDomain{1}           = level.minCell + P.offsetSub;              % First domain cell - next to left domain boundary - patch-based sub
edgeDomain{2}           = level.maxCell + P.offsetSub;              % Last domain cell - next to right domain boundary - patch-based sub

%=====================================================================
% Set coarse patch "pointers" (in matlab: we actually copy Q).
% Find Q-indices that lie under the fine patch.
%=====================================================================
if (P.parent < 0)                                                  % Base patch at coarsest level, nothing to delete
    if (param.verboseLevel >= 2)
        fprintf('No parent patch\n');
    end
    indUnder = [];
    return;
end
Q                           = grid.level{k-1}.patch{P.parent};          % Parent patch
underLower                  = coarsenIndex(grid,k,P.ilower);
underUpper                  = coarsenIndex(grid,k,P.iupper);
under                       = cell(grid.dim,1);
[indUnder,under,matUnder]   = indexBox(Q,underLower,underUpper);

% Coarse cells for deletion are the cells under the fine patch plus
% coarse boundary variables, if the fine patch is against a domain boundary.
indEdgeAll              = [];
for dim = 1:grid.dim,                                                       % Loop over dimensions of face
    for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
        sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
        fluxNum         = 2*dim+sideNum-2;
        nearBoundary    = cell(1,grid.dim);
        [nearBoundary{:}] = find(matEdge{dim} == edgeDomain{sideNum}(dim)); % Interior cell subs near DOMAIN boundary
        near{fluxNum}   = find(matEdge{dim} == edgeDomain{sideNum}(dim));     % Interior cell indices near DOMAIN boundary (actually Dirichlet boundaries only)
        far{fluxNum}    = find(~(matEdge{dim} == edgeDomain{sideNum}(dim)));   % The rest of the cells
        cellLength{fluxNum}(nearBoundary{:}) = alpha(fluxNum)*h(dim);       % Shrink cell in this direction
    end
end




%=====================================================================
% Delete the A-connections of underlying coarse nodes (including BC
% nodes) to the rest of the system.
%=====================================================================
[A,b,temp,indFull,fullList] = setupPatchInterior(grid,k-1,P.parent,A,b,underLower,underUpper,0);
Anew                = spconvert([fullList; [grid.totalVars grid.totalVars 0]]);
A(indFull,:)        = A(indFull,:) - Anew(indFull,:);                       % Do not replace the non-zeros in A here, rather subtract from them.

%=====================================================================
% Loop over all faces.
%=====================================================================

indEdgeAll              = [];
for d = 1:grid.dim,
    for s = [-1 1],
        if (param.verboseLevel >= 2)
            fprintf('  ==> (Face d = %d, s = %+d) ---\n',d,s);
        end

        %=====================================================================
        % Prepare a list of all cell indices on the coarse face of the C/F
        % interface.
        %=====================================================================
        % We assume that the domain is large enough so that a face can abut the
        % domain boundary only on one side: if s=-1, on its left, or if s=1, on its
        % right.
        ilower                  = P.ilower;
        iupper                  = P.iupper;
        if (s == -1)
            iupper(d)           = ilower(d);
        else
            ilower(d)           = iupper(d);
        end
        boxSize                 = iupper-ilower+1;
        [indEdge,edge,matEdge]  = indexBox(P,ilower,iupper);
        x                       = cell(grid.dim,1);                         % Cell centers coordinates
        for dim = 1:grid.dim,
            x{dim}              = (edge{dim} - P.offsetSub(dim) - 0.5) * h(d);
        end
        [x{:}]                  = ndgrid(x{:});

        %=====================================================================
        % Compute new coarse cell lengths depending on alpha.
        %=====================================================================
        cellLength              = cell(2*grid.dim,1);
        near                    = cell(2*grid.dim,1);
        far                     = cell(2*grid.dim,1);
        for dim = 1:grid.dim,                                                       % Loop over dimensions of face
            for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
                sideNum                 = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
                fluxNum                 = 2*dim+sideNum-2;
                cellLength{fluxNum}     = zeros(boxSize);
                cellLength{fluxNum}(:)  = 0.5*h(dim);
            end
        end

        for dim = 1:grid.dim,                                                       % Loop over dimensions of face
            for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
                sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
                fluxNum         = 2*dim+sideNum-2;
                nearBoundary    = cell(1,grid.dim);
                [nearBoundary{:}] = find(matEdge{dim} == edgeDomain{sideNum}(dim)); % Interior cell subs near DOMAIN boundary
                near{fluxNum}   = find(matEdge{dim} == edgeDomain{sideNum}(dim));     % Interior cell indices near DOMAIN boundary (actually Dirichlet boundaries only)
                far{fluxNum}    = find(~(matEdge{dim} == edgeDomain{sideNum}(dim)));   % The rest of the cells
                cellLength{fluxNum}(nearBoundary{:}) = alpha(fluxNum)*h(dim);       % Shrink cell in this direction
            end
        end
        if (param.verboseLevel >= 3)
            fprintf('Cell Lengths = \n');
            cellLength{:}
        end

        %=====================================================================
        % Create fluxes with corrections near boundaries.
        %=====================================================================
        volume                  = ones(boxSize);
        centroid                = cell(grid.dim,1);
        flux                    = cell(2*grid.dim,1);
        for dim = 1:grid.dim,                                                       % Loop over dimensions of face
            for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
                sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
                fluxNum         = 2*dim+sideNum-2;
                flux{fluxNum}   = ones(boxSize);
            end
            volume              = volume .* (cellLength{2*dim-1} + cellLength{2*dim});
            centroid{dim}		= x{dim} + 0.5*(-cellLength{2*dim-1} + cellLength{2*dim});
        end
        if (param.verboseLevel >= 3)
            volume
        end

        for dim = 1:grid.dim,                                                       % Loop over dimensions of face
            for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
                sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
                fluxNum         = 2*dim+sideNum-2;
                faceArea        = volume ./ (cellLength{2*dim-1} + cellLength{2*dim});
                flux{fluxNum}   = flux{fluxNum} .* faceArea ./ (2*cellLength{fluxNum});
            end
        end
        if (param.verboseLevel >= 3)
            fprintf('Fluxes = \n');
            flux{:}
        end

        %=====================================================================
        % Create a list of non-zeros to be added to A, consisting of the stencil
        % coefficients of all interior cells. Fluxes are modified near domain
        % boundaries.
        %=====================================================================
        Alist                   = zeros(0,3);

        dim = d;
        side = -s;                                                                  % We look in the direction along the interface from the coarse patch into fine patch

        % Direction vector ("normal") from cell to its nbhr
        nbhrNormal      = zeros(1,grid.dim);
        nbhrNormal(dim) = side;
        indNbhr         = indexNbhr(P,indEdge,nbhrNormal);

        % Add fluxes in dimension=dim, direction=side to list of non-zeros Alist and to b
        sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
        fluxNum         = 2*dim+sideNum-2;
        indFlux         = flux{fluxNum}(:);
        thisNear = near{fluxNum};
        thisFar = far{fluxNum};
        Alist = [Alist; ...                                                 % We are never near boundaries according to the C/F interface existence rules
            [indEdge(thisFar)    indEdge(thisFar)         indFlux(thisFar)]; ...    % Count only this part of the flux; the nbhr var was already removed from the system. Note that the flux has the reverse sign because we want to delete it.
            %           [indEdge(thisFar)    indNbhr(thisFar)         -indFlux(thisFar)] ...
            ];

        %=====================================================================
        % Add the links above to the relevant equations (rows) in A.
        %=====================================================================
        Anew                = spconvert([Alist; [grid.totalVars grid.totalVars 0]]);
        A(indEdge,:)        = A(indEdge,:) + Anew(indEdge,:);                       % Do not replace the non-zeros in A here, rather add to them.

        %=====================================================================
        % Update LHS vector b.
        %=====================================================================
        %rhsValues = rhs(x);
        rhsValues = rhs(centroid);

        b(indEdge) = volume(:) .* rhsValues(:);

        indEdgeAll          = union(indEdgeAll,indEdge);
    end
end
