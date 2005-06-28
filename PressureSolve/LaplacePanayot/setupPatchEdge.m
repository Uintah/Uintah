function [A,b,indEdge] = setupPatchEdge(grid,k,q,alpha,A,b)
%SETUPPATCCHEDGE  Set the discrete operator at a patch's edge.
%   [A,B,INDEDGE] = SETUPPATCHEDGE(GRID,K,Q,D,S,ALPHA,A,B) updates the LHS
%   matrix A and the RHS matrix B, adding to them all the equations at
%   edge nodes (near patch boundaries) near all faces of the patches.
%   INDEDGE is the list of interior cell indices. ALPHA is a parameter vector,
%   where each of its entries ALPHA(DIR), DIR=1,...,2*numDims is in (0,1)
%   that specifies the size of the face cells in the DIR direction.
%   ALPHA(DIR)=0.5 is a regular cell, 0.25 is used in Dirichlet B.C.
%   In the modified Panayot method we use ALPHA(DIR) smaller than 0.5
%   (maybe even smaller than 0.25) at C/F interfaces.
%
%   See also: TESTDISC, ADDGRIDPATCH, SETUPPATCHINTERIOR.

globalParams;

if (P.verboseLevel >= 1)
    fprintf('--- setupPatchEdge(k = %d, q = %d) ---\n',k,q);
end

%=====================================================================
% Initialize; set patch "pointers" (in matlab: we actually copy P).
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
% Loop over all faces.
%=====================================================================

for d = 1:grid.dim,
    for s = [-1 1],
        if (P.verboseLevel >= 2)
            fprintf('  ==> (Face d = %d, s = %+d) ---\n',d,s);
        end

        %=====================================================================
        % Prepare a list of all cell indices whose equations are created below.
        %=====================================================================
        % We assume that the domain is large enough so that a face can abut the
        % domain boundary only on one side: if s=-1, on its left, or if s=1, on its
        % right.
        nearBoundary            = 0;
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

        %=====================================================================
        % Compute cell lengths.
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
        fprintf('Cell Lengths = \n');
        cellLength{:}

        %=====================================================================
        % Create fluxes with corrections near boundaries.
        %=====================================================================
        volume                  = ones(boxSize);
        flux                    = cell(2*grid.dim,1);
        for dim = 1:grid.dim,                                                       % Loop over dimensions of face
            for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
                sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
                fluxNum         = 2*dim+sideNum-2;                                
                flux{fluxNum}   = ones(boxSize);
            end
            volume              = volume .* (cellLength{2*dim-1} + cellLength{2*dim});
        end
        volume
        
        for dim = 1:grid.dim,                                                       % Loop over dimensions of face
            for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
                sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
                fluxNum         = 2*dim+sideNum-2;                
                faceArea        = volume ./ (cellLength{2*dim-1} + cellLength{2*dim});
                flux{fluxNum}   = flux{fluxNum} .* faceArea ./ (2*cellLength{fluxNum});
            end
        end
        fprintf('Fluxes = \n');
        flux{:}
        
        %=====================================================================
        % Create a list of non-zeros to be added to A, consisting of the stencil
        % coefficients of all interior cells. Fluxes are modified near domain
        % boundaries.
        %=====================================================================
        Alist                   = zeros(0,3);

        for dim = 1:grid.dim,                                                       % Loop over dimensions of patch
            for side = [-1 1]                                                         % side=-1 (left) and side=+1 (right) directions in dimension d

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
                Alist = [Alist; ...                                                 % Far from Dirichlet boundaries, nbhrs are cell-center values
                    [indEdge(thisFar)    indEdge(thisFar)         indFlux(thisFar)]; ...
                    [indEdge(thisFar)    indNbhr(thisFar)         -indFlux(thisFar)] ...
                    ];

                Alist = [Alist; ...                                                 % near Dirichlet boundaries, nbhrs are fluxes
                    [indEdge(thisNear)   indNbhr(thisNear)        -indFlux(thisNear)] ...
                    ];
            end
        end

        %=====================================================================
        % Set the equations of A of these edge indices to be those specified by
        % Alist.
        %=====================================================================
        Anew                = spconvert([Alist; [grid.totalVars grid.totalVars 0]]);
        A(indEdge,:)    = Anew(indEdge,:);

        %=====================================================================
        % Update LHS vector b.
        %=====================================================================
        b(indEdge) = volume .* rhs(x{:});
    end
end
