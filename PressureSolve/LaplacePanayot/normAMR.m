function result = normAMR(grid,e,type)
% Volume-scaled L2 norm of an AMR function e.

result = 0;
for k = 1:grid.numLevels,
    level       = grid.level{k};
    h           = level.h;
    numPatches              = length(level.numPatches);

    for q = 1:numPatches,
        P                       = grid.level{k}.patch{q};
        ind                     = P.cellIndex;                              % Global 1D indices of cells
        edgeDomain              = cell(2,1);                                % Domain edges
        edgeDomain{1}           = level.minCell + P.offsetSub;              % First domain cell - next to left domain boundary - patch-based sub
        edgeDomain{2}           = level.maxCell + P.offsetSub;              % Last domain cell - next to right domain boundary - patch-based sub
        u                       = e{k}{q};

        %=====================================================================
        % Prepare a list of all interior cells
        %=====================================================================
        [indBox,box,matBox]     = indexBox(P,P.ilower,P.iupper);                % Indices whose equations are created and added to Alist below
        boxSize                 = P.iupper-P.ilower+1;
        x                       = cell(grid.dim,1);
        for dim = 1:grid.dim,
            x{dim}              = (box{dim} - P.offsetSub(dim) - 0.5) * h(dim);
        end
        [x{:}]                  = ndgrid(x{:});
        %=====================================================================
        % Compute cell lengths and volumes.
        %=====================================================================
        % Cell extent is
        % [x{1}-0.5*h(1),x{1}+0.5*h(1)] x ... x [x{dim}-0.5*h(dim),x{dim}+0.5*h(dim)].
        % Cell volume is thus prod(h).
        diffLength              = cell(2*grid.dim,1);                               % Distance along which we approximate the flux by finite difference
        volume                  = prod(h).*ones(boxSize);                           % Cell volume
        centroid                = x;                                                % Cell centroid
        near                    = cell(2*grid.dim,1);                               % Cells near domain boundaries in the fluxNum direction
        far                     = cell(2*grid.dim,1);                               % Cells far from domain boundaries in the fluxNum direction
        for dim = 1:grid.dim,                                                       % Loop over dimensions of face
            for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
                sideNum                 = (side+3)/2;                               % side=-1 ==> 1; side=1 ==> 2
                fluxNum                 = 2*dim+sideNum-2;
                diffLength{fluxNum}     = h(dim)*ones(boxSize);                     % Standard FD is over distance h
                % Adjust distances for FD near domain boundaries
                nearBoundary            = cell(1,grid.dim);
                [nearBoundary{:}] = find(matBox{dim} == edgeDomain{sideNum}(dim));  % Interior cell subs near DOMAIN boundary
                near{fluxNum}   = find(matBox{dim} == edgeDomain{sideNum}(dim));    % Interior cell indices near DOMAIN boundary (actually Dirichlet boundaries only)
                far{fluxNum}    = find(~(matBox{dim} == edgeDomain{sideNum}(dim))); % The rest of the cells
                diffLength{fluxNum}(nearBoundary{:}) = 0.5*h(dim);                  % Twice smaller distance in this direction
            end
        end

        %=====================================================================
        % Compute norm
        %=====================================================================
        uinterior = u(box{:});
        switch (type)
            case 'L1',
                result = result + sum(volume(:).*abs(uinterior(:)));
            case 'L2',
                result = result + sum(volume(:).*abs(uinterior(:)).^2);
            case 'max',
                result = max(result,max(abs(u(indBox))));
            case 'H1',
                for dim = 1:grid.dim,                                                       % Loop over dimensions of patch
                    for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
                        % Direction vector ("normal") from cell to its nbhr
                        nbhrNormal      = zeros(1,grid.dim);
                        nbhrNormal(dim) = side;
                        indNbhr         = indexNbhr(P,indBox,nbhrNormal);

                        % Add fluxes in dimension=dim, direction=side to list of non-zeros Alist and to b
                        sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
                        fluxNum         = 2*dim+sideNum-2;
                        indFlux         = flux{fluxNum}(:);
                        thisNear        = near{fluxNum};
                        thisFar         = far{fluxNum};

                        % Contribution of flux to interior equation at indBox
                        result = result + ...
                            sum(volume(:).*abs(indFlux(indBox).*(u(indBox) - u(indNbhr)).^2));
                    end
                end
        end
    end
end

% Scale result
switch (type)
    case 'L1',
    case 'L2',
        result = sqrt(result);
    case 'max',
    case 'H1',
end
