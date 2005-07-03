function result = normAMR(grid,e,type)
% Volume-scaled L2,H1,Linfinity norm of an AMR function e.
globalParams;

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
        % Create fluxes for each cell (with corrections near boundaries).
        %=====================================================================
        flux                    = cell(2*grid.dim,1);
        for dim = 1:grid.dim,                                                       % Loop over dimensions of face
            for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
                sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
                fluxNum         = 2*dim+sideNum-2;
                faceArea        = volume ./ h(dim);
                diffusion{fluxNum}   = 1 ./ diffLength{fluxNum};                  % Flux_coef / face area = avg_diffusion_coef / (FD length)
            end
        end
        if (param.verboseLevel >= 3)
            fprintf('Fluxes = \n');
            flux{:}
        end

        %=====================================================================
        % Compute norm
        %=====================================================================
        uinterior = u(box{:});
        switch (type)
            case 'L1',
                result = result + sum(volume(:).*abs(uinterior(:)));
            case 'L2',
                %u(box{:})
                %median(abs(uinterior(:)))
                %max(abs(uinterior(:)))
                result = result + sum(volume(:).*abs(uinterior(:)).^2);
            case 'max',
                result = max(result,max(abs(uinterior(:))));
            case {'H1','H1max'},
                for dim = 1:grid.dim,                                                       % Loop over dimensions of patch
                    for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
                        % Direction vector ("normal") from cell to its nbhr
                        nbhrNormal      = zeros(1,grid.dim);
                        nbhrNormal(dim) = side;
                        indNbhr         = indexNbhr(P,indBox,nbhrNormal);
                        nbhr               = cell(grid.dim,1);
                        for d = 1:grid.dim
                            nbhr{d}      = [P.ilower(d):P.iupper(d)] + nbhrNormal(d) + P.offsetSub(d);     % Patch-based cell indices including ghosts
                        end
                        unbhr           = u(nbhr{:});

                        % Add fluxes in dimension=dim, direction=side to list of non-zeros Alist and to b
                        sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
                        fluxNum         = 2*dim+sideNum-2;
                        i         = flux{fluxNum}(:);
                        thisNear        = near{fluxNum};
                        thisFar         = far{fluxNum};

                        % Contribution of flux to interior equation at indBox
                        switch (type)
                            case 'H1',
                                result = result + ...
                                    0.5*sum(volume(:).*abs(diffusion{fluxNum}(:).*(uinterior(:) - unbhr(:))).^2);
                            case 'H1max',
                                result = max(result,...
                                0.5*max(abs(diffusion{fluxNum}(:).*(uinterior(:) - unbhr(:)))));
                        end

                    end
                end
        end
    end
end

% Scale result
switch (type)
    case {'L1','max','H1max'},
    case {'L2','H1'},
        result = sqrt(result);
    case 'max',
end
