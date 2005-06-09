function [A,b] = setupOperator(grid)

Alist   = zeros(0,3);
b       = zeros(grid.totalVars,1);

for k = 1:grid.numLevels,
    level       = grid.level{k};
    numPatches  = length(level.numPatches);
    s           = level.stencilOffsets;
    numEntries  = size(s,1);
    h = level.h;

    for q = 1:numPatches,
        P = grid.level{k}.patch{q};
        map = P.cellIndex;

        %============== Construct stencil coefficients - flux-based =================

        leftSide    = 2*ones(size(P.ilower));
        rightSide   = P.size-1;

        i1 = [P.ilower(1):P.iupper(1)];         % Level-based cell indices
        i2 = [P.ilower(2):P.iupper(2)];
        j1 = i1 + P.offset(1);                  % Patch-based cell indices
        j2 = i2 + P.offset(2);        
        [mat1,mat2] = ndgrid(j1,j2);        
        diagIndex = sub2ind(P.size,mat1,mat2);
        diagIndex = diagIndex(:);

        % Flux vector of -Laplace operator.
        % Format: [west east north south] (west=2.0 means
        % 2.0*(uij-u_{i,j-1}), for instance)
        % rhsFactor multiplies the RHS average of the cell in the
        % discretization.

        % Fluxes for interior cells not near boundaries: [1 1 1 1]
        flux = zeros([P.size-2 4]);
        for direction = 1:4
            flux(:,:,direction) = 1;
        end
        rhsValues = prod(h) * ...
            rhs(([P.ilower(1):P.iupper(1)]-0.5)*level.h(1),([P.ilower(2):P.iupper(2)]-0.5)*level.h(2));
        
        face = cell(2,1);
        
        % x-minus face
        [face{:}] = find(mat1 == leftSide(1));
        flux(face{:},1) = 2*flux(face{:},1);
        flux(face{:},3:4) = 0.75*flux(face{:},3:4);
        rhsValues(face{:}) = 0.75*rhsValues(face{:});
        
        % x-plus face
        [face{:}] = find(mat1 == rightSide(1));
        flux(face{:},2) = 2*flux(face{:},2);
        flux(face{:},3:4) = 0.75*flux(face{:},3:4);
        rhsValues(face{:}) = 0.75*rhsValues(face{:});
        
        % y-minus face
        [face{:}] = find(mat2 == leftSide(2));
        flux(face{:},3) = 2*flux(face{:},3);
        flux(face{:},1:2) = 0.75*flux(face{:},1:2);
        rhsValues(face{:}) = 0.75*rhsValues(face{:});
        
        % y-plus face
        [face{:}] = find(mat2 == rightSide(2));
        flux(face{:},4) = 2*flux(face{:},4);
        flux(face{:},1:2) = 0.75*flux(face{:},1:2);
        rhsValues(face{:}) = 0.75*rhsValues(face{:});
                
        thisIndex = map(j1,j2);
        thisIndex = thisIndex(:);
        for direction = 1:4
            n1 = j1 + s(direction+1,1);
            n2 = j2 + s(direction+1,2);
            nbhrIndex = map(n1,n2);
            nbhrIndex = nbhrIndex(:);
            f = flux(:,:,direction);
            f = f(:);
            
            Alist = [Alist; ...
                [thisIndex thisIndex  f]; ...
                [thisIndex nbhrIndex -f] ...
                ];
        end
        b(thisIndex) = rhsValues(:);

        % Ghost cells: boundary conditions
        i1All = [P.ilower(1)-1:P.iupper(1)+1] + P.offset(1);
        i2All = [P.ilower(2)-1:P.iupper(2)+1] + P.offset(2);
        [mat1,mat2] = ndgrid(i1All,i2All);
        diagIndexAll = sub2ind(P.size,mat1,mat2);
        diagIndexAll = diagIndexAll(:);

        thisIndex = map(setdiff(diagIndexAll,diagIndex));       % Ghost cell indices
        Alist = [Alist; ...
            [thisIndex thisIndex repmat(1.0,size(thisIndex))]; ...
            ];
        b(thisIndex) = 0.0;
    end
end

A = sparse(Alist(:,1),Alist(:,2),Alist(:,3));
