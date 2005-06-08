function [A,b] = setupOperator(grid)

Alist   = zeros(0,3);
b       = zeros(grid.totalVars,1);

for k = 1:grid.numLevels,
    level       = grid.level{k};
    numPatches  = length(level.numPatches);
    s           = level.stencilOffsets;
    numEntries  = size(s,1);

    for q = 1:numPatches,
        P = grid.level{k}.patch{q};
        map = P.cellIndex;
        
        %============== Construct stencil coefficients - flux-based =================

        leftSide    = 2*ones(size(P.ilower));
        rightSide   = P.size-1;
        rhsValues   = rhs(([P.ilower(1):P.iupper(1)]-0.5)*level.h(1),([P.ilower(2):P.iupper(2)]-0.5)*level.h(2));

        % Loop over interior patch cells
        for i1 = P.ilower(1):P.iupper(1)
            for i2 = P.ilower(2):P.iupper(2)
                j1 = i1 + P.offset(1);
                j2 = i2 + P.offset(2);

                % Flux vector of -Laplace operator.
                % Format: [west east north south] (west=2.0 means
                % 2.0*(uij-u_{i,j-1}), for instance)
                % rhsFactor multiplies the RHS average of the cell in the
                % discretization.

                flux = [1 1 1 1];
                rhsFactor = 1.0;

                % Change fluxes near boundaries
                if (j1 == leftSide(1))
                    flux(1) = flux(1)*2;
                    flux(3:4) = flux(3:4)*0.75;
                    rhsFactor = rhsFactor*0.75;
                end
                if (j1 == rightSide(1))
                    flux(2) = flux(2)*2;
                    flux(3:4) = flux(3:4)*0.75;
                    rhsFactor  = rhsFactor*0.75;
                end
                if (j2 == leftSide(2))
                    flux(3) = flux(3)*2;
                    flux(1:2) = flux(1:2)*0.75;
                    rhsFactor  = rhsFactor*0.75;
                end
                if (j2 == rightSide(2))
                    flux(4) = flux(4)*2;
                    flux(1:2) = flux(1:2)*0.75;
                    rhsFactor  = rhsFactor*0.75;
                end

                % Assemble fluxes into a stencil
                thisIndex = map(j1,j2);
                for i = 1:4
                    n1 = j1 + s(i+1,1);
                    n2 = j2 + s(i+1,2);
                    nbhrIndex = map(n1,n2);
                    
                    Alist = [Alist; ...
                        [thisIndex thisIndex  flux(i)]; ...
                        [thisIndex nbhrIndex -flux(i)] ...
                        ];
                end
                b(thisIndex) = rhsFactor * rhsValues(j1-1,j2-1) * prod(level.h);

            end
        end
        
        % Ghost cells: boundary conditions
        for i1 = P.ilower(1)-1:P.iupper(1)+1
            for i2 = P.ilower(2)-1:P.iupper(2)+1
                if (    (i1 >= P.ilower(1)) & (i1 <= P.iupper(1)) & ...
                        (i2 >= P.ilower(2)) & (i2 <= P.iupper(2)))
                    continue;
                end
                j1 = i1 + P.offset(1);
                j2 = i2 + P.offset(2);
                thisIndex = map(j1,j2);
                Alist = [Alist; ...
                    [thisIndex thisIndex 1.0]; ...
                    ];
                b(thisIndex) = 0.0;
            end
        end
        
    end
end

A = sparse(Alist(:,1),Alist(:,2),Alist(:,3));
