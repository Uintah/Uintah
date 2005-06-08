function u = exactSolutionAMR(grid)
% Exact solution in patch-based AMR format.

u = cell(grid.numLevels,1);

for k = 1:grid.numLevels,
    u{k} = cell(grid.level{k}.numPatches,1);
    h = grid.level{k}.h;

    for q = 1:grid.level{k}.numPatches,
        P = grid.level{k}.patch{q};
        
        i1 = [P.ilower(1)-1:P.iupper(1)+1];
        i2 = [P.ilower(2)-1:P.iupper(2)+1];                
        [mat1,mat2] = ndgrid(i1,i2);
        u{k}{q} = exactSolution((mat1-0.5)*h(1),(mat2-0.5)*h(2));
        
        % Set boundary conditions
        for i1 = P.ilower(1)-1:P.iupper(1)+1
            for i2 = P.ilower(2)-1:P.iupper(2)+1
                if (    (i1 >= P.ilower(1)) & (i1 <= P.iupper(1)) & ...
                        (i2 >= P.ilower(2)) & (i2 <= P.iupper(2)))
                    continue;
                end
                j1 = i1 + P.offset(1);
                j2 = i2 + P.offset(2);                
                u{k}{q}(j1,j2) = 0.0;
            end
        end
        
    end

end
