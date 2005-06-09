function indexF = refineIndex(grid,k,indexC)

if (k == grid.numLevels)
    error('Cannot refine index from finest level');
end

r = grid.level{k+1}.refRatio;
indexF = r.*(indexC - 1) + 1;

