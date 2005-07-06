function indexF = refineIndex(grid,k,indexC)

if (k == grid.numLevels)
    error('Cannot refine index from finest level');
end

r = grid.level{k+1}.refRatio;
indexF = repmat(r,size(indexC)./size(r)).*(indexC - 1) + 1;

