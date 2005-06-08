function A = stencilToSparse(A,g,ilower,iupper,s,values)

numDims = length(ilower);
gridSize = g.iend - g.istart + 1;

i = cell(numDims,1);
diag = cell(numDims,1);
for d = 1:numDims,
    diag{d} = [ilower(d):iupper(d)] - g.istart(d) + 1;
    i{d} = diag{d} + s(d);
end
 cells = cell(numDims,1);
[cells{:}] = ndgrid(i{:});
 cellsDiag = cell(numDims,1);
[cellsDiag{:}] = ndgrid(diag{:});

numCells = length(cells{1}(:));
cellsMat = zeros(numCells,numDims);
cellsMatDiag = zeros(numCells,numDims);
for d = 1:numDims,
    cellsMat(:,d) = cells{d}(:);
    cellsMatDiag(:,d) = cellsDiag{d}(:);
end

for d = 1:numDims,
    %d
    outside = find(max((cellsMat < repmat(ones(size(gridSize)),[size(cellsMat,1) 1])) | ...
        (cellsMat > repmat(gridSize,[size(cellsMat,1) 1])),[],2));
    if (~isempty(outside))
        cellsMat(outside,:) = [];
        cellsMatDiag(outside,:) = [];
        values(outside,:) = [];
    end
end

for d = 1:numDims,
    cells{d} = cellsMat(:,d);
    cellsDiag{d} = cellsMatDiag(:,d);
end

index = sub2ind(gridSize,cells{:});
indexDiag = sub2ind(gridSize,cellsDiag{:});
%indexDiag
%length(indexDiag)

A = [A; [indexDiag + g.iBase - 1 index + g.iBase - 1 values]];
