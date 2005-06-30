function list = printSparse(A,fileName)
if (nargin < 2)
    f = 1;
else
    f = fopen(fileName,'w');
end

[i,j] = find(A);
data = full(A(find(A)));
list = [i j data];
list = sortrows(list);

for k = 1:size(list,1)
    if (k > 1)
        if (list(k,1) ~= list(k-1,1))
            fprintf(f,'\n');
        end
    end
    fprintf(f,'(%5d , %5d)   %+f\n',list(k,:));
end

if (nargin >= 2)
    fclose(f);
end
