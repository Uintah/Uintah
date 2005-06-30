function list = printSparse(A,b,fileName)
if (nargin < 3)
    f = 1;
else
    f = fopen(fileName,'w');
end

[i,j,data] = find(A);
list = [i j data];
list = sortrows(list);

for k = 1:size(list,1)
    if (k > 1)
        if (list(k,1) ~= list(k-1,1))
            fprintf(f,'b(%d) = %+f\n',list(k-1,1),b(list(k-1,1)));
            fprintf(f,'\n');
        end
    end
    fprintf(f,'(%5d , %5d)   %+f\n',list(k,:));
end

if (nargin >= 3)
    fclose(f);
end
