function data = setBoxValues(data,P,ilower,iupper,entry,values,type)
fprintf('================= setBoxValues ===============\n');

PSize   = prod(P.iupper-P.ilower+1);
POffset = -P.ilower+2;                   % Add to physical cell index to get patch cell index

range1 = [ilower(1):iupper(1)] + POffset(1);
range2 = [ilower(2):iupper(2)] + POffset(2);

switch lower(type)   % LHS matrix, data is LHS matrix data structure
    
    case 'matrix'
        
        data(range1,range2,entry) = values;
        data(range1,range2,:)
        
    case 'rhs',     % type == 'rhs', data is the rhs vector data structure
        
        data(range1,range2) = values;

    otherwise,
        
        error('setBoxValues: unknown type');
end
