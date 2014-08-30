classdef Table
  % a class to load TabProps tables
  
  properties (SetAccess=private, GetAccess=public)
    npts;      % number of points in each dimension of the table
    ndvar;     % number of dependent variables
    ivar;      % independent variables
    dvar;      % dependent variables
    ivarNames; % names of independent variables
    dvarNames; % names of dependent variables
  end
  
  
  methods (Access=public)
    
    function table = Table( fileName )
      eval(fileName);
      table.npts      = npts;
      table.ndvar     = ndvar;
      table.ivar      = ivar;
      table.dvar      = dvar;
      table.ivarNames = ivarNames;
      table.dvarNames = dvarNames;
    end
    
    function var = getVar( table, varName )
      % obtain the requested variable on a 1-D array
      ix = Table.getVarIndex( varName, table.ivarNames );
      if ix<1
        ix = Table.getVarIndex( varName, table.dvarNames );
        if ix<1
          error(strcat('no matching name found for: "',varName,'"'));
        end
      end
      var = table.dvar{ix};
    end
    
    function var = getNDVar( table, varName )
      % obtain the requested variable on an n-D array
      ix = Table.getVarIndex( varName, table.ivarNames );
      if ix<1
        ix = Table.getVarIndex( varName, table.dvarNames );
        if ix<1
          error('no matching name found');
        end
        var = table.reshape_it( table.dvar{ix} );
      else
        var = table.reshape_it( table.ivar{ix} );
      end
    end
    
    function phi = query( table, varName, varargin )
      % Interpolate the table to obtain the value of the variable at the
      % requested point(s).
      %
      % For a table with 1 independent variable:
      %  ivar1 = linspace(0,1);
      %  phi = table.query('Temperature', ivar1 );
      %
      % For a table with 2 independent variables:
      %   phi = table.query('Temperature', ivar1, ivar2 );
      %
      ix = Table.getVarIndex( varName, table.dvarNames );
      switch length(table.ivarNames)
        case 1
          assert( length(varargin) == 1 );
          xint = Table.clip( varargin{1}, min(table.ivar{1}), max(table.ivar{1}) );
          phi = interp1( table.ivar{1}, table.dvar{ix}, xint );
        case 2
          assert( length(varargin) == 2 );
          xint1 = Table.clip( varargin{1}, min(table.ivar{1}), max(table.ivar{1}) );
          xint2 = Table.clip( varargin{2}, min(table.ivar{2}), max(table.ivar{2}) );
          x1 = reshape( table.ivar{1}, table.npts(1), table.npts(2) );
          x2 = reshape( table.ivar{2}, table.npts(1), table.npts(2) );
          y  = reshape( table.dvar{ix},table.npts(1), table.npts(2) );
          phi = interp2( x1', x2', y', xint1, xint2 );
        case 3
          % jcs need to permute these.  see "help permute"
          error('not ready');
          assert( length(varargin) == 3 );
          xint1 = Table.clip( varargin{1}, min(table.ivar{1}), max(table.ivar{1}) );
          xint2 = Table.clip( varargin{2}, min(table.ivar{2}), max(table.ivar{2}) );
          xint3 = Table.clip( varargin{3}, min(table.ivar{3}), max(table.ivar{3}) );
          x1 = reshape( table.ivar{1}, table.npts(1), table.npts(2), table.npts(3) );
          x2 = reshape( table.ivar{2}, table.npts(1), table.npts(2), table.npts(3) );
          x3 = reshape( table.ivar{3}, table.npts(1), table.npts(2), table.npts(3) );
          y  = reshape( table.dvar{ix},table.npts(1), table.npts(2), table.npts(3) );
          phi = interp2( x1', x2', x3', y', xint1, xint2, xint3 );
          
      end
    end

  end
  
  
  methods (Access=private)
        
    function var = reshape_it( table, var )
      ndim = length( table.npts );
      switch ndim
        case 1
          var = var;
        case 2
          var = reshape( var, table.npts(1), table.npts(2) );
        case 3
          var = reshape( var, table.npts(1), table.npts(2), table.npts(3) );
      end
    end
    
  end

  
  methods (Static)
  
    function ix = getVarIndex( name, nameList )
      ix = -1;
      for i=1:length(nameList)
        if strcmp( name, nameList{i} )
          ix = i;
          break;
        end
      end
    end
    
    function x = clip( x, xmin, xmax )
      ilo = find( x < xmin );
      ihi = find( x > xmax );
      x(ilo) = xmin;
      x(ihi) = xmax;
    end
  
  end
  
end