function [GF] = gridFunctions
  GF.hasFinerCell     = @hasFinerCell;
  GF.isOutsideLevel   = @isOutsideLevel;
  GF.mapNodetoCoarser = @mapNodetoCoarser;
  GF.outputLevelInfo  = @outputLevelInfo;

  %__________________________________
  function[test] = hasFinerCell(x,curLevel,Levels,Limits, string)
      
    nextLevel = curLevel +1;  
    for l=nextLevel:Limits.maxLevels
      L = Levels{l};
      min = -9;
      max = -9;
      
      switch (string)
        case {'withExtraCells'}
          min = L.min;
          max = L.max;
        case {'withoutExtraCells'}
          min = L.interiorMin;
          max = L.interiorMax;
        otherwise
          disp('ERROR: hasFinerCell: unknown input');
      end
        
      if( x >= min && x <= max )  
        test = 1;                    
        return                       
      end                            
        
    end  % level
    
    test = 0;
  end

  %__________________________________
  function[node] = mapNodetoCoarser(x,curLevel,nodePos,Levels)
      
    coarseLevel = curLevel -1;  
    L = Levels{coarseLevel};
    tolerance=1e-10;
    
    for ig =1:L.NN
      diff= abs(nodePos(ig,coarseLevel) - x);
      
      if(diff <= tolerance)
        node = ig;
        return;
      end
    end 
    fprintf('ERROR: mapNodetoCoarser\n');
    fprintf('Could not find coarse node corresponding to position %g \n',x);
    input('hit return to continue');
  end  
  
  
  %__________________________________
  function[test] = isOutsideLevel(x,curLevel,Levels)
  
    if ( (x <= Levels{curLevel}.min) || (x >= Levels{curLevel}.max) )
      test = 1;
    else
      test = 0;
    end
  end  
  
  %__________________________________
  function outputLevelInfo(Levels,Limits)
    %  output grid
    for l=1:Limits.maxLevels
      L = Levels{l};
      
      for p=1:L.nPatches
        P = L.Patches{p};
        fprintf( 'level: %g patch %g, min: %g, \t max: %g \t refineRatio: %g dx: %g, NN: %g lo: %g hi: %g\n',l,p, P.min, P.max, P.refineRatio, P.dx, P.NN, P.nodeLo, P.nodeHi);
      end
    end
    fprintf('\n\n');
    
    for l=1:Limits.maxLevels
      L = Levels{l};
      fprintf('level: %g, dx: %g nPatches: %g NN:%g min: %g  max:%g  ',l, L.dx, L.nPatches, L.NN, L.min, L.max );
      fprintf('\t interiorNN: %g, N_interiorLo: %g N_interiorHi: %g interiorMin: %g   interiorMax: %g\n',L.interiorNN, L.N_interiorLo, L.N_interiorHi,L.interiorMin, L.interiorMax);
      fprintf('\t  ExtraCell Nodes: ');
      fprintf('%g, ',L.EC_nodes(:));
      fprintf('\n');
    end
  end
  
end  % gridFunctions
