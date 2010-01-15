function [GF] = gridFunctions
  GF.hasFinerCell = @hasFinerCell;
  GF.isOutsideLevel = @isOutsideLevel;
  GF.mapNodetoCoarser = @mapNodetoCoarser;

  %__________________________________
  function[test] = hasFinerCell(x,curLevel,Levels,Limits)
      
    nextLevel = curLevel +1;  
    for l=nextLevel:Limits.maxLevels
      L = Levels{l};
      
      for p=1:L.nPatches
        P = L.Patches{p};
        
        if( x >= P.min && x <= P.max)
          test = 1;
          return
        end
        
      end  % patches
    end  % level
    
    test = 0;
  end
  
  %__________________________________
  function[node] = mapNodetoCoarser(x,curLevel,nodePos,Levels)
      
    coarseLevel = curLevel -1;  
    L = Levels{coarseLevel};
    
    for ig =1:L.NN
      if(nodePos(ig,coarseLevel) == x)
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
  
end  % gridFunctions
