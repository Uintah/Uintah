function [GF] = gridFunctions
  GF.hasFinerCell = @hasFinerCell;

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
  
  
end  % gridFunctions
