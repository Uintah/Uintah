function [pf] = particleFunctions()
  global gf;

  [gf]  = gridFunctions;            % load all grid based function
  pf.createParticleStruct  = @createParticleStruct;
  pf.disolveParticleStruct = @disolveParticleStruct;  
  pf.createExtraCellParticles  = @createExtraCellParticles;
  pf.deleteGhostParticles  = @deleteGhostParticles;

  %__________________________________
  %  This puts the particle arrays into a struct
  function [P] = createParticleStruct(xp, massP, velP, stressP, vol, lp)
    P.xp    = xp;
    P.massP = massP;
    P.velP  = stressP;
    P.vol   = vol;
    P.lp    = lp;
  end
  
  %__________________________________
  %  This extracts the particle variables from the struct
  function [xp, massP, velP, stressP, vol, lp] = disolveParticleStruct(P)
    xp      = P.xp;
    massP   = P.massP;
    velP    = P.velP;
    vol     = P.vol;
    lp      = P.lp;
  end
  
  %__________________________________
  %  This function find extra Cell particles and creates a multi-level particle set 
  function [EC_P] = createExtraCellParticles(P, Levels, Limits, nodePos, interpolation)
    disp('inside create ghost particles')
    
    % copy the coarse level particles that are beneath fine level extra cells to the fine level extra cells 
    for fl=2:Limits.maxLevels 
      cl  = fl -1;  
      FL  = Levels{fl};
      CL  = Levels{cl};
      ECN = length(FL.EC_nodes);
      dx = FL.dx;
      
      for c=1:ECN     % find the particle ID on the 
        n = FL.EC_nodes(c);
        xLo = nodePos(n,fl);
        xHi = xLo + dx;
        
        pID(c) = findParticlesInRegion(xLo,xHi,P,cl,Levels);
      end
      fprintf('fl:%g \n',fl);
      pID
      
      EC_P.xp     = P.xp(pID,cl);
      EC_P.massP  = P.massP(pID,cl);
      EC_P.velP   = P.velP(pID,cl);
      EC_P.vol    = P.vol(pID,cl);
      EC_P.lp     = P.lp(pID,cl);
      
      % copy the fine level particles that are above the coarse level CFI cells to the coarse level.
      left = 1;
      right = 2;
      xLo = CL.coarseCFI_nodes(left);
      xHi = xLo + dx;
      findParticlesInRegion(xLo,xHi,P,fl,Levels);
      
      xLo = CL.coarseCFI_nodes(right);
      xHi = xLo - dx;
      findParticlesInRegion(xLo,xHi,P,fl,Levels);      
      
    end
  end
  
  %__________________________________
  function [levels, xp, massP, velP, stressP, vol] = deleteGhostParticles()
  end
  
  
  %__________________________________
  %  Find which particles are in a cell
  function [pID]= findParticlesInRegion(xLo,xHi,P,curLevel,Levels)
    xp = P.xp(:,curLevel);
    
    pID = find( (xp>=xLo) & (xp <= xHi) );
    
    fprintf('Particle in cell region [%g %g] \n indx: ',xLo,xHi);
    fprintf('%i ',pID);
    fprintf('\n Pos: ');
    fprintf(' %g ',xp(pID));
    fprintf('\n');
  end

end
