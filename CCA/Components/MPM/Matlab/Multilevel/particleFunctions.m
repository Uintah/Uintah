function [pf] = particleFunctions()

  pf.createParticleStruct  = @createParticleStruct;
  pf.disolveParticleStruct = @disolveParticleStruct;  
  pf.createGhostParticles  = @createGhostParticles;
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
  %  This function creates ghost particles
  function [levels, P] = createGhostParticles(P, Levels, Limits, interpolation)
  disp('inside create ghost particles')
  end
  
  %__________________________________
  function [levels, xp, massP, velP, stressP, vol] = deleteGhostParticles()
  end
  
  

end
