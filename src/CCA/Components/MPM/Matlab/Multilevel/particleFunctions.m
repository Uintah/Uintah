function [pf] = particleFunctions()
  global gf;

  [gf]  = gridFunctions;            % load all grid based function
  pf.createParticleStruct     = @createParticleStruct;
  pf.disolveParticleStruct    = @disolveParticleStruct;  
  pf.findExtraCellParticles   = @findExtraCellParticles;
  pf.createEC_particleStruct  = @createEC_particleStruct;
  pf.deleteGhostParticles     = @deleteGhostParticles;
  pf.getCopy                  = @getCopy;

  %__________________________________
  %  get copies of the particle data out of the particleSet.
  function [varargout] = getCopy(particleSet, varargin)

    for k = 1:length(varargin)    % loop over each input argument
      name = varargin{k};
      varargout{k} =getfield(particleSet,name);
    end
  end

  %__________________________________
  %  This puts the particle arrays into a struct
  function [P] = createParticleStruct(varargin)

    for k = 1:length(varargin)    % loop over each input argument
      pVar = varargin{k};
      name = inputname(k);
      P.(name)=pVar;
    end
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
  %  This function returns the particle indices of the extra Cell particles 
  function [pID] = findExtraCellParticles(P, Levels, Limits, nodePos, interpolation)
    %disp('findExtraCellParticles')
    

    for fl=2:Limits.maxLevels 
      cl  = fl -1;  
      FL  = Levels{fl};
      CL  = Levels{cl};
      
      % Identify the coarse level particles that are beneath fine level extra cells
      %          *--EC--*           *-EC--*           
      % L-1       |  |  |..|..|..|..|  |  |            RefineRatio = 2
      % L-0 | . . | . . |     |     | . . | . . |  
      %           ^-----^           ^-----^  
      %               In these regions 
      left = 1;
      right = 2;
      
      % Left ECs
      nodeHi    = FL.CFI_nodes(left);
      nodePosHi = nodePos(nodeHi,fl);
      nodePosLo = nodePosHi - CL.dx;      % Look at art
      
      pID_L = findParticlesInRegion(nodePosLo,nodePosHi,P,cl,Levels);

      % Right ECs      
      nodeLo    = FL.CFI_nodes(right);
      nodePosLo = nodePos(nodeLo,fl);
      nodePosHi = nodePosLo + CL.dx;      % Look at art
                  
      pID_R= findParticlesInRegion(nodePosLo,nodePosHi,P,cl,Levels);
      
      pID(:,cl) = vertcat(pID_L, pID_R);

      % Identify the fine level particles that are above the coarse level CFI cells
      %          *--EC--*           *-EC--*           
      % L-1       |  |  |..|..|..|..|  |  |            RefineRatio = 2
      % L-0 | . . | . . |     |     | . . | . . |  
      %                 ^--^     ^--^  
      %               In these regions
      % Left ECs      
      nodeLo    = CL.CFI_nodes(left);
      nodePosLo = nodePos(nodeLo,cl);
      nodePosHi = nodePosLo + FL.dx;      % Look at art
      
      pID_L = findParticlesInRegion(nodePosLo,nodePosHi,P,fl,Levels);
     
      % Right ECs      
      nodeHi    = CL.CFI_nodes(right);
      nodePosHi = nodePos(nodeHi,cl);
      nodePosLo = nodePosHi - FL.dx;      % Look at art
            
      pID_R = findParticlesInRegion(nodePosLo,nodePosHi,P,fl,Levels);      
      
      pID(:,fl) = vertcat(pID_L, pID_R);
    end
  end
  
  %__________________________________
  function [EC_P] = createEC_particleStruct(P, pID, Levels, Limits)
    
    f_names = fieldnames(P);              % find all of the field names in the struct
    
    for c=1:length(f_names)               % loop over all of the fields in the struct P
      fieldName = f_names{c};
      
      if( ~ strcmp(fieldName,'interpolation_dx') && ~ strcmp(fieldName,'CompDomain' ) && ~ strcmp(fieldName,'NP'))
        for cl=1:Limits.maxLevels-1 
          fl = cl + 1;

          cl_pID = pID(:,cl);                % extra cell particle indices on the coarse/fine level
          fl_pID = pID(:,fl);

          var = getfield(P,fieldName);      % get the field from the main array;

          % create the extra cell particle arrays for this field   
          EC_P.(fieldName)(:,fl) = vertcat(var(cl_pID, cl));
          EC_P.(fieldName)(:,cl) = vertcat(var(fl_pID, fl));


        end
      end
    end
    
     %set what dx should be used during interpolation in the extra cells 
    for cl=1:Limits.maxLevels-1 
      fl = cl + 1;                         
      EC_P.interpolation_dx(fl) = Levels{cl}.dx;    % on the fine level use the coarse level dx.  
      EC_P.interpolation_dx(cl) = Levels{fl}.dx;    % on the coarse level use the fine level dx. 
    end
    
    % define number of particles in the set
    for l=1:Limits.maxLevels
      EC_P.NP(l) = length(pID(:,l));
    end
    
    % define the computational domain
    EC_P.CompDomain = 'ExtraCells';
  end
  
  
  %__________________________________
  %  Find which particles are in a cell
  function [pID]= findParticlesInRegion(xLo,xHi,P,curLevel,Levels)
    xLo = min(xLo, xHi);    % just in case the user reverses them
    xHi = max(xLo, xHi);
    
    xp = P.xp(:,curLevel);
    
    pID = find( (xp>=xLo) & (xp <= xHi) );
    
    if(0)
      fprintf('Particle in cell region [%g %g] \n indx: ',xLo,xHi);
      fprintf('%i ',pID);
      fprintf('\n Pos: ');
      fprintf(' %g ',xp(pID));
      fprintf('\n');
    end
  end

end
