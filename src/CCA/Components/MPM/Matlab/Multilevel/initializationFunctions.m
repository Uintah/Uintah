function [IF] = initializationFunctions

  % create function handles that are used in AMRMPM.m
  IF.initialize_grid     = @initialize_grid;
  IF.initialize_NodePos  = @initialize_NodePos;
  IF.initialize_Lx       = @initialize_Lx;
  IF.initialize_xp       = @initialize_xp;
  
  [GF] = gridFunctions            % load grid based functions
 
  %______________________________________________________________________
  function[nodePos]  = initialize_NodePos(L1_dx, Levels, Limits, interpolation)
  
    nodePos = zeros(Limits.NN_max, Limits.maxLevels);              % node Position
    
    

    if(strcmp(interpolation,'GIMP'))    % Boundary condition Node on level 1
      nodePos(1,1) = -L1_dx;
    else
      nodePos(1,1) = 0.0;
    end;

    for l=1:Limits.maxLevels
      L = Levels{l};
      nodeNum = int32(1);
      
      if(l > 1)
        nodePos(1,l) = L.Patches{1}.min;
      end
      
      for p=1:L.nPatches
        P = L.Patches{p};
        % loop over all nodes and set the node position
        for  n=1:P.NN  
          if(nodeNum > 1)
            nodePos(nodeNum,l) = nodePos(nodeNum-1,l) + P.dx;
          end
          nodeNum = nodeNum + 1;
        end
        
      end  % patches
    end  % level
  end
  
  %______________________________________________________________________
  function[Lx]  = initialize_Lx(nodePos, Levels, Limits)
    Lx = zeros(Limits.NN_max,2, Limits.maxLevels);
    % compute the zone of influence
    l = 1;
    r = 2;
    
    for L=1:Limits.maxLevels
      NN = Levels{L}.NN;
      Lx(1,l,L)  = 0.0;
      Lx(1,r,L)  = nodePos(2,L) - nodePos(1,L);

      Lx(NN,l,L) = nodePos(NN,L) - nodePos(NN-1,L);
      Lx(NN,r,L) = 0.0;

      for n=2:NN-1
        Lx(n,l,L) = nodePos(n,L)   - nodePos(n-1,L);
        Lx(n,r,L) = nodePos(n+1,L) - nodePos(n,L);
      end
    end
    
    % output the regions and the Lx
    for l=1:Limits.maxLevels
      L  = Levels{l};
      nn = 1;
      
      for p=1:L.nPatches
        P = L.Patches{p};
        fprintf('-----------------------Level %g Patch %g\n',l,p);
        for  n=1:P.NN  
          fprintf( 'Node:  %g, nodePos: %6.5f, \t Lx(L): %6.5f Lx(R): %6.5f\n',nn, nodePos(nn,l),Lx(nn,1,l), Lx(nn,2,l));
          nn = nn + 1;
        end
      end
    end
    
  end

  %______________________________________________________________________
  function[xp_allLevels, Levels Limits] = initialize_xp( nodePos, interpolation, PPC, bar_min, bar_max, Limits, Levels)
    %__________________________________
    % create particles
    fprintf('Particle Position\n');

    allLevels_NP = 0;
    
    for L=1:Limits.maxLevels
      NN = Levels{L}.NN;

      fprintf('-----------------------Level %g\n',L);
      ip = 1;

      startNode = 1;
      if( strcmp(interpolation,'GIMP') && L == 1 )
        startNode = 2;
      end


      for n=startNode:NN-1
        dx_p = (nodePos(n+1,L) - nodePos(n,L) )/double(PPC);

        offset = dx_p/2.0;

        for p = 1:PPC
          xp_new = nodePos(n,L) + double(p-1) * dx_p + offset;
          
          test = GF.hasFinerCell(xp_new, L, Levels, Limits);

          if( xp_new >= bar_min && xp_new <= bar_max  && (test==0) )

            xp(ip) = xp_new;

            fprintf('nodePos: %4.5e \t xp(%g) %g \t dx_p: %g \t offset: %g',nodePos(n,L),ip, xp(ip),dx_p,offset);

            if(ip > 1)
              fprintf( '\t \tdx: %g \n',(xp(ip) - xp(ip-1)));
            else
              fprintf('\n');
            end

            ip = ip + 1;

          end
        end
      end

      NP=ip-1;  % number of particles
      
      % place individual particle positions in array
      for ip = 1:NP
        xp_allLevels(ip,L) = xp(ip);
      end

      allLevels_NP = allLevels_NP + NP;      % total number of particles
      Levels{L}.NP = NP;                     % number of particles on that level
      
    end  % levels loop
    
    %xp_allLevels
    Limits.NP_max = allLevels_NP;
  end
  %______________________________________________________________________
  function [Levels, dx_min, Limits] = initialize_grid(domain,PPC,L1_dx,interpolation,d_smallNum)
    
    if(0)
    fprintf('USING plotShapeFunction Patches\n');

    nPatches      = int32(1);              % partition the domain into numPatches
    Patches       = cell(nPatches,1);      % array that holds the individual region information

    R.min         = -1;                     % location of left node
    R.max         = 1;                     % location of right node
    R.dx          = 1;
    R.NN          = int32( (R.max - R.min)/R.dx +1 ); % number of nodes interior nodes
    R.lp          = R.dx/(2 * PPC);
    Patches{1}    = R;
    end


    %____________
    % single level  2 patches
    if(0)
    maxLevels     = 1;
    nPatches      = int32(2);             % partition the domain into nPatches
    Patches       = cell(nPatches,1);     % array that holds the individual region information
    
    P.min         = 0;                    % location of left point
    P.max         = domain/2;             % location of right point
    P.refineRatio = 1;
    P.dx          = L1_dx;
    P.volP        = P.dx/PPC;
    P.NN          = int32( (P.max - P.min)/P.dx +1 );
    P.lp          = P.dx/(2 * PPC);
    Patches{1}    = P;

    P.min         = domain/2;                       
    P.max         = domain;
    P.refineRatio = 1;
    P.dx          = L1_dx/P.refineRatio;
    P.volP        = P.dx/PPC;
    P.NN          = int32( (P.max - P.min)/P.dx );
    P.lp          = P.dx/(2 * PPC);
    Patches{2}    = P;
    end
    
    %______________________________________________________________________
    % 2 levels
    % 3 patches on the coarse level 1 patch on the fine level
    if(1)
    maxLevels     = 2;
    nPatchesL1    = int32(3);               % partition the domain into nPatches
    PatchesL1     = cell(nPatchesL1,1);     % array that holds the individual region information
    
    %Level 1
    P.min         = 0;                      % location of left point
    P.max         = domain/3.0;             % location of right point
    P.refineRatio = 1;
    P.dx          = L1_dx;
    P.volP        = P.dx/PPC;
    P.NN          = int32( (P.max - P.min)/P.dx );
    P.lp          = P.dx/(2 * PPC);
    PatchesL1{1}    = P;

    P.min         = domain/3.0;                       
    P.max         = 2.0*domain/3.0;
    P.refineRatio = 1;
    P.dx          = L1_dx;
    P.volP        = P.dx/PPC;
    P.NN          = int32( (P.max - P.min)/P.dx );
    P.lp          = P.dx/(2 * PPC);
    PatchesL1{2}    = P;

    P.min         = 2.0*domain/3.0;                       
    P.max         = domain;
    P.refineRatio = 1;
    P.dx          = L1_dx
    P.volP        = P.dx/PPC;
    P.NN          = int32( (P.max - P.min)/P.dx); 
    P.lp          = P.dx/(2 * PPC);
    PatchesL1{3}  = P;
    
    % Level 2
    nPatchesL2    = int32(1);               % partition the domain into nPatches
    PatchesL2     = cell(nPatchesL2,1);     % array that holds the individual region information
    P.min         = domain/3.0;                       
    P.max         = 2.0*domain/3.0;
    P.refineRatio = 2;
    P.dx          = L1_dx/P.refineRatio;
    P.volP        = P.dx/PPC;
    P.NN          = int32( (P.max - P.min)/P.dx +1 );
    P.lp          = P.dx/(2 * PPC);
    PatchesL2{1}    = P;
    
    end

    Levels            = cell(maxLevels,1);
    Levels{1}.Patches = PatchesL1;
    Levels{1}.nPatches= nPatchesL1;
    Levels{2}.Patches = PatchesL2;
    Levels{2}.nPatches= nPatchesL2;

    % increase the number of nodes in the first and last patch if using gimp on level 1;
    if(strcmp(interpolation,'GIMP'))
      L1 = Levels{1};
      firstP = L1.Patches{1};
      lastP  = L1.Patches{L1.nPatches};
      
      Levels{1}.Patches{1}.NN             = firstP.NN  + 1;
      Levels{1}.Patches{1}.min            = firstP.min - firstP.dx;
      
      Levels{1}.Patches{L1.nPatches}.NN   = lastP.NN  + 1;
      Levels{1}.Patches{L1.nPatches}.max  = lastP.max + lastP.dx;
    end;

if (0)        % currently extracells are not used.
    % Define the extra cells L & R for each region.
    for p=1:nPatches
      Patches{p}.EC(1) = 0;    
      Patches{p}.EC(2) = 0;    
    end

    if(strcmp(interpolation,'GIMP'))
      Patches{1}.EC(1)        = 1;
      Patches{nPatches}.EC(2) = 1;
    end;
end

    % Count the number of nodes in a level
    NN_allLevels = 0;
    
    for l=1:maxLevels
      L = Levels{l};
      NN = int32(0);
      
      for p=1:L.nPatches
        P = L.Patches{p};
        NN = NN + P.NN;
      end
      Levels{l}.NN = NN
      NN_allLevels = NN_allLevels + NN;
    end
    
    
    %  find the minimum dx on all levels
    dx_min = double(1e100);
    for l=1:maxLevels
      L = Levels{l};
      
      for p=1:L.nPatches
        P = L.Patches{p};
        dx_min = min(dx_min,P.dx);
        fprintf( 'level: %g patch %g, min: %g, \t max: %g \t refineRatio: %g dx: %g, NN: %g\n',l,p, P.min, P.max, P.refineRatio, P.dx, P.NN)
      end
    end

    % defines the limits
    Limits.maxLevels = maxLevels;
    Limits.NN_max    = NN_allLevels;

    % bulletproofing:
    for l=1:maxLevels
      L = Levels{l};
      
      for p=1:L.nPatches
        P = L.Patches{p};
        d = (P.max - P.min) + 100* d_smallNum;

        if( mod( d, P.dx ) > 1.0e-10 )
          fprintf('ERROR, level: %g the dx: %g in patch %g does not divide into the domain (P.max:%g P.min:%g) evenly\n', l,P.dx,p,P.max,P.min);
          return;
        end
      end  % patches
    end  % levels
    
  end
  
 
end
