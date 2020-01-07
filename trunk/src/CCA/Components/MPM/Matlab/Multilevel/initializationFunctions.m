

function [IF] = initializationFunctions

  % create function handles that are used in AMRMPM.m
  IF.initialize_grid     = @initialize_grid;
  IF.initialize_NodePos  = @initialize_NodePos;
  IF.initialize_Lx       = @initialize_Lx;
  IF.initialize_xp       = @initialize_xp;
  IF.NN_NP_allLevels     = @NN_NP_allLevels;
  IF.findCFI_nodes       = @findCFI_nodes;
  
  [GF] = gridFunctions;            % load grid based functions
  
  %______________________________________________________________________
  function[Levels]  = findCFI_nodes(Levels, nodePos, Limits)
     
    left  = 1;
    right = 2;
    
    for fl=Limits.maxLevels:-1:2     % fine level
      cl = fl -1;                    % coarse level
      L = Levels{fl};
      nExtraCells = L.EC;
      
      % fine level CFI nodes
      cfi_left  = L.Patches{1}.nodeLo + nExtraCells;
      cfi_right = L.Patches{L.nPatches}.nodeHi - nExtraCells;
      Levels{fl}.CFI_nodes(left)  = cfi_left;
      Levels{fl}.CFI_nodes(right) = cfi_right;
      
      % underlying CFI nodes on the coarse level
      cfi_left  = GF.mapNodetoCoarser(nodePos(cfi_left, fl), fl,nodePos,Levels);
      cfi_right = GF.mapNodetoCoarser(nodePos(cfi_right,fl), fl,nodePos,Levels);
      Levels{cl}.CFI_nodes(left)  = cfi_left;
      Levels{cl}.CFI_nodes(right) = cfi_right;
      
      fprintf('FineLevel:%g, CFI_nodes(%g, %g),  CoarseLevel:%g,  CFI_nodes(%g, %g)\n',fl,Levels{fl}.CFI_nodes(left), Levels{fl}.CFI_nodes(right),cl, Levels{cl}.CFI_nodes(left), Levels{cl}.CFI_nodes(right) ); 
    end
  end
 
  %______________________________________________________________________
  function[nodePos]  = initialize_NodePos(L1_dx, Levels, Limits, interpolation)
  
    nodePos = NaN(Limits.NN_max, Limits.maxLevels);              % node Position

    for l=1:Limits.maxLevels
      L = Levels{l};
      nodeNum = int32(1);
      
      nodePos(1,l) = L.Patches{1}.min;     
      
      for p=1:L.nPatches
        P = L.Patches{p};
        % loop over all nodes and set the node position

        if(p > 1)
          nodePos(nodeNum+1,l) = P.min;
        end

        while((nodePos(nodeNum,l) + P.dx) <= P.max)
          nodeNum = nodeNum +1;
          nodePos(nodeNum,l) = nodePos(nodeNum-1,l) + P.dx;
        end
      end
      
      
    end  % level
  end
  
  %______________________________________________________________________
  function[Lx]  = initialize_Lx(nodePos, Levels, Limits)
    
    Lx = NaN(Limits.NN_max,2, Limits.maxLevels);
    % compute the zone of influence
    left  = 1;
    right = 2;
    
    for L=1:Limits.maxLevels
      NN = Levels{L}.NN;
      Lx(1,left,L)   = 0.0;
      Lx(1,right,L)  = nodePos(2,L) - nodePos(1,L);

      Lx(NN,left,L)  = nodePos(NN,L) - nodePos(NN-1,L);
      Lx(NN,right,L) = 0.0;

      for n=2:NN-1
        Lx(n,left,L)  = nodePos(n,L)   - nodePos(n-1,L);
        Lx(n,right,L) = nodePos(n+1,L) - nodePos(n,L);
      end
    end
    
    %__________________________________
    % Set values at the fine/coarse level CFIs   
    for fl=2:Limits.maxLevels
      cl = fl-1;
      fL = Levels{fl};
      cL = Levels{fl-1};
      
      fineCFI_L  = fL.CFI_nodes(left);
      fineCFI_R  = fL.CFI_nodes(right);
      
      coarseCFI_L = cL.CFI_nodes(left);
      coarseCFI_R = cL.CFI_nodes(right);
      
      %fine level
      Lx(fineCFI_L,left, fl) = cL.dx;
      Lx(fineCFI_R,right,fl) = cL.dx;
      
      %coarseLevel
      Lx(coarseCFI_L,right, cl) = fL.dx;
      Lx(coarseCFI_R,left,  cl) = fL.dx;  
          
      range = [coarseCFI_L+1:coarseCFI_R-1];
      Lx(range,:,cl) = 0.0;
    end
    
    
    % output the regions and Lx
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

    NP_max = 0;
    maxLevels = Limits.maxLevels;
    
    for L=1:maxLevels
      
      N_interiorLo = Levels{L}.N_interiorLo;
      N_interiorHi = Levels{L}.N_interiorHi;

      fprintf('-----------------------Level %g\n',L);
      ip = 1;

      startNode = N_interiorLo;
      if( strcmp(interpolation,'GIMP') && L == 1 )
        startNode = 2;
        fprintf(' this needs to be fixed \n');
        input('');
      end


      for n=startNode:N_interiorHi-1
        dx_p = (nodePos(n+1,L) - nodePos(n,L) )/double(PPC);

        offset = dx_p/2.0;

        for p = 1:PPC
          xp_new = nodePos(n,L) + double(p-1) * dx_p + offset;
          
          test = GF.hasFinerCell(xp_new, L, Levels, Limits, 'withoutExtraCells');

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
      
      % place individual particle positions into a temporary array
      for ip = 1:NP
        xp_tmp(ip,L) = xp(ip);
      end

      NP_max = max(NP_max,NP);      % max number of particles on any level
      Levels{L}.NP = NP;            % number of particles on that level
      
    end  % levels loop
    
    %xp_allLevels
    Limits.NP_max = NP_max;
    
    
    xp_allLevels = NaN(NP_max,maxLevels);
    for L=1:maxLevels
      for ip = 1:Levels{L}.NP
        xp_allLevels(ip,L) = xp_tmp(ip,L);
      end
    end
    
    
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
    P.dx          = L1_dx;
    P.volP        = P.dx/PPC;
    P.NN          = int32( (P.max - P.min)/P.dx +1); 
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
    Levels{1}.dx      = PatchesL1{1}.dx;
    Levels{1}.Patches = PatchesL1;
    Levels{1}.nPatches= nPatchesL1;
    Levels{1}.RR      = 1;                          % refinement Ratio always 1
    
    Levels{2}.dx      = PatchesL2{1}.dx;
    Levels{2}.Patches = PatchesL2;
    Levels{2}.nPatches= nPatchesL2;
    Levels{2}.RR      = PatchesL2{1}.refineRatio;  % refinement Ratio

    %______________________________________________________________________
    % On the coarsest level & GIMP there must be 1 extra cell
    % On fine levels & GIMP there must be 3 extra cells
    %        *---EC---*           *---EC---*           
    % L-1    |..|..|..|..|..|..|..|..|..|..|         RefineRatio = 2
    % L-0 |.....|.....|.....|.....|.....|.....|
    %
    % On fine levels & LINEAR there must be 2 extra cells
    %          *--EC--*           *-EC--*           
    % L-1       |..|..|..|..|..|..|..|..|            RefineRatio = 2
    % L-0 |.....|.....|.....|.....|.....|.....|  
  
    % increase the number of nodes in the first and last patch if using gimp;
    if(strcmp(interpolation,'LINEAR'))
      [Levels] = AddExtraCells(Levels, 1, 0);
      
      for l=2:maxLevels
        [Levels] = AddExtraCells(Levels, l, 2);
      end
    end
    
    if(strcmp(interpolation,'GIMP'))
      [Levels] = AddExtraCells(Levels, 1, 1);
      
      for l=2:maxLevels
        [Levels] = AddExtraCells(Levels, l, 3);
      end
      
    end

    % Count the number of nodes (NN) and number of interior nodes in a level
    NN_max = 0;
    
    for l=1:maxLevels
      L = Levels{l};
      NN = int32(0);
      
      for p=1:L.nPatches
        P = L.Patches{p};
        NN = NN + P.NN;
      end
      Levels{l}.NN = NN;
      Levels{l}.interiorNN    = NN - 2*L.EC;
      Levels{l}.N_interiorLo  = 1 + L.EC;
      Levels{l}.N_interiorHi  = Levels{l}.N_interiorLo + Levels{l}.interiorNN - 1;
      NN_max = max(NN_max, NN);
    end
    
    %Determine what nodes are the extra cell nodes
    for l=1:maxLevels
      L = Levels{l};
      NN = L.NN;
      
      n = [1:NN];
      EC_nodes = find( n<L.N_interiorLo | n>L.N_interiorHi);
      Levels{l}.EC_nodes = EC_nodes;
    end
    
    
    % determine low and high nodes for each patch
    for l=1:maxLevels
      L = Levels{l};
      NN = int32(1);
      lp = L.nPatches;
      
      for p=1:lp
        P = L.Patches{p};
        Levels{l}.Patches{p}.nodeLo = NN;
        Levels{l}.Patches{p}.nodeHi = NN + P.NN -1;
        NN = NN + P.NN;
      end
      Levels{l}.Patches{1}.interiorNodeLo  = Levels{l}.Patches{1}.nodeLo  + L.EC;
      Levels{l}.Patches{lp}.interiorNodeHi = Levels{l}.Patches{lp}.nodeHi - L.EC;
    end
    
    % compute the extents of each level
    for l=1:maxLevels
      L = Levels{l};
      Lmax = 0;
      Lmin = 1000;
      
      for p=1:L.nPatches
        P = L.Patches{p};
        Lmax = max(Lmax,P.max);
        Lmin = min(Lmin,P.min);
      end
      Levels{l}.max = Lmax;
      Levels{l}.min = Lmin;
    end
    
    %  find the minimum dx on all levels
    dx_min = double(1e100);
    for l=1:maxLevels
      L = Levels{l};
      
      for p=1:L.nPatches
        P = L.Patches{p};
        dx_min = min(dx_min,P.dx);
      end
    end
   
    % defines the limits
    Limits.maxLevels = maxLevels;
    Limits.NN_max    = NN_max;


    %  output grid
    GF.outputLevelInfo(Levels,Limits);
    
    
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
  
  %__________________________________
  %  This function adds ncells of ghostcells to level l
  function [Levels] = AddExtraCells(Levels, l, ncells)
  
    L = Levels{l};                                              
    dx = L.dx;                                                        
                                                                      
    firstP = L.Patches{1};                                    
    Levels{l}.Patches{1}.NN             = firstP.NN  + ncells;      
    Levels{l}.Patches{1}.min            = firstP.min -  dx * ncells;  
    
    lastP = Levels{l}.Patches{L.nPatches};
    Levels{l}.Patches{L.nPatches}.NN          = lastP.NN  + ncells;
    Levels{l}.Patches{L.nPatches}.interiorMax = lastP.max;        
    Levels{l}.Patches{L.nPatches}.max         = lastP.max + dx * ncells; 
    
    Levels{l}.EC = ncells;
    Levels{l}.interiorMin  = firstP.min;
    Levels{l}.interiorMax  = lastP.max; 
    
    fprintf('\nAddExtraCells \n');
    fprintf('before: L-%i  P1.NN %g, P1.min: %g  LP.NN:%g LP.min: %g \n',l, firstP.NN, firstP.min, lastP.NN, lastP.max);
    fprintf('After:  L-%i  P1.NN %g, P1.min: %g  LP.NN:%g LP.min: %g \n',l,Levels{l}.Patches{1}.NN, Levels{l}.Patches{1}.min, Levels{l}.Patches{L.nPatches}.NN, Levels{l}.Patches{L.nPatches}.max);
  end
  
  %______________________________________________________________________
  function [Limits] = NN_NP_allLevels(Levels, Limits)
    NP_allLevels = 0;
    NN_allLevels = 0;
    
    for l=1:Limits.maxLevels
      L = Levels{l};
      NP_allLevels = NP_allLevels + L.NP;
      NN_allLevels = NN_allLevels + L.NN;
    end
    Limits.NP_allLevels = NP_allLevels;
    Limits.NN_allLevels = NN_allLevels;
  end  
 
end
