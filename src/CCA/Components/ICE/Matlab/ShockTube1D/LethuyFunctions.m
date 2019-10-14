function [LF] = LethuyFunctions()
  % create function handles that are used in .m
  LF.computeVel_FC  = @computeVel_FC;
  %______________________________________________________________________
  
  
  %_____________________________________________________
  %  Compute the face-centered velocities  
  function[xvel_FC] = computeVel_FC(rho_CC, xvel_CC, spvol_CC, E_CC, press_eq_CC, delT, P, G)

    dRho_CC = zeros(G.first_CC,G.last_CC);      
    dU_CC   = zeros(G.first_CC,G.last_CC);
    dE_CC   = zeros(G.first_CC,G.last_CC);
    
    display('Lethuy velFC');
    for j = G.first_CC+1:G.last_CC-1  % dRho, dU and dE at first and last cells are zero        
      dRho_CC(j) = limiter(rho_CC(j-1),  rho_CC(j),  rho_CC(j+1));                              
      dU_CC(j)   = limiter(xvel_CC(j-1), xvel_CC(j), xvel_CC(j+1));                             
      dE_CC(j)   = limiter(E_CC(j-1),    E_CC(j),    E_CC(j+1));                                
    end
                                                                                 
    for j = G.first_FC+1:G.last_FC-1                                                            
      rhoL = rho_CC(j-1) + 0.5*dRho_CC(j-1);                                                    
      rhoR = rho_CC(j)   - 0.5*dRho_CC(j);                                                      
                                                                                                
      uL   = xvel_CC(j-1) + 0.5*dU_CC(j-1);                                                     
      uR   = xvel_CC(j)   - 0.5*dU_CC(j);                                                       
      EL   = E_CC(j-1)    + 0.5*dE_CC(j-1);                                                     
      ER   = E_CC(j)      - 0.5*dE_CC(j);                                                       
                                                                                                
      pL   = (P.gamma-1) * rhoL * (EL -0.5*uL*uL);                                              
      cL   = sqrt(P.gamma*pL/rhoL);                                                             
                                                                                                
      pR   = (P.gamma-1) * rhoR * (ER -0.5*uR*uR);                                              
      cR   = sqrt(P.gamma*pR/rhoR);                                                             
                                                                                                
      val1 = uL - cL;                                                                           
      val2 = uR + cR;                                                                           
                                                                                                
      aL   = min(val1,val2);                                                                    
      aR   = max(val1,val2);                                                                    
      if( aR < 0)                                                                               
        velavg_FC(j) = uR;                                                                      
        rhoavg_FC(j) = rhoR;                                                                    
      elseif ( aL > 0)                                                                          
        velavg_FC(j) = uL;                                                                      
        rhoavg_FC(j) = rhoL;                                                                    
      else                                                                                      
        rhoLR = ((aR*rhoR - aL*rhoL) - (rhoR*uR - rhoL*uL))/(aR - aL);                          
        uLR   = ((aR*uR   - aL*uL)   - (uR*uR/2.0 - uL*uL/2.0 + (pR - pL)/rhoLR))/(aR - aL);    
        velavg_FC(j) = uLR;                                                                     
        rhoavg_FC(j) = rhoLR;                                                                   
      end                                                                                       
    end                                                                                         
        
    for j = G.first_FC+1:G.last_FC-1   % Loop over all xminus cell faces
        L = j-1;    R = j;
        term2b      = (press_eq_CC(R) - press_eq_CC(L))/G.delX;    
        xvel_FC(j)  =  velavg_FC(j) - delT*term2b/rhoavg_FC(j);
    end 
  end
  
  %__________________________________
  function WL = limiter(WU,WC,WD) 
    epsilon = 1e-100; 
    r = (WC - WU)/(WD - WC+epsilon);
    phir = max(0.0,min(min(2*r,0.5+0.5*r),2.0));
    WL = phir*(WD-WC); 
  end
  
     
end
