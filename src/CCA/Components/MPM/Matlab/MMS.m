function [s] = MMS()
  % create function handles that are used in AMRMPM.m
  s.displacement        = @MMS_displacement;
  s.deformationGradient = @MMS_deformationGradient;
  s.velocity            = @MMS_velocity;
  s.stress              = @MMS_stress;
  s.accl_bodyForce      = @MMS_accl_bodyForce;
  s.stress              = @MMS_stress;
  s.plotResults         = @MMS_plotResults;

  %______________________________________________________________________
  %  The following MMS functions are described in 
  % M. Steffen, P.C. Wallstedt, J.E. Guilkey, R.M. Kirby, and M. Berzins 
  % "Examination and Analysis of Implementation Choices within the Material Point Method (MPM)
  % CMES, vol. 31, no. 2, pp. 107-127, 2008
  %__________________________________
  %  Equation 47
  function [dp] = MMS_displacement(xp_initial, t, NP, speedSound, h)
    dp  = zeros(NP,1);
    A = 0.05;   % Hardwired

    for ip=1:NP
      dp(ip) = A * sin(2.0 * pi * xp_initial(ip)/h ) * cos( speedSound * pi * t/h);
    end  
  end  

  %__________________________________
  % Equation 48
  function [F] = MMS_deformationGradient(xp_initial, t, NP,speedSound, h)
    F  = zeros(NP,1);
    A = 0.05;   % Hardwired

    for ip=1:NP
      F(ip) = 1.0 + (2.0 * A * pi * cos(2.0 * pi * xp_initial(ip)/h) * cos( speedSound * pi * t/h) )/h;
    end  
  end
  %__________________________________
  function [velExact] = MMS_velocity(xp_initial, t, NP, speedSound, h)
    velExact  = zeros(NP,1);
    A = 0.05;   % Hardwired

    c1 = -speedSound * pi * A/h;

    for ip=1:NP
      velExact(ip)    = c1 * sin( 2.0 * pi * xp_initial(ip))*sin( speedSound * pi * t);
    end
  end

  %__________________________________
  function [stressExact] = MMS_stress(xp_initial, t, NP, speedSound, h, E)
    stressExact  = zeros(NP,1);

    [F] = MMS_deformationGradient(xp_initial, t, NP,speedSound, h);

    for ip=1:NP
      stressExact(ip)  = .5 * E * (F(ip) - 1.0 / F(ip));
    end
  end

  %__________________________________
  %  Equation 49
  function [b] = MMS_accl_bodyForce(xp_initial, t, NP, speedSound,h)
     b = zeros(NP,1);

    [dp] = MMS_displacement(       xp_initial, t, NP, speedSound, h);
    [F]  = MMS_deformationGradient(xp_initial, t, NP, speedSound, h);

    c1 = speedSound * speedSound * pi * pi/ (h * h);
    for ip=1:NP
      b(ip)    = c1 * dp(ip) * (1.0 + 2.0/( F(ip) * F(ip)));
    end  

    % debugging  
    if(0)  
      subplot(4,1,1),plot(xp_initial, dp)
      ylabel('dp')
      subplot(4,1,2),plot(xp_initial, F)
      ylabel('F')
      subplot(4,1,3),plot(xp_initial, massP)
      ylabel('massP')
      subplot(4,1,4),plot(xp_initial, b)
      ylabel('b')
    end  
  end
  
  %__________________________________
  % plotResults
  function [L2_norm, maxError] = MMS_plotResults(titleStr, plotSwitch, plotInterval, xp_initial, OV, P, G)
     xpExact  = zeros(OV.NP,1);
      
    [dpExact]  = MMS_displacement(       xp_initial, OV.t, OV.NP, OV.speedSound, OV.bar_length);
    [velExact] = MMS_velocity(           xp_initial, OV.t, OV.NP, OV.speedSound, OV.bar_length);
    [FpExact]  = MMS_deformationGradient(xp_initial, OV.t, OV.NP, OV.speedSound, OV.bar_length);
    xpExact = xp_initial + dpExact;
    
    % compute L2Norm
    d = abs(P.dp - dpExact);
    
    L2_norm = sqrt( sum(d.^2)/length(dpExact) );
    maxError = max(d);
    
    if( (plotSwitch == 1) && (mod(OV.tstep,plotInterval) == 0))
      figure(2)                                
      set(2,'position',[1000,100,700,700]);    

      subplot(4,1,1),plot(P.xp, P.dp,'rd', P.xp, dpExact,'b');
      %axis([0 50 -10000 0])           
      ylim([-0.05 0.05]);         

      title(titleStr)                          
      legend('Simulation','Exact')             
      xlabel('Position');                      
      ylabel('Particle displacement');
      
      vel_G_exact = interp1(P.xp, velExact, G.nodePos);
      e = ones(size(G.nodePos));
      e(:) = 1e100;

      subplot(4,1,2),plot(P.xp, P.velP,'rd', P.xp, velExact,'b');
      % ylim([1.1*min(velP) 1.1*max(velP)])
      ylim([-20 20])
      ylabel('Particle Velocity'); 
      hold on
      errorbar(G.nodePos,vel_G_exact, e,'LineStyle','none','Color',[0.8314 0.8157 0.7843]);
      hold off

      subplot(4,1,3),plot(xp_initial,P.extForceP);
      ylim([-2e4 2e4])
      ylabel('externalForce acceleration');
      
      subplot(4,1,4),plot(P.xp, P.Fp,'rd',P.xp, FpExact,'b');
      ylim([0.5 1.5])
      ylabel('Fp');

      f_name = sprintf('%g.2.ppm',OV.tstep-1);
      F = getframe(gcf);
      [X,map] = frame2im(F);
      imwrite(X,f_name);
 %     input('hit return');
    end
  end
  
  
end
