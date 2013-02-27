   // ====== Case 3  liu non-isothermal, uniform mixture in Chapter 3 =====

   // Composition field
// should I declare these in driver?

   // no soot in this case
   for ( int i = 0; i < VolElementNo; i ++ ) {
     CO2[i] = 0.1;
     H2O[i] = 0.2;
     SFV[i] = 0;
   }
   
   
   // Gas Temperature field
   double Te;
   double *Tc = new double[Npz];
   double R = 1;
   double xavg, yavg, Tcavg;
   double r, rR; // rR = r/R
   
   // Tc -- the centerline temperature
   // Tc(z =0 ) = 400K; Tc( z=4) = 800K, Tc(z=0.375) = 1800
   // Tc increases linearly from 400K to 1800K ( z = 0.375)
   //then decreases linearly to 800K at exit
   // T = (Tc - Te)f(r/R) + Te
   // f(r/R) = 1 - 3* (r/R)^2 + 2* (r/R)^3
   // R = 1m
   // T outside the circular zone is uniform at 800K
   
   // T = (1800-400)/0.375 * z + 400, if  0 <= z <= 0.375
   // T = -1000/3.625 * z + 6900/3.625,  if  0.375 <= z <= 4
   
   Te = 800; // Te -- the temperature at the exit Z = 4m , 800K   
   int Changei = int(0.375/dzconst);

// in my case, Z starts from bottom to top, centered at Z = 2m
   for ( int i = 0; i < Changei+1; i++ )
     Tc[i] = (1800-400)/0.375 * (Z[i] + 2) + 400;

for ( int i = Changei+1; i < Npz; i ++ )
     Tc[i] = (-1000 * (Z[i]+2) + 6900)/3.625;

   //  obTable.singleArrayTable(Tc, Npz, 1, "Tctable");

   for ( int i = 0; i < Ncz; i ++ ){
     for ( int j = 0; j < Ncy; j ++ ){
       for ( int k = 0; k < Ncx; k ++ ){
	 xavg = ( X[k] + X[k+1] )/2;
	 yavg = ( Y[j] + Y[j+1] )/2;
	 Tcavg = ( Tc[i] + Tc[i+1] )/2;
	 r = sqrt( xavg * xavg + yavg * yavg ) ;
	 rR = r / R;
	 
	 if ( rR <= 1 ) { // within the circular zone
	   T_Vol[i*Ncx*Ncy+j*Ncx+k] = (Tcavg - Te) *
	     ( 1- 3 * rR* rR + 2 * rR * rR * rR) + Te;
	 }else
	   T_Vol[i*Ncx*Ncy+j*Ncx+k] = 800;
       }
     }
   }

//  obTable.singleArrayTable(T_Vol,VolElementNo,1,"TVolTable");
   for ( int i = 0 ; i < VolElementNo ; i ++ )
     scatter_Vol[i] = 0;
   

	 
   // all surfaces are at 300K, cold black.

  // top bottom surfaces
  for ( int i = 0; i < TopBottomNo; i ++ ) {
    rs_surface[TOP][i] = 0;
    rs_surface[BOTTOM][i] = 0;

    rd_surface[TOP][i] = 0;
    rd_surface[BOTTOM][i] = 0;
    
    alpha_surface[TOP][i] = 1 - rs_surface[TOP][i] - rd_surface[TOP][i];
    alpha_surface[BOTTOM][i] = 1 - rs_surface[BOTTOM][i] - rd_surface[BOTTOM][i];
        
    emiss_surface[TOP][i] = alpha_surface[TOP][i];
    emiss_surface[BOTTOM][i] = alpha_surface[BOTTOM][i];

    T_surface[TOP][i] = 300;
    T_surface[BOTTOM][i] = 300;

  }


  // front back surfaces
  for ( int i = 0; i < FrontBackNo; i ++ ) {
    rs_surface[FRONT][i] = 0;
    rs_surface[BACK][i] = 0;

    rd_surface[FRONT][i] = 0;
    rd_surface[BACK][i] = 0;
    
    alpha_surface[FRONT][i] = 1 - rs_surface[FRONT][i] - rd_surface[FRONT][i];
    alpha_surface[BACK][i] = 1 - rs_surface[BACK][i] - rd_surface[BACK][i];
        
    emiss_surface[FRONT][i] = alpha_surface[FRONT][i];
    emiss_surface[BACK][i] = alpha_surface[BACK][i];

    T_surface[FRONT][i] = 300;
    T_surface[BACK][i] = 300;

  }

  
  // from left right surfaces
  for ( int i = 0; i < LeftRightNo; i ++ ) {
    rs_surface[LEFT][i] = 0;
    rs_surface[RIGHT][i] = 0;

    rd_surface[LEFT][i] = 0;
    rd_surface[RIGHT][i] = 0;
    
    alpha_surface[LEFT][i] = 1 - rs_surface[LEFT][i] - rd_surface[LEFT][i];
    alpha_surface[RIGHT][i] = 1 - rs_surface[RIGHT][i] - rd_surface[RIGHT][i];
        
    emiss_surface[LEFT][i] = alpha_surface[LEFT][i];
    emiss_surface[RIGHT][i] = alpha_surface[RIGHT][i];

    T_surface[LEFT][i] = 300;
    T_surface[RIGHT][i] = 300;

  }

 
   // ============== end of case 3 in chapter 3 =============== 
     

