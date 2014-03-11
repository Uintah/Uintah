#ifndef RadWsgg_H
#define RadWsgg_H

#include <cmath>

using namespace std;

class RadWsgg {

public:
  
  RadWsgg();
  ~RadWsgg();

  inline
  void WsggkVolwEmiss(const double *CO2,
		      const double *H2O,
		      const int &bands,
		      const double *T_Vol,
		      const double *SFV,
		      const int &VolElementNo,
		      double *kl_Vol, double *a_Vol){
    	if (bands == 1){

	  a = 0.4201;
	  b[0] = 6.508;
	  b[1] = -5.551;
	  b[2] = 3.029;
	  b[3] = -5.353;
	}
	else if (bands == 2){
	  
	  a = 6.516;
	  b[0] = -0.2504;
	  b[1] = 6.112;
	  b[2] = -3.882;
	  b[3] = 6.528;
	}
	else if (bands == 3){

	  a = 131.9; 
	  b[0] = 2.718;
	  b[1] = -3.118;
	  b[2] = 1.221;
	  b[3] = -1.612;
	}	
	else if (bands == 4){ 
            
	  // see the Smith's paper, for the clear gas, i = 0, weighting factor sum up to unity.           
	  a = 0.0; 
	  b[0] = 8.9756;
	  b[1] = -2.557;
	  b[2] = 0.368;
	  b[3] = -0.437;
	}
	

	for ( int i = 0; i < VolElementNo; i ++ ){
	  
	  // Use the data and model in Gautham's thesis (1.17) for Const = 402.834 -- Paula
	  
	  kl_Vol[i] = a * (CO2[i] + H2O[i]) + 402.834 * SFV[i] * T_Vol[i];
	  
	  a_Vol[i] = ( b[0]*1e-1) +
	    (b[1]*1e-4) * T_Vol[i] +
	    (b[2]*1e-7) * T_Vol[i] * T_Vol[i] +
	    (b[3]*1e-11) * T_Vol[i] * T_Vol[i] * T_Vol[i];

	  if (bands == 4) {
	    a_Vol[i] = 1.0 - a_Vol[i];
	  }

	}

}

  inline
   void WsggwEmissSurface(const int &surfaceFlag,
			  const int &elementNo,
			  const double * const T_surface[],
			  const int &bands,
			  double *a_surface[]){
     // OK, so this is the three gray gases band with one clear gas, band == 4 -- Paula
  // for Pwater/Pco2 = 2
  
	if (bands == 1){

	  a = 0.4201;
	  b[0] = 6.508;
	  b[1] = -5.551;
	  b[2] = 3.029;
	  b[3] = -5.353;
	}
	else if (bands == 2){
	  
	  a = 6.516;
	  b[0] = -0.2504;
	  b[1] = 6.112;
	  b[2] = -3.882;
	  b[3] = 6.528;
	}
	else if (bands == 3){

	  a = 131.9; 
	  b[0] = 2.718;
	  b[1] = -3.118;
	  b[2] = 1.221;
	  b[3] = -1.612;
	}	
	else if (bands == 4){ 

	  // due to   a_surface[surfaceFlag][4] = 1 - sum(a_surface[surfaceFlag][i]);
	  // b for bands = 4, are the summation of the three bands.
	  // see the Smith's paper, for the clear gas, i = 0, weighting factor sum up to unity.           
	  a = 0.0; 
	  b[0] = 8.9756; // = b[0]_band1 + b[0]_band2 + b[0]_band3
	  b[1] = -2.557;
	  b[2] = 0.368;
	  b[3] = -0.437;
	}

	
	//     Emissivity curve-fit stuff for wall bc's
	for ( int i = 0; i < elementNo; i ++ ){
	  
	  a_surface[surfaceFlag][i] = ( b[0]*1e-1) +
	    (b[1]*1e-4) * T_surface[surfaceFlag][i] +
	    (b[2]*1e-7) * T_surface[surfaceFlag][i] * T_surface[surfaceFlag][i] +
	    (b[3]*1e-11) * T_surface[surfaceFlag][i] * T_surface[surfaceFlag][i] * T_surface[surfaceFlag][i];
	  
	  // band 4 is the band 0 in Smith's paper
	  
	  if (bands == 4)
	    a_surface[surfaceFlag][i] = 1 - a_surface[surfaceFlag][i];
	}




	/*
	//     Absorptivity curve-fit stuff

// i compared the a_surface calculated this way and the way shown previous,
// they are about 100 magnitude different.

  //The data are for Pw/Pc = 2; a is the ki in Smith's paper; band 4 is band 0      
  // DATA c(1,1,1)/0.59324d0/,c(1,1,2)/-0.61741d-03/,
//      &c(1,1,3)/0.29248d-06/,c(1,1,4)/-0.45823d-10/,
//      &c(1,2,1)/0.35739d-03/,c(1,2,2)/0.22122d-06/,
//      &c(1,2,3)/-0.26380d-09/,c(1,2,4)/0.45951d-13/,
//      &c(1,3,1)/-0.71313d-06/,c(1,3,2)/0.46181d-09/,
//      &c(1,3,3)/-0.70858d-13/,c(1,3,4)/0.38038d-17/,
//      &c(1,4,1)/0.17806d-09/,c(1,4,2)/-0.11654d-12/,
//      &c(1,4,3)/0.19939d-16/,c(1,4,4)/-0.13486d-20/,
//      &c(2,1,1)/-0.35664d-01/,c(2,1,2)/0.21502d-03/,
//      &c(2,1,3)/-0.13648d-06/,c(2,1,4)/0.24284d-10/,
//      &c(2,2,1)/0.51605d-03/,c(2,2,2)/-0.70037d-06/,
//      &c(2,2,3)/0.38680d-09/,c(2,2,4)/0.70429d-13/,
//      &c(2,3,1)/0.12245d-06/,c(2,3,2)/0.99434d-10/,
//      &c(2,3,3)/-0.15598d-12/,c(2,3,4)/0.37664d-16/,
//      &c(2,4,1)/-0.57563d-10/,c(2,4,2)/-0.10109d-13/,
//      &c(2,4,3)/0.35273d-16/,c(2,4,4)/-0.89872d-20/,
//      &c(3,1,1)/0.12951d-00/,c(3,1,2)/0.54520d-04/,
//      &c(3,1,3)/-0.80049d-07/,c(3,1,4)/0.17813d-10/,
//      &c(3,2,1)/0.15210d-03/,c(3,2,2)/-0.37750d-06/,
//      &c(3,2,3)/0.21019d-09/,c(3,2,4)/-0.36011d-13/,
//      &c(3,3,1)/-0.13165d-06/,c(3,3,2)/0.20719d-09/,
//      &c(3,3,3)/-0.96720d-13/,c(3,3,4)/0.14807d-16/,
//      &c(3,4,1)/0.26872d-10/,c(3,4,2)/-0.34803d-13/,
//      &c(3,4,3)/0.14336d-16/,c(3,4,4)/-0.19754d-20/

	const double c[48] = {
	  0.59324, -0.61741e-1, 0.29248e-6, -0.45823e-10, // i = 1, j = 1
	  0.35739e-3, 0.22122e-6, -0.26380e-9, 0.45951e-13, // i = 1, j = 2
	  -0.71313e-6, 0.46181e-9, -0.70858e-13, 0.38038e-17, // i = 1, j = 3
	  0.17806e-9, -0.11654e-12, 0.19939e-16, -0.13486e-20, // i = 1, j = 4
	  -0.35664e-1, 0.21502e-3, -0.13648e-6, 0.24284e-10, // i = 2, j = 1
	  0.51605e-3, -0.70037e-6, 0.38680e-9, 0.70429e-13, // i = 2, j = 2
	  0.12245e-6, 0.99434e-10, -0.15598e-12, 0.37664e-16, // i = 2, j = 3
	  -0.57563e-10, -0.10109e-13, 0.35273e-16, -0.89872e-20, // i = 2, j = 4
	  0.12951, 0.54520e-4, -0.80049e-7, 0.17813e-10, // i = 3, j = 1
	  0.15210e-3, -0.37750e-6, 0.21019e-9, -0.36011e-13, // i = 3, j = 2
	  -0.13165e-6, 0.20719e-9, -0.96720e-13, 0.14807e-16, // i = 3, j = 3
	  0.26872e-10, -0.34803e-13, 0.14336e-16, -0.19754e-20} // i = 3, j = 4

	  
	for ( int i = 0; i < surfaceElementNo; i ++ ) {
	  
	  for( int j = 0; j < 4; j ++ ){
	    
	    absSum = 0;
	    
	    for ( int k = 0; k < 4; k ++ ) {
	      
	      absSum = absSum + c[bands*4*4 + j*4 + k] * pow(T_surface[i],k);
	    }
	    
	    wAbsorb_surface[i] = wAbsorb_surface[i] +
	      absSum * pow(800, j);

	  }

	}
	
	  
	if ( bands == 4 ) {
	  
	for ( int i = 0; i < surfaceElementNo; i ++ ) {

	  for ( int iband = 0; iband < 3; iband ++ ) {
	      
	    for( int j = 0; j < 4; j ++ ){
	      
	      absSum = 0;
	      
	      for ( int k = 0; k < 4; k ++ ) {
		
		absSum = absSum + c[iband*j + k] * pow(T_surface[i],k);
	      }
	      
	      wAbsorb_surface[i] = wAbsorb_surface[i] +
		absSum * pow(800, j);
	      
	    }
	    
	    wAbsorb_surface[i] = 1 - wAbsorb_surface[i];
	    
	  }
	  
	}

	  
	*/


	
}
   
 
private:
  double a;
  double b[4];
    
};

#endif
