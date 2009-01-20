// for fsk algorithm, k-g-a
// for k-g, use 9th polynomial function to fit

#include "KGA.h"
#include "MakeTableFunction.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace std;


// when dont know the gNo in prior,
// declare double *point_name in header file
// the define point_name array in function when get the size
class MakeTableFunction;

KGA::KGA(const int &gNo_, const double &pwr){

  //gNo is the size of gUni array
  // should be an odd number
  gNo = gNo_;
  // cout << "gNo = " << gNo << endl;
  
  gUni = new double[gNo];
  RgDmixture = new double[gNo];
  
  // calculate k vs gUni
  kgUni = new double[gNo];

  // calcualte a vs gUni
  agUni = new double[gNo];

  // calculate Rg
  fracA = new double[gNo];

  
  pwrg = new double[gNo];

  // pwrg -- power g array ( with power pwr ) 
  // because k-g function increases dramatically on larger g
  // we generate more g at larger values, so to catch better fit for RgDmixture
  double pwrg_min, pwrg_max, pwrg_step;
  pwrg_min = 0; // g[0] = 0 always, so pwrg_min = g[0]^ pwr = 0
  pwrg_max = 1; // g[last] = 1 always, so pwrg_max = g[last]^pwr = 1
  pwrg_step = (pwrg_max - pwrg_min)/(gNo-1);
  
  //dg = 1./ (gNo -1 );
  
  //cout << "dg = " << dg << endl;
  pwrg[0] = 0;
  gUni[0] = 0;
  
  for ( int i = 1; i < gNo; i ++ ){
    pwrg[i] = (i-1) * pwrg_step + pwrg_min;
    gUni[i] = pow(pwrg[i], 1./pwr);
    
    // cout << "gUni[ " << i << "]= " << gUni[i] << endl;

  }
  
  
};



KGA::~KGA(){
  delete[] gUni;
  delete[] RgDmixture;
  delete[] kgUni;
  delete[] agUni;
  delete[] fracA;  
  delete[] pwrg;
}





// use mixture's lgk-g curve fitting function to get k for gUni
// only for directly mixture
// then calculate Rg and save Rg
// Dmixture -- data are generated directly from gas mixture
// generate RgDmixture vs gUni table
void KGA::get_RgDmixture(const double *kgp,const double *kgpzone2,
			 const double *kgpzone3, const double *kgpzone4,
			 const double &klb, const double &kub,
			 const double &gExlb, const double &gExub,
			 const double &g2ndlast){
			 

  double fracSum;
  int zone1endi;
  int zone2starti, zone2endi;
  int zone3starti, zone3endi;
  int zone4starti;
  int searchi;
  
  bool findi;
  
  fracSum = 0;

  // zone2starti, zone1endi
  searchi = 0;
  findi = 0;
  do {
    if ( gUni[searchi] >= gExlb ){
      zone2starti = searchi;
      zone1endi = searchi-1;
      findi = 1;
    }
    else
      searchi ++;
  }while(findi == 0 );


  // zone3endi, zone4starti
  findi = 0;
  searchi = gNo -1;
  do {
    if ( gUni[searchi] <= g2ndlast ){
      zone3endi = searchi;
      zone4starti = searchi + 1;
      findi = 1;
    }
    else
      searchi --;
  }while(findi == 0);
  

  // zone3starti, zone2endi
  findi = 0;
  do {
    if ( gUni[searchi] <= gExub ){
      zone2endi = searchi;
      zone3starti = searchi + 1;
      findi = 1;
    }
    else
      searchi --;
  }while(findi == 0);

  
//   cout << "zone1endi = " << zone1endi << endl;
//   cout << "zone2starti = " << zone2starti << "; zone2endi = " << zone2endi << endl;
//   cout << "zone3starti = " << zone3starti << "; zone3endi = " << zone3endi << endl;
//   cout << "zone4starti = " << zone4starti << endl;
//   cout << "gUni[zone1endi] = " << gUni[zone1endi] << endl;
//   cout << "gUni[zone2starti] = " << gUni[zone2starti] << endl;
//   cout << "gUni[zone2endi] = " << gUni[zone2endi] << endl;
//   cout << "gUni[zone3starti] = " << gUni[zone3starti] << endl;
//   cout << "gUni[zone3endi] = " << gUni[zone3endi] << endl;
//   cout << "gUni[zone4starti] = " << gUni[zone4starti] << endl;          

  int i;
  
  // Zone1 -- [0, gExlb] section
  for ( i = 0; i <= zone1endi; i ++ ){
    kgUni[i] = get_kDmixtureZone1(klb);
    agUni[i] = 1; // for T0 = T
    fracA[i] = 0;
  }

  
  // Zone2 -- [gExlb, gExub] section  
  for ( i = zone2starti; i <= zone2endi; i ++ ) {
    kgUni[i] = get_kDmixtureZone2(kgpzone2,gUni[i]);
    agUni[i] = 1; // for T0 = T 
    //    agUni[i] = get_aDmixture(kgp,gUni[i]);
    fracA[i] = 0;  
  }


  // Zone3 -- [gExub, g2ndlast] section  
  for ( i = zone3starti; i <= zone3endi; i ++ ) {
    kgUni[i] = get_kDmixtureZone3(kgpzone3,gUni[i]);
    agUni[i] = 1; // for T0 = T 
    //    agUni[i] = get_aDmixture(kgp,gUni[i]);
    fracA[i] = 0;  
  }
  
  
  // Zone4 -- [g2ndlast,1] section
  for ( i = zone4starti; i < gNo; i ++ ){
    kgUni[i] = get_kDmixtureZone4(kgpzone4,gUni[i]);
    agUni[i] = 1; // for T0 = T
    fracA[i] = 0;
  }

  

  // calculate Rg = fracA/fracSum

  for ( int i = 1; i < gNo; i ++ ) 
    fracA[i] = fracA[i-1] +
      (kgUni[i-1] * agUni[i-1] + kgUni[i] * agUni[i]  ) *
      (gUni[i] - gUni[i-1])/2.;
    
  fracSum = fracA[gNo-1];


  // RgDmixture is corresponding to gUni
  for ( int i = 0; i < gNo; i ++ ){
    RgDmixture[i] = fracA[i] / fracSum;
    // cout << "RgDmixture[ " << i << "] = " << RgDmixture[i] << endl;
  }

  

}






// for direct mixture-- RgDmixture[gNo] vs gUni[gNo]
// given Rg, to get gDmixture

double KGA::get_gDmixture(const double &Rg){
			 
  
  // interpolate from RgDmixture
  // RgDmixture[0] = 0

  int i = 0;
  bool notGetg = 1;
  double gDmixture;
  
  do {
        
    if ( RgDmixture[i+1] > Rg ){
      
      gDmixture = gUni[i] + ( gUni[i+1] - gUni[i] )* ( Rg - RgDmixture[i] )
	/ ( RgDmixture[i+1] - RgDmixture[i] );
      
      notGetg = 0;
      
    }
    else 
      i ++;
    
    
  } while( i < gNo && notGetg );


  // Rg is greater than all RgDmixture
  if ( notGetg == 1 )
    gDmixture = 1;
  
  return gDmixture;

}






// for direct mixture, given gg , get k from lgk-g function
double KGA::get_kDmixture(const double *kgp,
			  const double &gg) {

  double kDmixture;
  int powerp1; // the power + 1
  
//   // 9th poly
//   powerth = 10;
//   double *ggPower = new double[powerp1];

//   ggPower[0] = 1;
//   ggPower[1] = gg;

//   for ( int i = 2; i < powerp1; i ++ )
//     ggPower[i] = ggPower[i-1] * gg;
  
//   kDmixture = pow(10.,(kgp[0] + kgp[1]*ggPower[1] + kgp[2]*ggPower[2] + kgp[3]*ggPower[3] +
// 		      kgp[4]*ggPower[4] + kgp[5]*ggPower[5] + kgp[6]*ggPower[6] +
// 		      kgp[7]*ggPower[7] + kgp[8]*ggPower[8] + kgp[9]*ggPower[9] ) );


  // curve fitting for after excluded data ( exclude both ends)
  // 5th poly
  
  powerp1 = 6;  
  double *ggPower = new double[powerp1];

  ggPower[0] = 1;
  ggPower[1] = gg;

  for ( int i = 2; i < powerp1; i ++ )
    ggPower[i] = ggPower[i-1] * gg;
  
   kDmixture = pow(10.,(kgp[0] + kgp[1]*ggPower[1] + kgp[2]*ggPower[2] + kgp[3]*ggPower[3] +
  		       kgp[4]*ggPower[4] + kgp[5]*ggPower[5] + kgp[6]*ggPower[6] )) ;



  
  delete[] ggPower;
  
  return kDmixture;
  
}


// temporarily for 20%H2O, 10%CO2, 3%CO and direct mixture
// lgk original full bound [-6.9957, 1.2923]
// g0 original full bound [3e-5, 1]

// there are 4 zones
// g0 [0, 0.0455] -- zone1 ; lgk [-6.9957, -4.9957] 
// g0 [0.0455, 0.995] -- zone2; lgk [ -4.9957, -0.00086946]
// g0 [0.995,0.99995] -- zone3; lgk [-0.00086946, 0.82856]
// g0 [0.99995,1] -- zone4; lgk [ 0.82856, 1.2923]

// for direct mixture, given gg , get k from lgk-g function
// for zone1, no function curve fitting , just plug in the klb
double KGA::get_kDmixtureZone1(const double &klb) {

  double kDmixtureZone1;
  
  kDmixtureZone1 = klb;
  //kDmixtureZone1 = 1;
  return kDmixtureZone1;
  
}




// Zone2
// g0 [0.0455, 0.995] -- zone2; lgk [ -4.9957, -0.00086946]
double KGA::get_kDmixtureZone2(const double *kgpzone2,
			       const double &gg) {

  double kDmixtureZone2;
  int powerp1; // the power +1

  // curve fitting for after excluded data ( exclude both ends)
  // 7th poly
  
  powerp1 = 8;  
  double *ggPower = new double[powerp1];

  ggPower[0] = 1;
  ggPower[1] = gg;

  for ( int i = 2; i < powerp1; i ++ )
    ggPower[i] = ggPower[i-1] * gg;

  
  kDmixtureZone2 = pow(10.,(kgpzone2[0] + kgpzone2[1]*ggPower[1] + kgpzone2[2]*ggPower[2] +
		       kgpzone2[3]*ggPower[3] +  kgpzone2[4]*ggPower[4] +
		       kgpzone2[5]*ggPower[5] + kgpzone2[6]*ggPower[6] +
		       kgpzone2[7]*ggPower[7] + kgpzone2[8]*ggPower[8])) ;

  
  //kDmixtureZone2 = 1;
  
  delete[] ggPower;
  
  return kDmixtureZone2;
  
}



//Zone3
// g0 [0.995,0.99995] -- zone3; lgk [-0.00086946, 0.82856]
double KGA::get_kDmixtureZone3(const double *kgpzone3,
			       const double &gg) {

  double kDmixtureZone3;

  kDmixtureZone3 = pow(10., kgpzone3[0] * pow(gg,kgpzone3[1]) );
  //kDmixtureZone3 = 1;
  return kDmixtureZone3;
  
}




//Zone4
// g0 [0.99995,1] -- zone4; lgk [ 0.82856, 1.2923]
double KGA::get_kDmixtureZone4(const double *kgpzone4,
			       const double &gg) {

  double kDmixtureZone4;
  
  //kDmixtureZone4 = 1;
  kDmixtureZone4 = pow(10., kgpzone4[0] * pow(gg,kgpzone4[1]) + kgpzone4[2]);
  
  return kDmixtureZone4;
  
}



double KGA::get_aDmixture(const double *kgp, const double &gg) {


  double aDmixture;

  return aDmixture;
  
}



// Remember other Dmixture's k needs to be multiply by 100


void KGA::get_RgDmixtureTable(const double *kgVoltest){
			 
  // fracA[5000]
  
  double fracSum;

  fracSum = 0;
     
  // calculate Rg = fracA/fracSum

  for ( int i = 0; i < 5000; i ++ )
    fracA[i] = 0;
  
  for ( int i = 1; i < 5000; i ++ )    
    fracA[i] = fracA[i-1] +
      (kgVoltest[(i-1)*2]  + kgVoltest[i*2]  ) *
      (kgVoltest[i*2+1] - kgVoltest[(i-1)*2+1])/2.;
    
  fracSum = fracA[gNo-1];


  // RgDmixture is corresponding to gUni
  for ( int i = 0; i < 5000; i ++ ){
    RgDmixture[i] = fracA[i] / fracSum;
    // cout << "RgDmixture[ " << i << "] = " << RgDmixture[i] << endl;
  }

  

}








double KGA::get_gDmixtureTable(const double &Rg, const double *kgVoltest){
			 
  
  // interpolate from RgDmixture
  // RgDmixture[0] = 0

  int i = 0;
  bool notGetg = 1;
  double gDmixture;
  
  do {
        
    if ( RgDmixture[i+1] > Rg ){
      
      gDmixture = kgVoltest[i*2+1] + ( kgVoltest[(i+1)*2+1] - kgVoltest[i*2+1] )*
	( Rg - RgDmixture[i] )/ ( RgDmixture[i+1] - RgDmixture[i] );
      
      notGetg = 0;
      
    }
    else 
      i ++;
    
    
  } while( i < 5000 && notGetg );


  // Rg is greater than all RgDmixture
  if ( notGetg == 1 )
    gDmixture = 1;
  
  return gDmixture;

}





double KGA::get_kDmixtureTable(const double &gg,
			       const int &ggNo,
			       const double *kgVoltest){
			 
  
  // interpolate from RgDmixture
  // RgDmixture[0] = 0

  int i = 0;
  bool notGetg = 1;
  double kDmixture;
  
  do {
        
    if ( kgVoltest[(i+1)*2+1] >= gg ){

      if ( kgVoltest[(i+1)*2+1] > gg)
	kDmixture = kgVoltest[i*2] + ( kgVoltest[(i+1)*2] - kgVoltest[i*2])*
	  ( gg - kgVoltest[i*2+1])/(kgVoltest[(i+1)*2+1] - kgVoltest[i*2+1]);
      else {
// 	cout << " we are equal " << "; gg = " << gg <<
// 	  "; kgVoltest[(i+1)*2+1] = " << kgVoltest[(i+1)*2+1] << endl;
	  
	kDmixture = kgVoltest[(i+1)*2];
      }
      
      
      notGetg = 0;
      
    }
    else 
      i ++;
    
    
  } while( i < ggNo && notGetg );


  // Rg is greater than all RgDmixture
  if ( notGetg == 1 ){
    cout << " wrong with kDmixtureTable" << endl;
    exit(1);
  }
  
  return kDmixture;

}




// when T != T0

//k_g_a_k*a
double KGA::get_gTDmixtureTable(const double &Rg, const double *kgaVoltest){
			 
  
  // interpolate from RgDmixture
  // RgDmixture[0] = 0

  int i = 0;
  bool notGetg = 1;
  double gDmixture;
  
  do {
        
    if ( RgDmixture[i+1] > Rg ){
      
      gDmixture = kgaVoltest[i*4+1] + ( kgaVoltest[(i+1)*4+1] - kgaVoltest[i*4+1] )*
	( Rg - RgDmixture[i] )/ ( RgDmixture[i+1] - RgDmixture[i] );
      
      notGetg = 0;
      
    }
    else 
      i ++;
    
    
  } while( i < 5000 && notGetg );


  // Rg is greater than all RgDmixture
  if ( notGetg == 1 )
    gDmixture = 1;
  
  return gDmixture;

}



//k_g_a_k*a
double KGA::get_kTDmixtureTable(const double &gg,
			       const int &ggNo,
			       const double *kgaVoltest){
			 
  
  // interpolate from RgDmixture
  // RgDmixture[0] = 0

  int i = 0;
  bool notGetg = 1;
  double kDmixture;
  
  do {
        
    if ( kgaVoltest[(i+1)*4+1] >= gg ){

      if ( kgaVoltest[(i+1)*4+1] > gg)
	kDmixture =  kgaVoltest[i*4] + ( kgaVoltest[(i+1)*4] - kgaVoltest[i*4])*
	  ( gg - kgaVoltest[i*4+1])/(kgaVoltest[(i+1)*4+1] - kgaVoltest[i*4+1]);
      else {

	kDmixture = kgaVoltest[(i+1)*4] ;
      }
      
      
      notGetg = 0;
      
    }
    else 
      i ++;
    
    
  } while( i < ggNo && notGetg );


  // Rg is greater than all RgDmixture
  if ( notGetg == 1 ){
    cout << " wrong with kDmixtureTable" << endl;
    exit(1);
  }
  
  return kDmixture;

}



// kga is k_g3_a_k*a3 , still directly from mixture, but T(1000K) != T0 (300K)
void KGA::get_RgTDmixtureTable(const double *kgaVoltest){
			 
  // fracA[5000]
  
  double fracSum;

  fracSum = 0;
     
  // calculate Rg = fracA/fracSum

  for ( int i = 0; i < 5000; i ++ )
    fracA[i] = 0;
  
  for ( int i = 1; i < 5000; i ++ )    
    fracA[i] = fracA[i-1] +
      (kgaVoltest[(i-1)*4+3] + kgaVoltest[i*4+3]  ) * 
      (kgaVoltest[i*4+1] - kgaVoltest[(i-1)*4+1])/2.;
    
  fracSum = fracA[5000-1];


  // RgDmixture is corresponding to gUni
  for ( int i = 0; i < 5000; i ++ ){
    RgDmixture[i] = fracA[i] / fracSum;
    // cout << "RgDmixture[ " << i << "] = " << RgDmixture[i] << endl;
  }

  

}




/*
// kga is k_g3_a_k*a3 , still directly from mixture, but T(1000K) != T0 (300K)
void KGA::get_RgTDmixtureTable(const double *kgaVoltest){
			 
  // fracA[5000]
  
  double fracSum;

  fracSum = 0;
     
  // calculate Rg = fracA/fracSum

  for ( int i = 0; i < 5000; i ++ )
    fracA[i] = 0;
  
  for ( int i = 1; i < 5000; i ++ )    
    fracA[i] = fracA[i-1] +
      (kgaVoltest[(i-1)*4+3] + kgaVoltest[i*4+3]  ) * 
      (kgaVoltest[i*4+1] - kgaVoltest[(i-1)*4+1])/2.;
    
  fracSum = fracA[5000-1];


  // RgDmixture is corresponding to gUni
  for ( int i = 0; i < 5000; i ++ ){
    RgDmixture[i] = fracA[i] / fracSum;
    // cout << "RgDmixture[ " << i << "] = " << RgDmixture[i] << endl;
  }

}



*/




// k_g3_a3_k*a3
double KGA::get_aTDmixtureTable(const double &gg,
			       const int &ggNo,
			       const double *kgaVoltest) {


  double aDmixture;

  int i = 0;
  bool notGetg = 1;
  
  do {
        
    if ( kgaVoltest[(i+1)*4+1] >= gg ){

      if ( kgaVoltest[(i+1)*4+1] > gg)
	
	aDmixture = kgaVoltest[i*4+2] + ( kgaVoltest[(i+1)*4+2] - kgaVoltest[i*4+2])*
	  ( gg - kgaVoltest[i*4+1])/(kgaVoltest[(i+1)*4+1] - kgaVoltest[i*4+1]);
      
      else {
	
	aDmixture = kgaVoltest[(i+1)*4+2];
      }
      
      
      notGetg = 0;
      
    }
    else 
      i ++;
    
    
  } while( i < ggNo && notGetg );


  // Rg is greater than all RgDmixture
  if ( notGetg == 1 ){
    cout << " wrong with aDmixtureTable" << endl;
    exit(1);
  }
  
  return aDmixture;
  
}







