#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>

#include "KGA.h"
#include "MakeTableFunction.h"

using namespace std;

void ToArray(int size, double *array, char *_argv){
 
  ifstream in(_argv); // open table
  if (!in){
    cout << " cannot open " << _argv << endl;
    exit(1);
  }
  
  for ( int i =0; i < size; i ++ )
      in >> array[i];
 
  in.close();
}



void RgggArrayTable(const int  &arraySize,
		    const double *array1,
		    const double *array2,
		    const double *array3,
		    char *_argv) {
  
  ofstream out(_argv);
  
  for ( int arrayCounter = 0; arrayCounter < arraySize; arrayCounter ++ ) {
    
    out << setw(16) << setprecision(5) << arrayCounter; // VolIndex
    out << setw(16) << setprecision(5) << array1[arrayCounter*4+1]; // temp
    out << setw(16) << setprecision(5) << array2[arrayCounter]; // temp    
    out << setw(16) << setprecision(5) << array3[arrayCounter]; // absorb coeff
    out << endl;

  }

  out.close();
}





int main(){

  double *kgp = new double[10];
  double *kgpzone2 = new double[8];
  double *kgpzone3 = new double[2];
  double *kgpzone4 = new double[3];
  double *kgVoltest = new double[10000];
  
  kgpzone2[0] = -5.24;  
  kgpzone2[1] = 5.426; 
  kgpzone2[2] = 4.531;
  kgpzone2[3] = -2.98;
  kgpzone2[4] = -110.1;
  kgpzone2[5] = 318.5;
  kgpzone2[6] = -335.5;
  kgpzone2[7] = 125.3;
  
  kgpzone3[0] = 0.8232;
  kgpzone3[1] = 598.1;

  kgpzone4[0] = 0.5197;
  kgpzone4[1] = 3.949e4;
  kgpzone4[2] = 0.7716;

//   kgp[0] = -5.637;
//   kgp[1] = 13.82;
//   kgp[2] = -47.49;
//   kgp[3] = 109.8;
//   kgp[4] = -123.3;
//   kgp[5] = 52.67;


  
  int gUnisize = 5000;
  
  double gExlb = 0.0455; // gExcluded lower bound
  double gExub = 0.995;  // gExcluded upper bound
  double g2ndlast = 0.99995;
  
  // when g < gExlb, k = pow(10.,klb); g > gExub, k = pow(10.,kub)
  double klb = -6.9957;
  klb = pow(10.,klb);

  // k(g = 0.999 ) = 0.59329; k(g=1 ) = 1.2923
  double kub = 1.2923;
  kub = pow(10.,kub);


  // seems pwr = 6 has the reasonable distribution for Rg and g
  // we want get accurate result for smaller g ( smaller Rg ), which require pwr to be small,
  // also, we want have enough data points at larger Rg, where g -> 1, which require pwr to be large
  
  double pwr = 1;
  
  KGA obKGA(gUnisize,pwr);
  MakeTableFunction obMakeTable;

  obKGA.get_RgDmixture(kgp,kgpzone2, kgpzone3, kgpzone4,
		       klb,kub,gExlb,gExub,g2ndlast);


  obMakeTable.twoArrayTable(gUnisize,
			    obKGA.gUni, obKGA.RgDmixture,
			    "gRpwr10");

  
  // given Rg to get g
  int ggNo = 51;
  double *gg = new double[ggNo];
  double *Rg = new double[ggNo];
  
  // In order to make the integration easier, set Rg acendingly 
  // Rg from 0 to 1
  
  double *kk = new double[ggNo]; // for volume elements

  double dg;
  
  dg = 1./ ( ggNo - 1);
  for ( int ig = 0; ig < ggNo; ig ++ ){
    Rg[ig] = ig * dg;    
    // when Rg is acending, gg is acending too
    gg[ig] = obKGA.get_gDmixture(Rg[ig]);

  }



  int zone1endi;
  int zone2starti, zone2endi;
  int zone3starti, zone3endi;
  int zone4starti;
  
  bool findi;
  
  int searchi;
  
  // zone2starti, zone1endi
  searchi = 0;
  findi = 0;
  do {
    if ( gg[searchi] >= gExlb ){
      zone2starti = searchi;
      zone1endi = searchi-1;
      findi = 1;
    }
    else
      searchi ++;
  }while(findi == 0 );


  // zone3endi, zone4starti
  findi = 0;
  searchi = ggNo -1;
  do {
    if ( gg[searchi] <= g2ndlast ){
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
    if ( gg[searchi] <= gExub ){
      zone2endi = searchi;
      zone3starti = searchi + 1;
      findi = 1;
    }
    else
      searchi --;
  }while(findi == 0);

  
   cout << "zone1endi = " << zone1endi << endl;
   cout << "zone2starti = " << zone2starti << "; zone2endi = " << zone2endi << endl;
   cout << "zone3starti = " << zone3starti << "; zone3endi = " << zone3endi << endl;
   cout << "zone4starti = " << zone4starti << endl;
   cout << "gg[zone1endi] = " << gg[zone1endi] << endl;
   cout << "gg[zone2starti] = " << gg[zone2starti] << endl;
   cout << "gg[zone2endi] = " << gg[zone2endi] << endl;
   cout << "gg[zone3starti] = " << gg[zone3starti] << endl;
   cout << "gg[zone3endi] = " << gg[zone3endi] << endl;
   cout << "gg[zone4starti] = " << gg[zone4starti] << endl;          

  int i;
  
  // Zone1 -- [0, gExlb] section
  for ( i = 0; i <= zone1endi; i ++ ){
    kk[i] = obKGA.get_kDmixtureZone1(klb);

  }

  
  // Zone2 -- [gExlb, gExub] section  
  for ( i = zone2starti; i <= zone2endi; i ++ ) {
    kk[i] = obKGA.get_kDmixtureZone2(kgpzone2,gg[i]);
 
  }


  // Zone3 -- [gExub, g2ndlast] section  
  for ( i = zone3starti; i <= zone3endi; i ++ ) {
    kk[i] = obKGA.get_kDmixtureZone3(kgpzone3,gg[i]);
 
  }
  
  
  // Zone4 -- [g2ndlast,1] section
  for ( i = zone4starti; i < ggNo; i ++ ){
    kk[i] = obKGA.get_kDmixtureZone4(kgpzone4,gg[i]);
 
  }

  


  
//   int starti, endi;
//   bool findi;
  
//   findi = 0;
//   starti = 0;
//   endi = ggNo-1;
  
//   // from gUni to get the range of i which uses the curve fitting function
//   do {
//     if ( gg[starti] >= gExlb ){ 
//       starti = starti;
//       findi = 1;
//     }
//     else
//       starti ++;
//   }while(findi == 0 );

  
//   findi = 0;
//   do {
//     if ( gg[endi] <= gExub ){ 
//       endi = endi + 1; // use endi +1 , thats the one who is greater than gExub
//       findi = 1;
//     }
//     else
//       endi --;
//   }while(findi == 0);

  
//   // [0, gExlb] section
//   for ( int i = 0; i < starti; i ++ ){
//     kk[i] = klb;

//   }

  
//   // [gExlb, gExub] section  
//   for ( int i = starti; i < endi; i ++ ) {
//     kk[i] = obKGA.get_kDmixture(kgp,gg[i]);

    
//   }


//   // [gExub,1] section
//   for ( int i = endi; i < ggNo; i ++ ){
//     kk[i] = kub;

//   }




  obMakeTable.threeArrayTable(ggNo,
			      Rg,gg,kk,"Rgggkkpwr10");

  double *testarray = new double[ggNo*6];
  for ( int i = 0; i < ggNo * 6; i ++ )
    testarray[i] = i;
  
  obMakeTable.coupleTwoArrayTable(ggNo, gg, testarray,
				  6,"testarraytable");

  double *kgaVoltest = new double[20000];
  
  // read in kgVoltest from table kgtestmixture
  //  ToArray(10000,kgVoltest,"kgtestmixture");
  
  ToArray(20000,kgaVoltest,"k_g3_ka3_a3");

  
  ofstream out("kgaReMakeTable");
  
  for ( int i = 0; i < 5000; i ++ ) {
    
    for ( int j = 0; j < 4; j ++ ) {
      out << setw(13) << setprecision(5) << kgaVoltest[i*4 +j];
      
    }

    out << endl;
  }

  
  out.close();


   obKGA.get_RgTDmixtureTable(kgaVoltest);
   RgggArrayTable(5000,kgaVoltest,obKGA.fracA,obKGA.RgDmixture,"ggFracRg5000_300");
   
  
  
  delete[] gg;
  delete[] Rg;
  delete[] kk;
  delete[] kgp;
  delete[] kgpzone2;
  delete[] kgpzone3;
  delete[] kgpzone4;
  delete[] testarray;
  delete[] kgVoltest;
  delete[] kgaVoltest;
  
  return 0;
  
}
