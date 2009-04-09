#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

// this code is to smooth out the same R's in R-g relations.
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

void singleArrayTable(const double *Array,
		      const int &ArraySize,
		      const int &No, char * _argv){
  ofstream out( _argv);

  for ( int i = 0; i < ArraySize; i ++ ) {    
    out << setw(16) << setprecision(14) << Array[i];
    if ( ( (i+1) % No ) == 0 )
      out << endl;
  }
  
   out.close();
}


int main(){
  //% get absc with wvnm, use wvnm to get CDF of R and wvnm
  //% need to calculate Ib_wvn at preset T.
  //% here T = 1000 for LBLHITEMPabcswvnm case
  //% watchout unit

  //% wvnm -- cm-1,  absc -- cm-1
  //% Ebeta -- W/m

  double T, C1, C2, Ebeta1, Ebeta2, Ibeta1, Ibeta2, pi;
  pi = atan(1) * 4;
  T = 1000;

  //%Ibeta = Ebeta/pi;
  //% Ebeta = C1 wvn^3 / ( exp(C2 * eta/T) - 1)
  //% C1 = 2*pi*h*c0^2
  //% C2 = h*c0/k
  C1 = 3.7418e-16;// % [W m^2]
  C2 = 1.4388; //%[cm K]


  //% LBLHITEMPabcswvnm(:,1) == wvnm
  //% LBLHITEMPabcswvnm(:,2) == absc
 
  int abcsSize, abcswvnmSize;
  abcsSize = 1495001; // the last nonzero length get the number which is huge
  abcswvnmSize = abcsSize * 2;
  double *abcswvnm = new double[abcswvnmSize];
  double *Rwvnabcs = new double[abcsSize];
  
  //  ToArray(abcswvnmSize,abcswvnm, "LBLabsc-wvnm-T1000Trad1000-CO201H2O02CO003.dat");
  ToArray(abcswvnmSize,abcswvnm, "LBLabsc-wvnm-T1500Trad1500-CO202H2O04CO006.dat");
  // do integration
  // direct FSK, stretching factor a = 1, no Ibeta needed
  // R = \frac{int_0^g k(g)}{int_0^1 k(g)}

  double sumR = 0;
 
  Ebeta1 = C1 * abcswvnm[0] *  abcswvnm[0] *  abcswvnm[0]  * 1e6 /
      ( exp( C2*  abcswvnm[0] / T )- 1);
  
  Ibeta1 = Ebeta1 / pi;

  for ( int i = 0; i < abcsSize-1; i ++) {
    
    Ebeta2 = C1 *
      abcswvnm[(i+1)*2] *  abcswvnm[(i+1)*2] *  abcswvnm[(i+1)*2] * 1e6 /
      ( exp( C2*  abcswvnm[(i+1)*2]  / T ) - 1);
    
    Ibeta2 = Ebeta2 / pi;

    // convert absc from cm-1 to m-1
    // convert wvnm from cm-1 to m-1
    // but this should not change Rwvnabcs
    sumR = sumR +
      (  abcswvnm[i * 2 + 1] * Ibeta1 + abcswvnm[(i+1)*2 + 1] * Ibeta2 ) * 100 * 
      ( abcswvnm[(i+1) * 2] -abcswvnm[(i)*2] ) * 100 /2;
    
    Ibeta1 = Ibeta2;
    
    Rwvnabcs[i] = sumR;
  }
  
  Rwvnabcs[abcsSize-1] = sumR;
  
  for ( int i = 0; i < abcsSize; i ++){
    Rwvnabcs[i] = Rwvnabcs[i] / sumR;
    //  cout << Rwvnabcs[i] << endl;
  }
  
  singleArrayTable(Rwvnabcs, abcsSize, 1, "RwvnabcsNosorting1500K-CO202H2O04CO006.dat");
  
  /*
  // sort Rwvnabcs to ascending order
  vector<double> RIter (Rwvnabcs, Rwvnabcs+abcsSize);
  vector<int>::iterator it;
  
  sort(RIter.begin(), RIter.end());
  
  singleArrayTable(Rwvnabcs, abcsSize, 1, "aftersortedRwvnabcs");

  // now searching for the same Rwvnabcs
  vector<int> sameIndex;
  bool sameRwvnabcs = false;
  double dR, counter;
  
  for ( int j = 0; j < abcsSize-1; j ++) {
    
    if ( Rwvnabcs[j] == Rwvnabcs[j+1] && (j != (abcsSize-2) ) ) { // same Rwvnabcs
      sameIndex.push_back(j);      
      sameRwvnabcs = true;
      
    }
    else if (Rwvnabcs[j] == Rwvnabcs[j+1] && (j+1) == (abcsSize-1)   ) { // the end of file
      sameIndex.push_back(j);
      sameIndex.push_back(j+1);
      dR = (Rwvnabcs[j+1] - Rwvnabcs[sameIndex[0]-1])/sameIndex.size();
      cout << "========================================" << endl;
      counter = 0;
      cout << "dR = " << dR << endl;
      cout << "size = " << sameIndex.size() << endl;
      for ( int k = 0; k < sameIndex.size(); k ++)
	cout << "sameIndex = " << sameIndex[k] << endl;
      
      for ( int i = sameIndex[0]; i < sameIndex[sameIndex.size()-1]+1 ; i ++){
	counter ++;
	cout << "counter = " << counter << endl;
	Rwvnabcs[i] =  Rwvnabcs[sameIndex[0]-1] + counter * dR;
	cout << "Rwvnabcs[ " << i << "] =" << 
	  Rwvnabcs[i] << endl;
	
      }
     
      sameRwvnabcs = false;
      sameIndex.clear();
      // exit(1);
      
    }
    else if (  Rwvnabcs[j] != Rwvnabcs[j+1] && sameRwvnabcs ) { // reset Rwvnabcs values
      sameIndex.push_back(j);    
      dR = (Rwvnabcs[j+1] - Rwvnabcs[j])/sameIndex.size();
      cout << "========================================" << endl;
      counter = 0;
      cout << "dR = " << dR << endl;
      cout << "size = " << sameIndex.size() << endl;
      for ( int k = 0; k < sameIndex.size(); k ++)
	cout << "sameIndex = " << sameIndex[k] << endl;
      
      for ( int i = sameIndex[0]; i < sameIndex[sameIndex.size()-1]+1 ; i ++){

	cout << "counter = " << counter << endl;
	Rwvnabcs[i] = Rwvnabcs[i] + counter * dR;
	cout << "Rwvnabcs[ " << i << "] =" << 
	         Rwvnabcs[i] << endl;
	counter ++;
	
      }
     
      sameRwvnabcs = false;
      sameIndex.clear();
      // exit(1); 

    } 
     else if (Rwvnabcs[j] != Rwvnabcs[j+1] ){
       sameRwvnabcs = false;
     }
    
  }

  singleArrayTable(Rwvnabcs, abcsSize, 1, "afterInterpolationRwvnabcs");
  
  return 0;
  */

  return 0;
}


