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

  int gkSize, gSize;
  gSize = 5000;
  gkSize = gSize * 2;
  double *gk = new double[gkSize];
  double *Rkg = new double[gSize];
  
  ToArray(gkSize,gk, "HITEMPoldLBLkgT1000Trad1000-CO201H2O02CO003.dat");

  // do integration
  // direct FSK, stretching factor a = 1
  // R = \frac{int_0^g k(g)}{int_0^1 k(g)}

  double sumR = 0;
  double smallNo = 1e-8;
  
  for ( int i = 0; i < gSize-1; i ++) {

    sumR = sumR +
      (  gk[i * 2 + 1] + gk[(i+1)*2 + 1] ) *
      ( gk[(i+1) * 2] -gk[(i)*2] ) /2;

    Rkg[i] = sumR;
  }

  Rkg[gSize-1] = sumR;
  
  for ( int i = 0; i < gSize; i ++){
    Rkg[i] = Rkg[i] / sumR;
    //  cout << Rkg[i] << endl;
  }

  // sort Rkg to ascending order
  vector<double> RIter (Rkg, Rkg+gSize);
  vector<int>::iterator it;
  
  sort(RIter.begin(), RIter.end());
  
  singleArrayTable(Rkg, gSize, 1, "aftersortedRkg");

  // now searching for the same Rkg
  vector<int> sameIndex;
  bool sameRkg = false;
  double dR, counter;
  
  for ( int j = 0; j < gSize-1; j ++) {
    
    if ( Rkg[j] == Rkg[j+1] && (j != (gSize-2) ) ) { // same Rkg
      sameIndex.push_back(j);      
      sameRkg = true;
      
    }
    else if (Rkg[j] == Rkg[j+1] && (j+1) == (gSize-1)   ) { // the end of file
      sameIndex.push_back(j);
      sameIndex.push_back(j+1);
      dR = (Rkg[j+1] - Rkg[sameIndex[0]-1])/sameIndex.size();
      cout << "========================================" << endl;
      counter = 0;
      cout << "dR = " << dR << endl;
      cout << "size = " << sameIndex.size() << endl;
      for ( int k = 0; k < sameIndex.size(); k ++)
	cout << "sameIndex = " << sameIndex[k] << endl;
      
      for ( int i = sameIndex[0]; i < sameIndex[sameIndex.size()-1]+1 ; i ++){
	counter ++;
	cout << "counter = " << counter << endl;
	Rkg[i] =  Rkg[sameIndex[0]-1] + counter * dR;
	cout << "Rkg[ " << i << "] =" << 
	  Rkg[i] << endl;
	
      }
     
      sameRkg = false;
      sameIndex.clear();
      // exit(1);
      
    }
    else if (  Rkg[j] != Rkg[j+1] && sameRkg ) { // reset Rkg values
      sameIndex.push_back(j);    
      dR = (Rkg[j+1] - Rkg[j])/sameIndex.size();
      cout << "========================================" << endl;
      counter = 0;
      cout << "dR = " << dR << endl;
      cout << "size = " << sameIndex.size() << endl;
      for ( int k = 0; k < sameIndex.size(); k ++)
	cout << "sameIndex = " << sameIndex[k] << endl;
      
      for ( int i = sameIndex[0]; i < sameIndex[sameIndex.size()-1]+1 ; i ++){

	cout << "counter = " << counter << endl;
	Rkg[i] = Rkg[i] + counter * dR;
	cout << "Rkg[ " << i << "] =" << 
	         Rkg[i] << endl;
	counter ++;
	
      }
     
      sameRkg = false;
      sameIndex.clear();
      // exit(1); 

    } 
     else if (Rkg[j] != Rkg[j+1] ){
       sameRkg = false;
     }
    
  }

  singleArrayTable(Rkg, gSize, 1, "afterInterpolationRkg");
  
  return 0;
}


// sorting
/*
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

bool myfunction (int i,int j) { return (i<j); }

struct myclass {
  bool operator() (int i,int j) { return (i<j);}
} myobject;

int main () {
  int myints[] = {32,71,12,45,26,80,53,33};
  vector<int> myvector (myints, myints+8);               // 32 71 12 45 26 80 53 33
  vector<int>::iterator it;

  // using default comparison (operator <):
  sort (myvector.begin(), myvector.begin()+4);           //(12 32 45 71)26 80 53 33

  // using function as comp
  sort (myvector.begin()+4, myvector.end(), myfunction); // 12 32 45 71(26 33 53 80)

  // using object as comp
  sort (myvector.begin(), myvector.end(), myobject);     //(12 26 32 33 45 53 71 80)

  // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
*/
