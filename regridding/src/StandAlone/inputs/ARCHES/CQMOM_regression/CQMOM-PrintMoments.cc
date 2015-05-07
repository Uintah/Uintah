#include <vector>
#include <cstdlib>
#include <iostream>

using namespace std;

//This is a "dummy" version of the CQMOM solver that will step through the algorithm
//and output to commadn line  whenver a given moment of the system is required.  In this way  
//the script can be run to determine the set of moments that are required for a number
//of internal coordiantes and quadratures nodes prior to setup of a simulation.
//Usage is ./a.out N_i N_i N_i .... where N_i is the number of quadrature nodes for a 
//given internal coordinate i.e ./a.out 3 2 2 will give the moments required for 3 nodes
//in IC1 and 2 in IC2 and IC3

/***************************
 CQMOMInversion Function
 ****************************/
void CQMOMInversion( const int& M, const vector<int>& N_i)
{
  cout << "List of required moments" << endl;
  
  for (int i = 0; i<N_i[0]*2; i++ ) {
    cout << "m_" << i << endl;
  }
  
  if (M == 1) {
    return;
  }
  
  //loop over each k_2 1,...,2N2-1
  // 0th moment for all conditional = 1  
  for (int k = 1; k<2*N_i[1]; k++) {  
    for (int i=0; i<N_i[0]; i++) {
      cout << "m_" << i << k << endl;
    }
  }
  
  if (M == 2) {
    return;
  }
  
  //start the 3D quadrature method
  for (int k3 = 1; k3<2*N_i[2]; k3++) {
    for (int k2 = 0; k2<N_i[1]; k2++) { //loop through all combinations of k_2/k_3 for zeta values
      // solve V_1R_1 * zeta^(k1,k2) = m for zeta 1 to N1
      //populate the moment vector
      for (int i = 0; i<N_i[0]; i++) {
        cout << "m_" << i << k2 << k3 << endl;
      }
    }
  }  

  if ( M == 3 ) {
    return;
  }
  
  
  //start the 4D quadrature
  for (int k2 = 0; k2 < N_i[1]; k2++ ) {
    for (int k3 = 0; k3 < N_i[2]; k3++ ) {
      for (int k4 = 1; k4 < 2*N_i[3]; k4++ ) {
        //solve each vandermonde matrix of zeta
        //fill in RHS of moments
        for (int k1 = 0; k1<N_i[0]; k1++) {
          cout << "m_" << k1 << k2 << k3 << k4 << endl;
        }
      }
    }
  }

  if ( M == 4 ) {
    return;
  }
  
  /*after M = 4, limit internal coordinates to 1 per direction, otherwise the number of scalar transport eqns
   starts to become prohibitively large, along with the number of matricies to solve */
  
  //with this limitation, only the conditional means of the system need to be calculated, since all the conditional
  //moments are normalized to 1, the weights are unchanged
  for (int m = 5; m <= M; m++) {
    //all the steps in the 4D quadrature need to be repeated, but with  different moment sets based on m

    for (int k2 = 0; k2 < N_i[1]; k2++ ) {
      for (int k3 = 0; k3 < N_i[2]; k3++ ) {
        for (int k4 = 0; k4 < N_i[3]; k4++ ) {
          //solve each vandermonde matrix of zeta
          for (int k1 = 0; k1<N_i[0]; k1++) {
            int flatIndex;
            cout << "m_" << k1 << k2 << k3 << k4;
            for (int i = 5; i <= m; i++) {
              int product; //= 1; //=ki
              if (i == m) {
                product = 1;
              } else {
                product = 0;
              }
              cout << product;
              if ( i == m) {
                cout << endl;
              } 
	    }
          }
        }
      }
    }
  }
  return;
  
} //end CQMOMInversion


/***************************
 MAIN Function
 ****************************/
int main( int argc, const char* argv[] )
{
  int M = argc - 1;
  vector<int> N_i (M, 0);
  for ( int i = 1; i < argc; i++ ) {
    N_i[i-1] = atoi(argv[i]);
  }

  cout << "M = " << M << endl;
    
  int reqMoments;
  reqMoments = 2*N_i[0];
  for ( int i = 1; i < M; i++ ) {
    int product = N_i[0];
    for ( int j = 1; j < i; j ++) {
      product *= N_i[j];
    }
    product*= (2*N_i[i]-1);
    reqMoments += product;
  }
    
  cout << "Number of required Moments: " << reqMoments << endl;

  CQMOMInversion(M, N_i);
}
