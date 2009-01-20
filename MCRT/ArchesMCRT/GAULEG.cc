#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;


int main(int argc, char *argv[]) {

  double x1, x2;
  int N;
  int M;
  double xM, xL, z, p1, p2, p3, pp, z1;
  
  cout << "Please enter lower and upper limits of integration x1, x2" << endl;
  cin >> x1;
  cin >> x2;
  cout << "Please enter the length N" << endl;
  cin >> N;
  double *x = new double[N];
  double *W = new double[N];
  
  // GAULEG(x1, x2, x, W, N)
  
  double eps = 3e-14;

  M = int ( N+1)/2;
  xM = 0.5 * ( x1 + x2 );
  xL = 0.5 * ( x2 - x1 ); 

  for (int i = 1; i < M+1; i ++ ) {
    z = cos(3.141592654 * ( i - .25) / ( N + 0.5 ));

    do { 
      p1 = 1;
      p2 = 0;
      
      for ( int j = 1; j < N+1; j ++ ) {
	
	p3 = p2;
	p2 = p1;
	p1 = ( ( 2.0 * j - 1) * z * p2 - ( j-1) * p3 )/j;
      }
      
      pp = N * ( z * p1 - p2 )/ ( z * z - 1 );
      z1 = z;
      z = z1 - p1 /pp;
      
    }while ( abs(z-z1) > eps );

    x[i-1] = xM - xL * z;
    x[N+1-i-1] = xM + xL * z;
    W[i-1] = 2 * xL / ( ( 1 - z * z ) * pp * pp );
    W[N+1-i-1] = W[i-1] ;
    
  }

  for ( int i = 0; i < N; i ++ ){
    cout << " W[ " << i << "] = " << W[i] << endl;
  }

  for ( int i = 0; i < N; i ++ )
    cout << " x[ " << i << "] = " << x[i] << endl;
  
  return 0;

  
}
		    
  
