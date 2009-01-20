#include <cmath>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <vector>
#include <sstream>

using namespace std;

int main(){
  double *test = new double [500];
  for ( int i = -400; i < 9; i ++ ){
    test[i] = i;
    cout << "test[" << i << "] = " << test[i] << endl;
  }

  //  delete[] test;
  return 0;

}
  
  
