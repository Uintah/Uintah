#ifndef BinarySearchTree_H
#define BinarySearchTree_H
#include <iostream>

using namespace std;

class BinarySearchTree{

public:

  BinarySearchTree();
  ~BinarySearchTree();
  
  
  inline
  int get_lowI(){
    return lowI;
  }
  
  
  inline
  int get_highI(){
    return highI;
  }
  
  
  
  inline
  void search(const double &Rgg,
	      const double *Rkg,
	      const int &gSize){
    
    int rootI, endI, startI;
    startI = 0;
    endI = gSize -1;
    rootI = int((startI + endI)/2);
    bool found;
    found = false;
    
    while ( !found) {

      if ( Rgg == Rkg[rootI] ){
	found = true;
	lowI = rootI;
	highI = rootI;
      }
      else if ( Rgg > Rkg[rootI] ) { // go to right
	startI = rootI;
	// int() is taking floor
	rootI = int( (startI + endI)/2 );
	
      }
      else if ( Rgg < Rkg[rootI]) { // go to left
	endI = rootI;
	rootI = int( (startI + endI)/2 );
	
      }
      
      if ( (endI - startI) == 1 ){// Rgg sits in between
	found = true;
	lowI = startI;
	highI = endI;
      }
      //   cout << "startI = " << startI << endl;
      //  cout << "endI = " << endI << endl;

    }// end of while
    
    // cout << "lowI = " << lowI << endl;
    // cout << "highI = " << highI << endl;
      
    
  }// end of search


  inline
  void calculate_gk(const double *gk,
		    const double *Rkg,
		    const double &Rgg){
    double A;
     
    if ( lowI == highI ) {// find the Rgg exactly there
      g = gk[lowI *2];
      k = gk[lowI *2 +1];
    }
    else{
      // can set a max on (Rgg-Rkg[lowI]), in case it is too small
      A = (Rkg[highI] - Rgg)/(Rgg-Rkg[lowI]);
      g = ( gk[highI *2] + A * gk[lowI*2])/
	(A+1);
      k = ( gk[highI *2+1] + A * gk[lowI*2+1])/
	(A+1); 
    }
    
  }

  
  inline
  double get_g(){
    return g;
  }


  inline
  double get_k(){
    return k;
  }

  
private:
  int lowI, highI;
  double g, k; // k in cm-1 unit
  
  
};
  
  
#endif
