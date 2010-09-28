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
  
  
  // given Rgg, find corresponding  k
  // of course, this function can be used to search for any variable position
  // using binary tree search
  
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
  void search(const double &gg,
	      const double *gk,
	      const int &gSize,
	      const int &d, const int &ii){
    
    int rootI, endI, startI;
    startI = 0;
    endI = gSize -1;
    rootI = int((startI + endI)/2);
    bool found;
    found = false;
    
    while ( !found) {

      if ( gg == gk[rootI*d+ii] ){
	found = true;
	lowI = rootI;
	highI = rootI;
      }
      else if ( gg > gk[rootI*d+ii] ) { // go to right
	startI = rootI;
	// int() is taking floor
	rootI = int( (startI + endI)/2 );
	
      }
      else if ( gg < gk[rootI*d+ii]) { // go to left
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
    double A,B;
     
    if ( lowI == highI ) {// find the Rgg exactly there
      g = gk[lowI *2];
      k = gk[lowI *2 +1];
    }
    else{
      // can set a max on (Rgg-Rkg[lowI]), in case it is too small
      // R--g relation or R-eta
      
      A = (Rkg[highI] - Rgg)/(Rgg-Rkg[lowI]);
      g = ( gk[highI *2] + A * gk[lowI*2])/
	max( (A+1), 1e-9);
      
      // cout << "lowI = " << lowI << "; highI = " << highI << endl;
      // cout << " A = " << A << "; g = " << g << endl;

      // B and A are always the same.
      
	// then from the index get k-g or k-eta
      //   B =  (gk[highI*2] - g)/(g-gk[lowI*2]);
	
	k = ( gk[highI *2+1] + A * gk[lowI*2+1])/
	  max ( (A+1), 1e-9);
	
      
	//	cout << " k = " << k << endl;
      
    }
    
  }


  inline
  void calculate_k(const double *gk,
		    const double &gg){
    double A;
     
    if ( lowI == highI ) {// find the Rgg exactly there
        k = gk[lowI *2 +1];
    }
    else{
      // can set a max on (Rgg-Rkg[lowI]), in case it is too small
      // R--g relation or R-eta
      
      A = (gk[highI *2] - gg)/( gg - gk[lowI*2]);
   
      k = ( gk[highI *2+1] + A * gk[lowI*2+1])/
	max ( (A+1), 1e-9);
      //  cout << "highI = " << highI << "; lowI = " << lowI << endl;
      //  cout << " gg = " << gg << endl;
      //  cout << "k = " << k << endl;

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
