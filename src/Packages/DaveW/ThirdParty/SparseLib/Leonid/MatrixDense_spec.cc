#include "MatrixDense.h"
  
//---------------------------------------------------------------------
template<> void MatrixDense<double>:: info(){
  
  cout<<"********************************************"<<endl; 
  cout<<"Matrix:"<<endl;
  cout<<"Data Type = 'double'"<<endl;
  cout<<"Matrix Type = 'dense'"<<endl;
  cout<<"Size = "<<nrows<<" x "<<ncols<<endl;
  cout<<"********************************************"<<endl;
}

#if 0
void MatrixDense<Complex>:: info(){
  
  cout<<"********************************************"<<endl; 
  cout<<"Matrix:"<<endl;
  cout<<"Data Type = 'Complex'"<<endl;
  cout<<"Matrix Type = 'dense'"<<endl;
  cout<<"Size = "<<nrows<<" x "<<ncols<<endl;
  cout<<"********************************************"<<endl;
}
#endif

//---------------------------------------------------------------------



