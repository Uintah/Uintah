#include "MatrixDense.h"
  
//---------------------------------------------------------------------
template<> void MatrixDense<double>:: info(){
  
  std::cout<<"********************************************"<<std::endl; 
  std::cout<<"Matrix:"<<std::endl;
  std::cout<<"Data Type = 'double'"<<std::endl;
  std::cout<<"Matrix Type = 'dense'"<<std::endl;
  std::cout<<"Size = "<<nrows<<" x "<<ncols<<std::endl;
  std::cout<<"********************************************"<<std::endl;
}

#if 0
void MatrixDense<Complex>:: info(){
  
  std::cout<<"********************************************"<<std::endl; 
  std::cout<<"Matrix:"<<std::endl;
  std::cout<<"Data Type = 'Complex'"<<std::endl;
  std::cout<<"Matrix Type = 'dense'"<<std::endl;
  std::cout<<"Size = "<<nrows<<" x "<<ncols<<std::endl;
  std::cout<<"********************************************"<<std::endl;
}
#endif

//---------------------------------------------------------------------



