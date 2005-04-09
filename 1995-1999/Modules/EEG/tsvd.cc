#include "reg_tools.h"


void tsvd(MatrixDense<double>* A, ZVector<double>* b, ZVector<double> *x,
	  double truncate){ 

  // you can also call SVD without copying matrix, but then
  // it will be changed in computations
  SVD<double> svd(*A,"copy");
  svd.solve();
  svd.info();
  svd.print();
  ZVector<double>* s = svd.get_S();
  MatrixDense<double>* U =  svd.get_U();
  MatrixDense<double>* VT = svd.get_VT();

  int nrows = A->nr();
  int ncols = A->nc();

    
  for(int k=0;k<nrows;k++){
    if ((*s)(k) > truncate){
      double tmp = 0;
      for (int i=0;i<nrows;i++)
	tmp = tmp + (*U)(i,k)*(*b)(i);
      for (i=0;i<ncols;i++)
      	  (*x)(i) = (*x)(i) + (*VT)(k,i)*tmp/(*s)(k);
    }}
  
}
 

