#include "TSVD.h"

#define Max(a,b) (a>b?a:b)
#define Min(a,b) (a<b?a:b)

template<> void TSVD<double>::set_sigma(double sigma_l){
  
  sigma = sigma_l;
  
}

//----------------------------------------------------------------------------
template<> void TSVD<double>::factor(){
  
  SVD<double> svd(*A);
  svd.solve();
  
  U = svd.get_U();
  VT = svd.get_VT();
  S = svd.get_S();
  
  
  for(int i=0;i<nrows;i++){
    for(int j=0;j<nrows;j++){
      (*P)(i) = (*P)(i) + (*U)(j,i)*(*B)(j);
    }}
  
}

//----------------------------------------------------------------------------
template<> void TSVD<double>::solve(){
  
  (*X)=0.0;
  
  
  for(int k=0;k<ncols;k++){
    for(int i = 0;i<Min(nrows,ncols);i++){
      if ((*S)(i) > sigma)
	{
	  (*X)(k) = (*X)(k) + (*P)(i)/(*S)(i)*(*VT)(i,k);
	}
    }}
  
  
} 
