#ifndef Gauss_Legendre_Chebyshev_quadrature_h
#define Gauss_Legendre_Chebyshev_quadrature_h
extern "C"{
# define DGEEV FIX_NAME(dgeev)
void DGEEV( char* jobvl, char* jobvr, int* n, double* a,
    int* lda, double* wr, double* wi, double* vl, int* ldvl,
    double* vr, int* ldvr, double* work, int* lwork, int* info );
}


//****************************************************************************
//  Use "quicksort" algorithm to sort doubles
//  //  Pulled from  http://www.algolist.net/Algorithms/Sorting/Quicksort
//  //  and altered for use with doubles, and added a second array
//  //  conditioned on the first array's sorting order. 
//  //  This is a recursive method.
//****************************************************************************
void dualQuickSort(double arr[], int left, int right, double arr2[]) {

  int i = left, j = right;
  double tmp;
  double tmp2;
  double pivot = arr[(left + right) / 2];
  /* partition */
  while (i <= j) {
    while (arr[i] > pivot) // Is this comparison ok for doubles?
      i++;
    while (arr[j] < pivot)
      j--;
    if (i <= j) {
      tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;

      tmp2 = arr2[i];
      arr2[i] = arr2[j];
      arr2[j] = tmp2;

      i++;
      j--;
    }
  };

  /* recursion */
  if (left < j)
    dualQuickSort(arr, left, j,arr2);
  if (i < right)
    dualQuickSort(arr, i, right,arr2);
}

//****************************************************************************
//// Computes the weights and abscissa for gauss-legendre quadrature 
//   using the Golub-Welsch algorithm.
////****************************************************************************
void
GaussQuadrature(int N, double* weights, double* abscissa ){

  //GAUSS   nodes x (Legendre points) and weights w
  // //        for Gauss quadrature
  double A_Matrix[N*N];
  //Initialize A matrix
  for (int j=0; j<N; j++){
    for (int i=0; i<N; i++){
      A_Matrix[i+j*N] =0.0;
    }
  }


  //Populate A matrix
  for (int i=0; i<N-1; i++){
    double beta=0.5/sqrt(1.0-std::pow(2.0*((double)i+1.0),-2.0));
    A_Matrix[i   +  N*(i+1)] = beta;
    A_Matrix[i+1 +  N*(i)] = beta;
  }

  //Solve for EigenValues and EigenVectors (this method solves a full non-symmetric matrix - our system is sparse and symmetric. TODO: Find a more relevant solver)
  int ldvl = N;
  int ldvr = N;
  int lda = N;

  double wi[N];
  double vl[ldvl*N];
  double vr[ldvr*N];
  double wkopt;
  double* work;
  int lwork;
  int info;
  lwork = -1;
  char useVector[8]="Vectors";
  DGEEV( useVector, useVector, &N, A_Matrix, &lda, abscissa, wi, vl, &ldvl, vr, &ldvr, &wkopt, &lwork, &info );
  lwork = (int)wkopt;
  work = (double*)malloc( lwork*sizeof(double) );
  //* Solve for eigenvalues */
  DGEEV( useVector,useVector, &N, A_Matrix, &lda, abscissa, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info );
  delete work; 

  // Eigenvalues are the  Abscissa and the first elements of the eigenvectors are used to compute the weights        
  for (int j=0; j<N; j++){
    weights[j]=2*vl[j*N]*vl[j*N];
  }  
  // Sort eigenvalues and eigenvectors
  dualQuickSort(abscissa,0,N-1,weights);

}

//****************************************************************************
//// Computes a triangular using Gauss-Legendre along the polar axis.
//   Chebyshev quadrature is across the Azimuthal Angle.  
//   The weights are normalized to sum to pi/2 for each octant.
////****************************************************************************
void computeLegendreChebyshevQuadratureSet(int N, std::vector<double> &xComp, std::vector<double> &yComp,
    std::vector<double> &zComp, std::vector<double> &pWeight){

  double LWeights[N];// level weights
  double abscissa[N];// level abscissa
  GaussQuadrature(N,LWeights,abscissa);

  int octant =N*(N+2)/8;
  std::vector<double> azAngle(octant,0.0);

  double zero;
  int ix=0;
  for (int i=0; i< N/2; i++){
    for (int j=0; j<= i; j++){
      azAngle[ix]= (2.0*((double) i+1)-2.0*((double) j+1)+1.0)/2.0/((double) i+1)/2.0*M_PI; //   % chebyshev of first kind Longoni 2004
      zComp[ix]=abscissa[i];
      pWeight[ix]=LWeights[i]/(i+1)*M_PI/2.0;
      double R=sqrt(1.0-zComp[ix]*zComp[ix]);
      xComp[ix]=sin(azAngle[ix])*R;
      yComp[ix]=cos(azAngle[ix])*R;
      zero=zero+pWeight[ix];
      ix=ix+1;
    }
  }
  // Generate 7 additional octants using symmetry
  for (int i=0; i< 2; i++){
    for (int j=0; j< 2; j++){
      for (int k=0; k< 2; k++){
        if (i==0 && j==0 && k==0)
          continue;
        for (int w=0; w< octant; w++){
          xComp[w+  i*octant*4  + j*octant*2 + k*octant]=i ?  -xComp[w] : xComp[w];
          yComp[w+  i*octant*4  + j*octant*2 + k*octant]=j ?  -yComp[w] : yComp[w];
          zComp[w+  i*octant*4  + j*octant*2 + k*octant]=k ?  -zComp[w] : zComp[w];
          pWeight[w+i*octant*4  + j*octant*2 + k*octant]=pWeight[w];
        }
      }
    }
  }
}

#endif  
