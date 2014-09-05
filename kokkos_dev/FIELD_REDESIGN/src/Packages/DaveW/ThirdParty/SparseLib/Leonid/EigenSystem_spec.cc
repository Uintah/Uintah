#include "EigenSystem.h"

//----------------------------------------------------------------------------
template<> void EigenSystem<double>::solve(){

  if (!strcmp(matrix_type,"dense")){
    
//LAPACK staff:
    char JOBVL,JOBVR;
    int INFO,LDA,LDVL,LDVR,LWORK,N;
    double *VL,*VR,*WORK,*WI,*WR;
    
    N = NN;
    LDA = N;
    WR = new double[N];
    WI = new double[N];
    
    if (rvec == 1) { 
      JOBVR ='V';
      LDVR = N;
      VR = new double[LDVR*N];
      LWORK = 4*N;
      WORK = new double[LWORK];
    }
    else{
      JOBVR = 'N';
      LDVR = 1;
      VR = new double[1];
    }
    
    
    if (lvec == 1){
      JOBVL = 'V';
      LDVL = N; 
      VL = new double[LDVL*N];
      LWORK = 4*N;
      WORK = new double[LWORK];
    }
    else{
      JOBVL = 'N';
      LDVL = 1;
      VL = new double[1];
    }

    
    if ((lvec !=1) && (rvec != 1)){
      LWORK = 3*N;
      WORK = new double[LWORK];
    }
    
    std::cout << "Calling LAPACK dgeev_() "<<std::endl;
    dgeev_(&JOBVL,&JOBVR, &N,((MatrixDense<double>*) A)->get_p(), &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);
    
    if (INFO == 0)
      messege_ = "Done!";
    if (INFO < 0)
      messege_ =  "Wrong Arguments!";
    if (INFO > 0)
      messege_ = "QR Algorithm Failed !"; 
    
    info_ = INFO;
    
    cEV = new ZVector<Complex>(N,WR,WI);
    
    if (rvec == 1)
      EVR = new MatrixDense<double>(N,N,VR);
    
    if (lvec == 1)
      EVL = new MatrixDense<double>(N,N,VL);
    
  }
  
  if (!strcmp(matrix_type,"tridiag")){

//LAPACK staff:
    char JOBZ;
    int INFO,LDZ,N;
    double *WORK,*Z;
    
    N = NN;
    
    if (rvec == 1)  {
      JOBZ ='V';
      LDZ = N;
      Z = new double[LDZ*N];
      WORK = new double[2*N-2];
    }
    else{
      JOBZ ='N';
      LDZ = 1;
      Z = new double[1];
      WORK = new double[1];
    }
    
    std::cout << "Calling LAPACK dstev_() "<<std::endl;
    dstev_(&JOBZ, &N,((MatrixTridiag<double>*)A)->get_pd(),((MatrixTridiag<double>*)A)->get_pdu(),Z,&LDZ,WORK,&INFO);
    
    if (INFO == 0)
      messege_ = "Done!";
    if (INFO < 0)
      messege_ = "Wrong Arguments!";
    if (INFO > 0)
      messege_ = "Failed to Converge!"; 

    info_ = INFO;

    dEV = new ZVector<double>(N,((MatrixTridiag<double>*)A)->get_pd(),((MatrixTridiag<double>*)A)->get_pd());
    
    if (rvec == 1)
      EVR = new MatrixDense<double>(N,N,Z);
    
  }
  
  else
    std::cerr << "I should not be here!"<<std::endl;
  
  
}  
//---------------------------------------------------------------------------
#if 0
void EigenSystem<Complex>::solve(){
  
//LAPACK staff:
  char JOBVL,JOBVR;
  int INFO,LDA,LDVL,LDVR,LWORK,N;
  double *RWORK;
  Complex *WORK,*W,*VL,*VR;
  
  
  N = NN;
  LDA = N;
  W = new Complex[N];
  LWORK = 2*N;
  WORK = new Complex[LWORK];
  RWORK= new double[2*N];

  
  if (rvec == 1) { 
    JOBVR ='V';
    LDVR = N;
    VR = new Complex[LDVR*N];
  }
  else{
    JOBVR = 'N';
    LDVR = 1;
    VR = new Complex[1];
  }
  
  
  if (lvec == 1){
    JOBVL = 'V';
    LDVL = N; 
    VL = new Complex[LDVL*N];
  }
  else{
    JOBVL = 'N';
    LDVL = 1;
    VL = new Complex[1];
  }
   

  cout << "Calling LAPACK zgeev_() "<<endl;
  zgeev_(&JOBVL,&JOBVR, &N,((MatrixDense<Complex>*) A)->get_p(), &LDA, W, VL, &LDVL, VR, &LDVR, WORK, &LWORK,RWORK, &INFO);
  
  if (INFO == 0)
    messege_ = "Done!";
  if (INFO < 0)
    messege_ = "Wrong Arguments!";
  if (INFO > 0)
    messege_ = "QR Algorithm Failed !"; 

  info_ = INFO;
  
  cEV = new ZVector<Complex>(N,W); 

  if (rvec == 1)
   EVR = new MatrixDense<Complex>(N,N,VR);

  if (lvec == 1)
   EVL = new MatrixDense<Complex>(N,N,VL);
    
}
#endif
