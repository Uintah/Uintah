#include "LinearSystem.h"

#define Max(a,b) (a>b?a:b)
#define Min(a,b) (a<b?a:b)

template<> void LinearSystem<double>::solve(){
  
  if (!strcmp(matrix_type,"dense")){
    
//LAPACK staff: 
    int INFO,LDA,LDB,N,NRHS;
    int* IPIV;
    
    N = n;
    NRHS = nrhs;
    LDA = N;
    IPIV = new int[N];
    LDB = N;
     
    time = clock();
    dgesv_(&N, &NRHS, ((MatrixDense<double>*) A)->get_p(), &LDA, IPIV,B->get_p(), &LDB, &INFO );
    time = clock() - time;
    
    if (INFO == 0)
      messege_ = "Done!";
    if (INFO < 0)
      messege_ = "Wrong Arguments!";
    if (INFO > 0)
      messege_ = "Singular Matrix!";
    
    info_ = INFO;

    X = B;   
  }
  
  else if (!strcmp(matrix_type,"tridiag")){
    
//LAPACK staff:
    
    int INFO,LDB,N,NRHS;
    
    N = n;
    NRHS = nrhs;
    LDB = N;
    
//    std::cout << "Calling LAPACK dgtsv_() "<<std::endl; 
    time = clock();
    dgtsv_(&N, &NRHS, ((MatrixTridiag<double>*) A)->get_pdl(),((MatrixTridiag<double>*) A)->get_pd(),((MatrixTridiag<double>*) A)->get_pdu(),B->get_p(), &LDB, &INFO );
    time = clock() - time;

    
    if (INFO == 0)
      messege_ = "Done!";
    if (INFO < 0)
      messege_ = "Wrong Arguments!";
    if (INFO > 0)
      messege_ = "Singular Matrix!";   
    
    info_ = INFO;
    X = B;
    
  }
  
  else if (!strcmp(matrix_type,"sparse")){

    std::cout <<"Not implemented in this version!"<<std::endl;
  }
   else
    std::cerr << "I should not be here!"<<std::endl;  
}
//----------------------------------------------------------------------------
#if 0
void LinearSystem<Complex>::solve(){
    
  if (!strcmp(matrix_type,"dense")){
    
//LAPACK staff: 
    int INFO,LDA,LDB,N,NRHS;
    int* IPIV;
    
    N = n;
    NRHS = nrhs;
    LDA = N;
    IPIV = new int[N];
    LDB = N;
    
    time = clock(); 
    zgesv_(&N, &NRHS,((MatrixDense<Complex>*) A)->get_p(), &LDA, IPIV,B->get_p(), &LDB, &INFO );
    time = clock() - time;
    
    if (INFO == 0)
      messege_ = "Done!";
    if (INFO < 0)
      messege_ =  "Wrong Arguments!";
    if (INFO > 0)
      messege_ =  "Singular Matrix!";  
    
    info_ = INFO;
    X = B; 
    
  }
  else if (!strcmp(matrix_type,"tridiag")){

    
//LAPACK staff:
    int INFO,LDB,N,NRHS;
    
    N = n;
    NRHS = nrhs;
    LDB = N;
    
    time = clock();
    zgtsv_(&N, &NRHS,((MatrixTridiag<Complex>*) A)->get_pdl(),((MatrixTridiag<Complex>*) A)->get_pd(),((MatrixTridiag<Complex>*) A)->get_pdu(),B->get_p(),&LDB, &INFO );
    time = clock() - time;
    
    
    if (INFO == 0)
      messege_ = "Done!";
    if (INFO < 0)
      messege_ = "Wrong Arguments!";
    if (INFO > 0)
      messege_ = "Singular Matrix!";   
    
    info_ = INFO;
    X = B;
   
  }

  else if (!strcmp(matrix_type,"sparse")){
    std::cout <<"Not implemented in this version!"<<std::endl;}

  else
    cerr << "I should not be here!"<<std::endl;
  
}
#endif

//----------------------------------------------------------------------------
template<> void LinearSystem<double>:: info(){

  char *solver;
  if(!strcmp(matrix_type,"dense"))
    solver = "dgesv_";
  else if(!strcmp(matrix_type,"tridiag"))
    solver = "dgtsv_";
  else if(!strcmp(matrix_type,"sparse"))
    solver = "dgssv_";

  std::cout<<"********************************************"<<std::endl; 
  std::cout<<"Linear System:"<<std::endl;
  std::cout<<"Data Type = 'double'"<<std::endl;
  std::cout<<"Matrix = "<<n<<" x "<<n<<std::endl;
  std::cout<<"Matrix Type:"<<matrix_type<<std::endl;
  std::cout<<"RHS = "<<n<<" x "<< nrhs<<std::endl;
  std::cout<<"************"<<std::endl;
  std::cout<<"LAPACK Solver:"<<solver<<std::endl; 
  std::cout<<"LAPACK info = "<<info_<<std::endl;
  std::cout<<"LAPACK result =  "<<messege_<<std::endl;
  std::cout<<"LAPACK time = "<<(double)time/CLOCKS_PER_SEC<<"s"<<std::endl;
  std::cout<<"********************************************"<<std::endl;
}

#if 0
void LinearSystem<Complex>:: info(){

  char *solver;
  if(!strcmp(matrix_type,"dense"))
    solver = "zgesv_";
  else if(!strcmp(matrix_type,"tridiag"))
    solver = "zgtsv_";
  else if(!strcmp(matrix_type,"sparse"))
    solver = "zgssv_";

  std::cout<<"********************************************"<<std::endl; 
  std::cout<<"Linear System:"<<std::endl;
  std::cout<<"Data Type = 'Complex'"<<std::endl;
  std::cout<<"Matrix = "<<n<<" x "<<n<<std::endl;
  std::cout<<"Matrix Type:"<<matrix_type<<std::endl;
  std::cout<<"RHS = "<<n<<" x "<< nrhs<<std::endl;
  std::cout<<"************"<<std::endl;
  std::cout<<"LAPACK Solver:"<<solver<<std::endl; 
  std::cout<<"LAPACK info = "<<info_<<std::endl;
  std::cout<<"LAPACK result =  "<<messege_<<std::endl;
  std::cout<<"LAPACK time = "<<(double)time/CLOCKS_PER_SEC<<"s"<<std::endl;
  std::cout<<"********************************************"<<std::endl;



  
}
#endif
//----------------------------------------------------------------------------
