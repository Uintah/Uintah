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
    
//    cout << "Calling LAPACK dgtsv_() "<<endl; 
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

    cout <<"Not implemented in this version!"<<endl;
  }
   else
    cerr << "I should not be here!"<<endl;  
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
    cout <<"Not implemented in this version!"<<endl;}

  else
    cerr << "I should not be here!"<<endl;
  
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

  cout<<"********************************************"<<endl; 
  cout<<"Linear System:"<<endl;
  cout<<"Data Type = 'double'"<<endl;
  cout<<"Matrix = "<<n<<" x "<<n<<endl;
  cout<<"Matrix Type:"<<matrix_type<<endl;
  cout<<"RHS = "<<n<<" x "<< nrhs<<endl;
  cout<<"************"<<endl;
  cout<<"LAPACK Solver:"<<solver<<endl; 
  cout<<"LAPACK info = "<<info_<<endl;
  cout<<"LAPACK result =  "<<messege_<<endl;
  cout<<"LAPACK time = "<<(double)time/CLOCKS_PER_SEC<<"s"<<endl;
  cout<<"********************************************"<<endl;
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

  cout<<"********************************************"<<endl; 
  cout<<"Linear System:"<<endl;
  cout<<"Data Type = 'Complex'"<<endl;
  cout<<"Matrix = "<<n<<" x "<<n<<endl;
  cout<<"Matrix Type:"<<matrix_type<<endl;
  cout<<"RHS = "<<n<<" x "<< nrhs<<endl;
  cout<<"************"<<endl;
  cout<<"LAPACK Solver:"<<solver<<endl; 
  cout<<"LAPACK info = "<<info_<<endl;
  cout<<"LAPACK result =  "<<messege_<<endl;
  cout<<"LAPACK time = "<<(double)time/CLOCKS_PER_SEC<<"s"<<endl;
  cout<<"********************************************"<<endl;



  
}
#endif
//----------------------------------------------------------------------------
