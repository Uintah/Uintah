#include "Vector.h"
#include "Complex.h"

template<> ZVector<Complex>::ZVector(int N,double x_re[],double x_im[]){
  Size = N;
  
  a = new Complex [N];
  
  for(int i=0;i<N;i++)
    a[i].set(x_re[i],x_im[i]);

}
template<> ZVector<double>::ZVector(int N,double x_re[],double x_im[]){

 cout << "Not implemented for doubles!"<<endl;  

}

template<>
ostream &operator<< (ostream &output, ZVector<double>  &b){

  output<<endl;
  for(int i=0 ;i < b.size();i++);
//    output<<"["<<b.a[i]<<"]"<<endl;
  output<<endl;
  
  return(output);
}


//-----------------------------------------------------------------
template<> void ZVector<double>:: info(){
  
  cout<<"********************************************"<<endl; 
  cout<<"Vector:"<<endl;
  cout<<"Data Type = 'double'"<<endl;
  cout<<"Size = "<<Size<<endl;
  cout<<"********************************************"<<endl;
}

#if 0
void ZVector<Complex>:: info(){
  
  cout<<"********************************************"<<endl; 
  cout<<"Vector:"<<endl;
  cout<<"Data Type = 'Complex'"<<endl;
  cout<<"Size = "<<Size<<endl;
  cout<<"********************************************"<<endl;
}
#endif
//---------------------------------------------------------------------
