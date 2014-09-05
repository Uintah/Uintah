
#ifndef Math_CatmullRomSpline_h
#define Math_CatmullRomSpline_h 1

#include <Core/Persistent/Pstreams.h>

namespace rtrt {
template <class T> class CatmullRomSpline;
}

namespace SCIRun {
template <class T>
void Pio(Piostream&, rtrt::CatmullRomSpline<T>&);
}

namespace rtrt {

template<class T>
class CatmullRomSpline {
  T* d;
  int nintervals;
  int nset;
  int mx;
public:
  CatmullRomSpline(const T&,const T&,const T&,const T&);
  CatmullRomSpline(const T&,const T&,const T&,const T&,const T&);
  CatmullRomSpline(const T&,const T&,const T&,const T&,const T&,
		   const T&);
  CatmullRomSpline(const T&,const T&,const T&,const T&,const T&,
		   const T&,const T&);
  CatmullRomSpline(const T&,const T&,const T&,const T&,const T&,
		   const T&,const T&,const T&);
  CatmullRomSpline(const T&,const T&,const T&,const T&,const T&,
		   const T&,const T&,const T&,const T&);
  CatmullRomSpline(const T&,const T&,const T&,const T&,const T&,
		   const T&,const T&,const T&,const T&,const T&);
  CatmullRomSpline(const T&,const T&,const T&,const T&,const T&,
		   const T&,const T&,const T&,const T&,const T&,
		   const T&);
  CatmullRomSpline(const T&,const T&,const T&,const T&,const T&,
		   const T&,const T&,const T&,const T&,const T&,
		   const T&,const T&);
  CatmullRomSpline(const T&,const T&,const T&,const T&,const T&,
		   const T&,const T&,const T&,const T&,const T&,
		   const T&,const T&,const T&);
  CatmullRomSpline(int);
  CatmullRomSpline(const CatmullRomSpline<T> &);
  CatmullRomSpline();
  
  void add(const T&);
  T operator()(double);

  T*& get_d() { return d; }
  int& get_nintervals() { return nintervals; }
  int& get_nset() { return nset; }
  int& get_mx() { return mx; }
};
} // end namespace rtrt
namespace SCIRun {
template<class T>
void Pio(SCIRun::Piostream &str, rtrt::CatmullRomSpline<T> &crs) {
  str.begin_cheap_delim();
  Pio(str, crs.get_nintervals());
  Pio(str, crs.get_nset());
  Pio(str, crs.get_mx());
  
  if (str.reading()) {
    crs.get_d() = new T[crs.get_nset()];
  }
  for (int i = 0; i < crs.get_nset(); i++) {
    Pio(str, crs.get_d()[i]);
  }

  str.end_cheap_delim();
}
} // end namespace SCIRun
namespace rtrt {

template<class T>
CatmullRomSpline<T>::CatmullRomSpline(const T& t1, const T& t2,
				      const T& t3, const T& t4)
  : nset(4)
{
  d=new T[4];
  d[0]=t1; d[1]=t2; d[2]=t3; d[3]=t4;
  nintervals=1;
  mx=0;
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline(const T& t1, const T& t2,
				      const T& t3, const T& t4,
				      const T& t5)
  : nset(5)
{
  d=new T[5];
  d[0]=t1; d[1]=t2; d[2]=t3; d[3]=t4; d[4]=t5;
  nintervals=2;
  mx=1;
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline(const T& t1, const T& t2, const T& t3,
				      const T& t4, const T& t5, const T& t6)
  : nset(6)
{
  d=new T[6];
  d[0]=t1; d[1]=t2; d[2]=t3; d[3]=t4; d[4]=t5; d[5]=t6;
  nintervals=3;
  mx=2;
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline(const T& t1, const T& t2, const T& t3,
				      const T& t4, const T& t5, const T& t6,
				      const T& t7)
  : nset(7)
{
  d=new T[7];
  d[0]=t1; d[1]=t2; d[2]=t3; d[3]=t4; d[4]=t5; d[5]=t6; d[6]=t7;
  nintervals=4;
  mx=3;
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline(const T& t1, const T& t2, const T& t3,
				      const T& t4, const T& t5, const T& t6,
				      const T& t7, const T& t8)
  : nset(8)
{
  d=new T[8];
  d[0]=t1; d[1]=t2; d[2]=t3; d[3]=t4; d[4]=t5; d[5]=t6;
  d[6]=t7; d[7]=t8;
  nintervals=5;
  mx=4;
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline(const T& t1, const T& t2, const T& t3,
				      const T& t4, const T& t5, const T& t6,
				      const T& t7, const T& t8, const T& t9)
  : nset(9)
{
  d=new T[9];
  d[0]=t1; d[1]=t2; d[2]=t3; d[3]=t4; d[4]=t5; d[5]=t7;
  d[6]=t7; d[7]=t8; d[8]=t9;
  nintervals=6;
  mx=5;
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline(const T& t1, const T& t2, const T& t3,
				      const T& t4, const T& t5, const T& t6,
				      const T& t7, const T& t8, const T& t9,
				      const T& t10)
  : nset(10)
{
  d=new T[10];
  d[0]=t1; d[1]=t2; d[2]=t3; d[3]=t4; d[4]=t5; d[5]=t6;
  d[6]=t7; d[7]=t8; d[8]=t9; d[9]=t10;
  nintervals=7;
  mx=6;
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline(const T& t1, const T& t2, const T& t3,
				      const T& t4, const T& t5, const T& t6,
				      const T& t7, const T& t8, const T& t9,
				      const T& t10, const T& t11)
  : nset(11)
{
  d=new T[11];
  d[0]=t1; d[1]=t2; d[2]=t3; d[3]=t4; d[4]=t5; d[5]=t6;
  d[6]=t7; d[7]=t8; d[8]=t9; d[9]=t10; d[10]=t11;
  nintervals=8;
  mx=7;
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline(const T& t1, const T& t2, const T& t3,
				      const T& t4, const T& t5, const T& t6,
				      const T& t7, const T& t8, const T& t9,
				      const T& t10, const T& t11, const T& t12)
  : nset(12)
{
  d=new T[12];
  d[0]=t1; d[1]=t2; d[2]=t3; d[3]=t4; d[4]=t5; d[5]=t6; d[6]=t7;
  d[7]=t8; d[8]=t9; d[9]=t10; d[10]=t11; d[11]=t12;
  nintervals=9;
  mx=8;
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline(const T& t1, const T& t2, const T& t3,
				      const T& t4, const T& t5, const T& t6,
				      const T& t7, const T& t8, const T& t9,
				      const T& t10, const T& t11, const T& t12,
				      const T& t13)
  : nset(13)
{
  d=new T[13];
  d[0]=t1; d[1]=t2; d[2]=t3; d[3]=t4; d[4]=t5; d[5]=t7;
  d[6]=t7; d[7]=t8; d[8]=t9; d[9]=t10; d[10]=t11;
  d[11]=t12; d[12]=t13;
  nintervals=10;
  mx=9;
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline(int n)
{ 
  d=new T[n];
  nset=0;
  nintervals=n-3;
  mx=n-4;
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline()
{
  d=0;
  nset=0;
  nintervals=0;
  mx=0;
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline(const CatmullRomSpline& s)
  : d(s.d), nset(s.nset)
{
  d=new T[s.nintervals+3];
  for(int i=0;i<s.nintervals+3;i++)d[i]=s.d[i];
  nintervals=s.nintervals;
  mx=s.mx;
}

template<class T>
void CatmullRomSpline<T>::add(const T& obj)
{
  nset++;
  T* dd=new T[nset];
  int i;
  for(i=0;i<nset-1;i++)dd[i]=d[i];
  dd[i]=obj;
  delete[] d;
  d=dd;
  nintervals=nset-3;
  mx=nset-4;
}

template<class T>
T CatmullRomSpline<T>::operator()(double x)
{
  double xs=x*nintervals;
  int idx=int(xs);
  double t=xs-idx;
  if(idx<0){idx=0;t=0;}
  if(idx>mx){idx=mx;t=1;}
  double t2=t*t;
  double t3=t*t*t;
#if 0
  T y=(d[idx]*-1   + d[idx+1]*3 + d[idx+2]*-3 + d[idx+3])*(t3*0.5)+
    (d[idx]*2 + d[idx+1]*-5 + d[idx+2]*4 + d[idx+3]*-1)*(t2*0.5)+
    (d[idx]*-1                + d[idx+2]             )*(t*0.5)+
    (            d[idx+1]                          );
	
  return y;
#else
  t*=0.5;
  double w0 = -0.5*t3 + t2 - t;
  double w1 = 1.5*t3 - 2.5*t2 + 1;
  double w2 = -1.5*t3 + 2*t2 + t;
  double w3 = 0.5*t3 - 0.5*t2;
  return d[idx]*w0 + d[idx+1]*w1 + d[idx+2]*w2 + d[idx+3]*w3;
#endif
}

} // end namespace rtrt

#endif
