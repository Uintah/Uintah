
/*
 *  BrickArray2.h: Interface to dynamic 2D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2000
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Classlib_BrickArray2_h
#define SCI_Classlib_BrickArray2_h
#include <Core/Persistent/Pstreams.h>
#include <Core/Util/Assert.h>
#include <sci_config.h>

#include <math.h>
#include <iostream>

namespace rtrt {
template<class T> class BrickArray2;
template<class T> void Pio(SCIRun::Piostream& stream, BrickArray2<T>& data);
template<class T> void Pio(SCIRun::Piostream& stream, BrickArray2<T>*& data);

template<class T>
class BrickArray2 {
  T* objs;
  char* data;
  int* refcnt;
  int* idx1;
  int* idx2;
  int dm1;
  int dm2;
  int totaldm1;
  int totaldm2;
  int L1, L2;
  void allocate();
  BrickArray2<T>& operator=(const BrickArray2&);
public:
  typedef T data_type;

  BrickArray2();
  BrickArray2(int, int);
  ~BrickArray2<T>();
  inline T& operator()(int d1, int d2) const
  {
    return objs[idx1[d1]+idx2[d2]];
  }
  inline int dim1() const {return dm1;}
  inline int dim2() const {return dm2;}
  void resize(int, int);
  void initialize(const T&);

  inline T* get_dataptr() {return objs;}
  inline unsigned long get_datasize() {
    return totaldm1*totaldm2*sizeof(T);
  }
  void share(const BrickArray2<T>& copy);

  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (SCIRun::Piostream&, BrickArray2<T>&);
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (SCIRun::Piostream&, BrickArray2<T>*&);

};

template<class T>
BrickArray2<T>::BrickArray2()
{
  objs=0;
  data=0;
  dm1=dm2==0;
  totaldm1=totaldm2=0;
  idx1=idx2=0;
  L1=L2=0;
}

template<class T>
void BrickArray2<T>::allocate()
{
  if(dm1==0 || dm2==0){
    objs=0;
    data=0;
    refcnt=0;
    dm1=dm2=0;
    totaldm1=totaldm2=0;
    idx1=idx2=0;
    L1=L2=0;
    return;
  }
    
  // 128 byte cache line
  L1=(int)(sqrt(128./sizeof(T))+.1);
  // 16K page size
  L2=(int)(sqrt(16384./double(sizeof(T)*L1*L1))+.1);
  cerr << "sizeof(T)=" << sizeof(T) << '\n';
  cerr << "L1=" << L1 << ", L2=" << L2 << '\n';
  int totalx=(dm1+L2*L1-1)/(L2*L1);
  int totaly=(dm2+L2*L1-1)/(L2*L1);

  idx1=new int[dm1];
  cerr << "totalx=" << totalx << ", totaly=" << totaly << '\n';
  for(int x=0;x<dm1;x++){
    int m1x=x%L1;
    int xx=x/L1;
    int m2x=xx%L2;
    int m3x=xx/L2;
    idx1[x]=m3x*totaly*L2*L2*L1*L1+m2x*L2*L1*L1+m1x*L1;
  }
  idx2=new int[dm2];
  for(int y=0;y<dm2;y++){
    int m1y=y%L1;
    int yy=y/L1;
    int m2y=yy%L2;
    int m3y=yy/L2;
    idx2[y]=m3y*L2*L2*L1*L1+m2y*L1*L1+m1y;
  }
  totaldm1=totalx*L2*L1;
  totaldm2=totaly*L2*L1;
  int totalsize=totaldm1*totaldm2;
  ASSERT(idx1[dm1-1]+idx2[dm2-2] < totalsize);
  cerr << "totalsize=" << totalsize*sizeof(T) << '\n';
  //objs=new T[totalsize];
  data=new char[totalsize*sizeof(T)+128+4096];
  unsigned long off=(unsigned long)data%128;
  if(off)
    objs=(T*)(data+128-off);
  else
    objs=(T*)data;
  refcnt=new int;
  *refcnt=1;
}

template<class T>
void BrickArray2<T>::resize(int d1, int d2)
{
  if(objs && dm1==d2 && dm2==d2)return;
  dm1=d1;
  dm2=d2;
  if(objs){
    (*refcnt--);
    if(*refcnt == 0){
      delete[] data;
      delete[] idx1;
      delete[] idx2;
      delete refcnt;
    }
  }
  allocate();
}

template<class T>
BrickArray2<T>::BrickArray2(int dm1, int dm2)
  : dm1(dm1), dm2(dm2)
{
  allocate();
}

template<class T>
BrickArray2<T>::~BrickArray2()
{
  if(objs){
    if(*refcnt == 0){
      delete[] data;
      delete[] idx1;
      delete[] idx2;
      delete refcnt;
    }
  }
}

template<class T>
void BrickArray2<T>::initialize(const T& t)
{
  int n=dm1*dm2;
  for(int i=0;i<n;i++)
    objs[i]=t;
}

template<class T>
void BrickArray2<T>::share(const BrickArray2<T>& copy)
{
  if(objs){
    (*refcnt--);
    if(*refcnt == 0){
      delete[] data;
      delete[] idx1;
      delete[] idx2;
    }
  }
  objs=copy.objs;
  data=copy.data;
  refcnt=copy.refcnt;
  dm1=copy.dm1;
  dm2=copy.dm2;
  totaldm1=copy.totaldm1;
  totaldm2=copy.totaldm2;
  idx1=copy.idx1;
  idx2=copy.idx2;
  L1=copy.L1;
  L2=copy.L2;
  (*refcnt)++;
}

#define BrickArray2_VERSION 1

template<class T>
void Pio(SCIRun::Piostream& stream, BrickArray2<T>& data)
{
  stream.begin_class("rtrtBrickArray2", BrickArray2_VERSION);
  if(stream.reading()){
    // Allocate the array...
    int d1, d2;
    SCIRun::Pio(stream, d1);
    SCIRun::Pio(stream, d2);
    data.resize(d1, d2);
  } else {
    SCIRun::Pio(stream, data.dm1);
    SCIRun::Pio(stream, data.dm2);
  }
  for(int i=0;i<data.dm1;i++){
    for(int j=0;j<data.dm2;j++){
      float &f = data(i, j);
      SCIRun::Pio(stream, f);
    }
  }
  stream.end_class();
}

template<class T>
void Pio(SCIRun::Piostream& stream, BrickArray2<T>*& data) {
  if (stream.reading()) {
    data=new BrickArray2<T>;
  }
  Pio(stream, *data);
}

} // end namespace rtrt

#endif
