/*
 *  Array2.cc: Implementation of dynamic 2D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */
namespace rtrt {

template<class T>
Array2<T>::Array2()
{
    objs=0;
    dm1=dm2=0;
}

template<class T>
void Array2<T>::allocate()
{
    if(dm1==0 || dm2==0){
	objs=0;
	refcnt=0;
	return;
    }
    objs=new T*[dm1];
    T* p=new T[dm1*dm2];
    for(int i=0;i<dm1;i++){
        objs[i]=p;
        p+=dm2;
    }
    refcnt=new int;
    *refcnt=1;
}

template<class T>
void Array2<T>::resize(int d1, int d2)
{
    if(objs && dm1==d1 && dm2==d2)return;
    dm1=d1;
    dm2=d2;
    if(objs){
	(*refcnt)--;
	if(*refcnt == 0){
	    delete[] objs[0];
	    delete[] objs;
	    delete refcnt;
	}
    }
    allocate();
}

template<class T>
Array2<T>::Array2(int dm1, int dm2)
: dm1(dm1), dm2(dm2)
{
    allocate();
}

template<class T>
Array2<T>::~Array2()
{
    if(objs){
	(*refcnt)--;
	if(*refcnt == 0){
	    delete[] objs[0];
	    delete[] objs;
	    delete refcnt;
	}
    }
}

template<class T>
void Array2<T>::initialize(const T& t)
{
    ASSERT(objs != 0);
    for(int i=0;i<dm1;i++){
        for(int j=0;j<dm2;j++){
	    objs[i][j]=t;
        }
    }
}

template<class T>
void Array2<T>::share(const Array2<T>& copy)
{
    if(objs){
	(*refcnt)--;
	if(*refcnt == 0){
	    delete[] objs[0];
	    delete[] objs;
	    delete refcnt;
	}
    }
    objs=copy.objs;
    refcnt=copy.refcnt;
    dm1=copy.dm1;
    dm2=copy.dm2;
    (*refcnt)++;
}

template<class T>
Array2<T>& Array2<T>::operator=(const Array2<T>& copy)
{
    resize(copy.dm1,copy.dm2);

    for(int i=0; i<dm1; i++)
    {
        for (int j=0; j<dm2; j++)
        {
            objs[i][j] = copy.objs[i][j];
        }
    }
    return *this;

}


#define Array2_VERSION 1

template<class T>
void Pio(SCIRun::Piostream& stream, Array2<T>& data)
{
  stream.begin_class("rtrtArray2", Array2_VERSION);
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
      SCIRun::Pio(stream, data.objs[i][j]);
    }
  }
  stream.end_class();
}

template<class T>
void Pio(SCIRun::Piostream& stream, Array2<T>*& data) {
  if (stream.reading()) {
    data=new Array2<T>;
  }
  Pio(stream, *data);
}

} // End namespace rtrt
