
/*
 *  Array2.h: Interface to dynamic 2D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef Package_rtrt_Core_Array2_h
#define Package_rtrt_Core_Array2_h 1

#include <sci_config.h>

namespace rtrt {
template<class T> class Array2;
template<class T> void Pio(SCIRun::Piostream& stream, rtrt::Array2<T>& data);
template<> void Pio(SCIRun::Piostream&, rtrt::Array2<int>& data);
template<> void Pio(SCIRun::Piostream&, rtrt::Array2<float>& data);
template<> void Pio(SCIRun::Piostream&, rtrt::Array2<double>& data);
template<class T> void Pio(SCIRun::Piostream& stream, rtrt::Array2<T>*& data);


template<class T>
class Array2 {
  T** objs;
  int* refcnt;
  int dm1;
  int dm2;
  void allocate();
public:
  typedef T data_type;

  Array2();
  Array2(int, int);
  ~Array2();
  inline T& operator()(int d1, int d2) const
  {
    return objs[d1][d2];
  }
  Array2<T>& operator=(const Array2&);
  inline int dim1() const {return dm1;}
  inline int dim2() const {return dm2;}
  void resize(int, int);
  void initialize(const T&);
  
  inline T** get_ptr_to_row(int d1) { return &(objs[d1]); } 
  inline T* get_dataptr() {return objs[0];}
  inline unsigned long get_datasize() {
    return dm1*dm2*sizeof(T);
  }
  void share(const Array2<T>& copy);


  //friend template<int> void Pio(SCIRun::Piostream&, rtrt::Array2<int>& data);
  //friend template<float> void Pio(SCIRun::Piostream&, 
  //			  rtrt::Array2<float>& data);
//friend template<double> void Pio(SCIRun::Piostream&, 
  //			   rtrt::Array2<double>& data);
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (SCIRun::Piostream&, Array2<T>&);
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (SCIRun::Piostream&, Array2<T>*&);

};

} // end namespace rtrt

#include <Packages/rtrt/Core/Array2.cc>



#endif // Package_rtrt_Core_Array2_h
