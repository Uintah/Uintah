/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



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

#include <Core/Persistent/Pstreams.h>
#include <sci_defs/template_defs.h>

namespace rtrt {
  template<class T> class Array2;
}

namespace SCIRun {
  template<class T> void Pio(SCIRun::Piostream& stream, rtrt::Array2<T>& data);
  template<class T> void Pio(SCIRun::Piostream& stream, rtrt::Array2<T>*& data);
}

namespace rtrt{

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
  friend void TEMPLATE_TAG SCIRun::Pio TEMPLATE_BOX (SCIRun::Piostream&, Array2<T>&);
  friend void TEMPLATE_TAG SCIRun::Pio TEMPLATE_BOX (SCIRun::Piostream&, Array2<T>*&);
};

} // end namespace rtrt

#include <Packages/rtrt/Core/Array2.cc>



#endif // Package_rtrt_Core_Array2_h
