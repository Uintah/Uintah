#ifndef UINTAH_HOMEBREW_Array3Window_H
#define UINTAH_HOMEBREW_Array3Window_H

#include "RefCounted.h"
#include "Array3Data.h"

namespace Uintah {
  namespace Grid {
    class RefCounted;
  }
}

using Uintah::Grid::RefCounted;

/**************************************

CLASS
   Array3Window
   
GENERAL INFORMATION

   Array3Window.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Array3Window

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

template<class T>
class Array3Window : public RefCounted {
public:
    Array3Window(Array3Data<T>*);
    virtual ~Array3Window();

    inline Array3Data<T>* getData() const {
	return data;
    }

    void initialize(const T&);
    void initialize(const T&, int sx, int sy, int sz, int ex, int ey, int ez);
    inline int getOff1() {
	return off1;
    }
    inline int getOff2() {
	return off2;
    }
    inline int getOff3() {
	return off3;
    }
    inline int dim1() {
	return s1;
    }
    inline int dim2() {
	return s2;
    }
    inline int dim3() {
	return s3;
    }
    inline T& get(int i, int j, int k) {
	//ASSERT(data);
	return data->get(i+off1, j+off2, k+off3);
    }

private:

    Array3Data<T>* data;
    int off1, off2, off3;
    int s1, s2, s3;
    Array3Window(const Array3Window<T>&);
    Array3Window<T>& operator=(const Array3Window<T>&);
};

template<class T>
void Array3Window<T>::initialize(const T& val)
{
    data->initialize(val, off1, off2, off3, s1, s2, s3);
}

template<class T>
void Array3Window<T>::initialize(const T& val, int sx, int sy, int sz,
				 int ex, int ey, int ez)
{
    data->initialize(val, sx+off1, sy+off2, sz+off3,
		     ex-sx, ey-sy, ez-sz);
}

template<class T>
Array3Window<T>::Array3Window(Array3Data<T>* data)
    : data(data), off1(0), off2(0), off3(0),
      s1(data->dim1()), s2(data->dim2()), s3(data->dim3())
{
    data->addReference();
}

template<class T>
Array3Window<T>::~Array3Window()
{
    if(data && data->removeReference())
	delete data;
}

//
// $Log$
// Revision 1.3  2000/03/21 02:22:57  dav
// few more updates to make it compile including moving Array3 stuff out of namespace as I do not know where it should be
//
// Revision 1.2  2000/03/16 22:07:58  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
