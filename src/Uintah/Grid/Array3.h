#ifndef UINTAH_HOMEBREW_ARRAY3_H
#define UINTAH_HOMEBREW_ARRAY3_H

#include "Array3Window.h"
#include "Array3Index.h"
#include <iostream> // TEMPORARY

namespace Uintah {
namespace Grid {

/**************************************

CLASS
   Array3
   
GENERAL INFORMATION

   Array3.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Array3

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

template<class T>
class Array3 {

public:
    Array3() {
	d_window = 0;
    }
    Array3(int size1, int size2, int size3) {
	d_window=new Array3Window<T>(new Array3Data<T>(size1, size2, size3));
	d_window->addReference();
    }
    virtual ~Array3();
    Array3(const Array3& copy)
	: d_window(copy.d_window)
    {
	if(d_window)
	    d_window->addReference();
    }

    Array3& operator=(const Array3& copy) {
	if(copy.d_window)
	    copy.d_window->addReference();
	if(d_window && d_window->removeReference()){
	    delete d_window;
	}
	d_window = copy.d_window;
	return *this;
    }

    int dim1() const {
	return d_window->dim1();
    }
    int dim2() const {
	return d_window->dim2();
    }
    int dim3() const {
	return d_window->dim3();
    }
    T& operator()(int i, int j, int k) const {
	//ASSERT(d_window);
	return d_window->get(i, j, k);
    }
    void initialize(const T& value) {
	d_window->initialize(value);
    }

    void initialize(const T& value, int sx, int sy, int sz, int ex, int ey, int ez) {
	d_window->initialize(value, sx, sy, sz, ex, ey, ez);
    }

    void resize(int size1, int size2, int size3) {
	if(d_window){
	    Array3Data<T>* data = d_window->getData();
	    if(data && size1 == data->dim1() && size2 == data->dim2() && size3 == data->dim3())
		return;
	}
	if(d_window && d_window->removeReference())
	    delete d_window;
	//std::cerr << "Creating array: " << size1 << "x" << size2 << "x" << size3 << " (size " << sizeof(T) << ")\n";
	d_window=new Array3D_window<T>(new Array3Data<T>(size1, size2, size3));
	d_window->addReference();
    }
    T& operator[](const Array3Index& idx) const {
	return d_window->get(idx.i(), idx.j(), idx.k());
    }

    inline Array3Window<T>* getWindow() const {
	return d_window;
    }
    T& operator[](const Array3Index& idx) {
	return d_window->get(idx.i(), idx.j(), idx.k());
    }

private:
    Array3Window<T>* d_window;
};

template<class T>
Array3<T>::~Array3()
{
    if(d_window && d_window->removeReference()){
	delete d_window;
    }
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/16 22:07:57  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
