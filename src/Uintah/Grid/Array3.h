
#ifndef UINTAH_HOMEBREW_Array3_H
#define UINTAH_HOMEBREW_Array3_H

#include "Array3Window.h"
#include "Array3Index.h"
#include <iostream> // TEMPORARY

template<class T>
class Array3 {
    Array3Window<T>* window;
public:
    Array3() {
	window = 0;
    }
    Array3(int size1, int size2, int size3) {
	window=new Array3Window<T>(new Array3Data<T>(size1, size2, size3));
	window->addReference();
    }
    virtual ~Array3();
    Array3(const Array3& copy)
	: window(copy.window)
    {
	if(window)
	    window->addReference();
    }

    Array3& operator=(const Array3& copy) {
	if(copy.window)
	    copy.window->addReference();
	if(window && window->removeReference()){
	    delete window;
	}
	window = copy.window;
	return *this;
    }

    int dim1() const {
	return window->dim1();
    }
    int dim2() const {
	return window->dim2();
    }
    int dim3() const {
	return window->dim3();
    }
    T& operator()(int i, int j, int k) const {
	//ASSERT(window);
	return window->get(i, j, k);
    }
    void initialize(const T& value) {
	window->initialize(value);
    }

    void initialize(const T& value, int sx, int sy, int sz, int ex, int ey, int ez) {
	window->initialize(value, sx, sy, sz, ex, ey, ez);
    }

    void resize(int size1, int size2, int size3) {
	if(window){
	    Array3Data<T>* data = window->getData();
	    if(data && size1 == data->dim1() && size2 == data->dim2() && size3 == data->dim3())
		return;
	}
	if(window && window->removeReference())
	    delete window;
	//std::cerr << "Creating array: " << size1 << "x" << size2 << "x" << size3 << " (size " << sizeof(T) << ")\n";
	window=new Array3Window<T>(new Array3Data<T>(size1, size2, size3));
	window->addReference();
    }
    T& operator[](const Array3Index& idx) const {
	return window->get(idx.i(), idx.j(), idx.k());
    }

    inline Array3Window<T>* getWindow() const {
	return window;
    }
    T& operator[](const Array3Index& idx) {
	return window->get(idx.i(), idx.j(), idx.k());
    }
};

template<class T>
Array3<T>::~Array3()
{
    if(window && window->removeReference()){
	delete window;
    }
}

#endif
