
#ifndef UINTAH_HOMEBREW_Array3Data_H
#define UINTAH_HOMEBREW_Array3Data_H

#include "RefCounted.h"

template<class T>
class Array3Data : public RefCounted {
public:
    Array3Data(int size1, int size2, int size3);
    virtual ~Array3Data();

    inline int dim1() {
	return size1;
    }
    inline int dim2() {
	return size2;
    }
    inline int dim3() {
	return size3;
    }
    void initialize(const T& val, int o1, int o2, int o3,
		    int s1, int s2, int s3);
    inline T& get(int i, int j, int k) {
	//ASSERTRANGE(i, 0, size1);
	//ASSERTRANGE(j, 0, size2);
	//ASSERTRANGE(k, 0, size3);
#if 0
	int idx = i*size3*size2+j*size3+k;
	ASSERTRANGE(idx, 0, totalsize);
	return data[idx];
#else
	return data3[i][j][k];
#endif
    }
private:
    T* data;
    T*** data3;
    int size1, size2, size3;
    int totalsize;
    Array3Data& operator=(const Array3Data&);
    Array3Data(const Array3Data&);
};

template<class T>
void Array3Data<T>::initialize(const T& val,
			       int off1, int off2, int off3,
			       int s1, int s2, int s3)
{
    ASSERTRANGE(off1, 0, size1);
    ASSERTRANGE(off2, 0, size2);
    ASSERTRANGE(off3, 0, size3);
    ASSERTRANGE(s1, 1, size1-off1+1);
    ASSERTRANGE(s2, 1, size2-off2+1);
    ASSERTRANGE(s3, 1, size3-off3+1);
    T* d=data + off1*size3*size2 + off2*size3 + off3;
    for(int i=0;i<s1;i++){
	T* dd=d;
	for(int j=0;j<s2;j++){
	    T* ddd=dd;
	    for(int j=0;j<s3;j++)
		ddd[j]=val;
	    dd+=size3;
	}
	d+=size3*size2;
    }
}

template<class T>
Array3Data<T>::Array3Data(int size1, int size2, int size3)
    : size1(size1), size2(size2), size3(size3)
{
    if(size1 && size2 && size3)
	data=new T[size1*size2*size3];
    else
	data=0;
    totalsize = size1*size2*size3;
    data3=new T**[size1];
    data3[0]=new T*[size1*size2];
    data3[0][0]=data;
    for(int i=1;i<size1;i++){
	data3[i]=data3[i-1]+size2;
    }
    for(int j=1;j<size1*size2;j++){
	data3[0][j]=data3[0][j-1]+size3;
    }
}

template<class T>
Array3Data<T>::~Array3Data()
{
    if(data){
	delete[] data;
	delete[] data3;
    }
}

#endif
