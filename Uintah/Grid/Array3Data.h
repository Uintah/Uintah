#ifndef UINTAH_HOMEBREW_ARRAY3DATA_H
#define UINTAH_HOMEBREW_ARRAY3DATA_H

#include "RefCounted.h"

namespace Uintah {
namespace Grid {

/**************************************

CLASS
   Array3Data
   
GENERAL INFORMATION

   Array3Data.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Array3Data

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

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
	int idx = i*d_size3*d_size2+j*d_size3+k;
	ASSERTRANGE(idx, 0, d_totalsize);
	return d_data[idx];
#else
	return d_data3[i][j][k];
#endif
    }
private:
    T*    d_data;
    T***  d_data3;
    int   d_size1, d_size2, d_size3;
    int   d_totalsize;

    Array3Data& operator=(const Array3Data&);
    Array3Data(const Array3Data&);
};

template<class T>
void Array3Data<T>::initialize(const T& val,
			       int off1, int off2, int off3,
			       int s1, int s2, int s3)
{
    ASSERTRANGE(off1, 0, d_size1);
    ASSERTRANGE(off2, 0, d_size2);
    ASSERTRANGE(off3, 0, d_size3);
    ASSERTRANGE(s1, 1, d_size1-off1+1);
    ASSERTRANGE(s2, 1, d_size2-off2+1);
    ASSERTRANGE(s3, 1, d_size3-off3+1);
    T* d = d_data + off1*d_size3*d_size2 + off2*d_size3 + off3;
    for(int i=0;i<s1;i++){
	T* dd=d;
	for(int j=0;j<s2;j++){
	    T* ddd=dd;
	    for(int j=0;j<s3;j++)
		ddd[j]=val;
	    dd+=d_size3;
	}
	d+=d_size3*d_size2;
    }
}

template<class T>
Array3Data<T>::Array3Data(int size1, int size2, int size3)
    : d_size1(size1), d_size2(size2), d_size3(size3)
{
    if(d_size1 && d_size2 && d_size3)
	data=new T[d_size1*d_size2*d_size3];
    else
	data=0;
    totalsize = d_size1*d_size2*d_size3;
    data3=new T**[d_size1];
    data3[0]=new T*[d_size1*d_size2];
    data3[0][0]=d_data;
    for(int i=1;i<d_size1;i++){
	d_data3[i]=d_data3[i-1]+d_size2;
    }
    for(int j=1;j<d_size1*d_size2;j++){
	d_data3[0][j]=d_data3[0][j-1]+d_size3;
    }
}

template<class T>
Array3Data<T>::~Array3Data()
{
    if(data){
	delete[] d_data;
	delete[] d_data3;
    }
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/03/21 01:29:42  dav
// working to make MPM stuff compile successfully
//
// Revision 1.2  2000/03/16 22:07:58  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
