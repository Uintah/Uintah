#ifndef UINTAH_HOMEBREW_ARRAY3DATA_H
#define UINTAH_HOMEBREW_ARRAY3DATA_H

#include "RefCounted.h"

namespace Uintah {
  namespace Grid {
    class RefCounted;
  }
}

using Uintah::Grid::RefCounted;

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
	return d_size1;
    }
    inline int dim2() {
	return d_size2;
    }
    inline int dim3() {
	return d_size3;
    }
    void initialize(const T& val, int o1, int o2, int o3,
		    int s1, int s2, int s3);
    inline T& get(int i, int j, int k) {
	//CHECKARRAYBOUNDS(i, 0, size1);
	//CHECKARRAYBOUNDS(j, 0, size2);
	//CHECKARRAYBOUNDS(k, 0, size3);
#if 0
	int idx = i*d_size3*d_size2+j*d_size3+k;
	CHECKARRAYBOUNDS(idx, 0, d_totalsize);
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
    CHECKARRAYBOUNDS(off1, 0, d_size1);
    CHECKARRAYBOUNDS(off2, 0, d_size2);
    CHECKARRAYBOUNDS(off3, 0, d_size3);
    CHECKARRAYBOUNDS(s1, 1, d_size1-off1+1);
    CHECKARRAYBOUNDS(s2, 1, d_size2-off2+1);
    CHECKARRAYBOUNDS(s3, 1, d_size3-off3+1);
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
	d_data=new T[d_size1*d_size2*d_size3];
    else
	d_data=0;
    d_totalsize = d_size1*d_size2*d_size3;
    d_data3=new T**[d_size1];
    d_data3[0]=new T*[d_size1*d_size2];
    d_data3[0][0]=d_data;
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
    if(d_data){
	delete[] d_data;
	delete[] d_data3;
    }
}

//
// $Log$
// Revision 1.5  2000/03/22 23:41:27  sparker
// Working towards getting arches to compile/run
//
// Revision 1.4  2000/03/21 02:22:57  dav
// few more updates to make it compile including moving Array3 stuff out of namespace as I do not know where it should be
//
// Revision 1.3  2000/03/21 01:29:42  dav
// working to make MPM stuff compile successfully
//
// Revision 1.2  2000/03/16 22:07:58  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
