
/*
 *  BrickArray3.h: Interface to dynamic 3D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Classlib_BrickArray3_h
#define SCI_Classlib_BrickArray3_h 1

#include <iostream>

namespace rtrt {

using namespace std;

template<class T>
class BrickArray3 {
    T* objs;
    char* data;
    int* refcnt;
    int* idx1;
    int* idx2;
    int* idx3;
    int dm1;
    int dm2;
    int dm3;
    int totaldm1;
    int totaldm2;
    int totaldm3;
    int L1, L2;
    void allocate();
    BrickArray3<T>& operator=(const BrickArray3&);
public:
    BrickArray3();
    BrickArray3(int, int, int);
    ~BrickArray3();
    inline T& operator()(int d1, int d2, int d3) const
        {
	  //cerr << "d1 = " << d1 << "  d2 = " << d2 << "  d3 = " << d3 << endl;
	  //cerr << "idx1[d1] = " << idx1[d1];
	  //cerr << "idx2[d1] = " << idx2[d1];
	  //cerr << "idx3[d1] = " << idx3[d1] << endl;
            return objs[idx1[d1]+idx2[d2]+idx3[d3]];
        }
    inline int dim1() const {return dm1;}
    inline int dim2() const {return dm2;}
    inline int dim3() const {return dm3;}
    void resize(int, int, int);
    void initialize(const T&);

    inline T* get_dataptr() {return objs;}
    inline unsigned long get_datasize() {
	return totaldm1*totaldm2*totaldm3*sizeof(T);
    }
    void share(const BrickArray3<T>& copy);
};

#include <Packages/rtrt/Core/BrickArray3.cc>

} // end namespace rtrt

#endif
