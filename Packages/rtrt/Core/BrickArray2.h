
/*
 *  BrickArray2.h: Interface to dynamic 2D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2000
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Classlib_BrickArray2_h
#define SCI_Classlib_BrickArray2_h

namespace rtrt {

template<class T>
class BrickArray2 {
    T* objs;
    char* data;
    int* refcnt;
    int* idx1;
    int* idx2;
    int dm1;
    int dm2;
    int totaldm1;
    int totaldm2;
    int L1, L2;
    void allocate();
    BrickArray2<T>& operator=(const BrickArray2&);
public:
    typedef T data_type;

    BrickArray2();
    BrickArray2(int, int);
    ~BrickArray2();
    inline T& operator()(int d1, int d2) const
        {
            return objs[idx1[d1]+idx2[d2]];
        }
    inline int dim1() const {return dm1;}
    inline int dim2() const {return dm2;}
    void resize(int, int);
    void initialize(const T&);

    inline T* get_dataptr() {return objs;}
    inline unsigned long get_datasize() {
	return totaldm1*totaldm2*sizeof(T);
    }
    void share(const BrickArray2<T>& copy);
};

} // end namespace rtrt

#endif
