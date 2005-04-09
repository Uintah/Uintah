
/*
 *  Array3.h: Interface to dynamic 3D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Datatypes_Array3_h
#define SCI_Datatypes_Array3_h 1

#include <Classlib/Assert.h>
#include <Datatypes/Datatype.h>
class Piostream;

template<class T>
class LockArray3:public Datatype {
    T*** objs;
    int dm1;
    int dm2;
    int dm3;
    void allocate();
public:
    LockArray3();
    LockArray3(const LockArray3&);
    LockArray3(int, int, int);
    LockArray3<T>& operator=(const LockArray3&);
    virtual ~LockArray3();

    LockArray3<T>* clone() const;

    inline T& operator()(int d1, int d2, int d3) const
	{
	    ASSERTL3(d1>=0 && d1<dm1);
	    ASSERTL3(d2>=0 && d2<dm2);
	    ASSERTL3(d3>=0 && d3<dm3);
	    return objs[d1][d2][d3];
	}
    inline int dim1() const {return dm1;}
    inline int dim2() const {return dm2;}
    inline int dim3() const {return dm3;}
    void newsize(int, int, int);
    void initialize(const T&);

    T* get_onedim();
    void get_onedim_byte( unsigned char *v );

    inline T*** get_dataptr() {return objs;}

    friend void Pio(Piostream&, LockArray3<T>&);
    friend void Pio(Piostream&, LockArray3<T>*&);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


#endif
