
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

#ifndef SCI_Classlib_Array3_h
#define SCI_Classlib_Array3_h 1

#include <Classlib/Assert.h>

#ifdef __GNUG__
#pragma interface
#endif

class Piostream;
class RigorousTest;

/**************************************

CLASS
   Array3
   
KEYWORDS
   Array3

DESCRIPTION
    Array3.h: Interface to dynamic 3D array class
  
    Written by:
     Steven G. Parker
     Department of Computer Science
     University of Utah
     March 1994
  
    Copyright (C) 1994 SCI Group
PATTERNS
   
WARNING
  
****************************************/

template<class T>
class Array3 {
    T*** objs;
    int dm1;
    int dm2;
    int dm3;
    void allocate();
public:
    //////////
    //Default Constructor
    Array3();
    
    //////////
    //Copy Constructor
    Array3(const Array3&);

    //////////
    //Constructor
    Array3(int, int, int);
    
    //////////
    //Assignment Operator
    Array3<T>& operator=(const Array3&);
    
    //////////
    //Class Destructor
    ~Array3();
    
    //////////
    //Access the nXnXn element of the array
    inline T& operator()(int d1, int d2, int d3) const
	{
	    ASSERTL3(d1>=0 && d1<dm1);
	    ASSERTL3(d2>=0 && d2<dm2);
	    ASSERTL3(d3>=0 && d3<dm3);
	    return objs[d1][d2][d3];
	}
    
    //////////
    //Returns the number of spaces in dim1	    
    inline int dim1() const {return dm1;}
    //////////
    //Returns the number of spaces in dim2
    inline int dim2() const {return dm2;}
    //////////
    //Returns the number of spaces in dim3
    inline int dim3() const {return dm3;}
    
    //////////
    //Re-size the Array
    void newsize(int, int, int);

    //////////
    //Initialize all elements to T
    void initialize(const T&);

    T* get_onedim();
    void get_onedim_byte( unsigned char *v );

    inline T*** get_dataptr() {return objs;}

    //////////
    //Rigorous Tests
    static void test_rigorous(RigorousTest* __test);
    
    friend void Pio(Piostream&, Array3<T>&);
    friend void Pio(Piostream&, Array3<T>*&);

};


#endif
