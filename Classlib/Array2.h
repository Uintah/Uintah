
/*
 *  Array2.h: Interface to dynamic 2D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Classlib_Array2_h
#define SCI_Classlib_Array2_h 1

#include <Classlib/Assert.h>
#include <Tester/RigorousTest.h>


#ifdef __GNUG__
#pragma interface
#endif

class Piostream;
/**************************************

CLASS
   Array2
   
KEYWORDS
   Array2

DESCRIPTION
    Array2.h: Interface to dynamic 2D array class
  
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
class Array2 {
    T** objs;
    int dm1;
    int dm2;
    void allocate();
public:
    //////////
    //Create a 0X0 Array
    Array2();
    
    //////////
    //Array2 Copy Constructor
    Array2(const Array2&);
    
    //////////
    //Create an n by n array
    Array2(int, int);

    Array2<T>& operator=(const Array2&);
    
    //////////
    //Class Destructor
    ~Array2();

    //////////
    //Used for accessing elements in the Array
    inline T& operator()(int d1, int d2) const
	{
	    ASSERTL3(d1>=0 && d1<dm1);
	    ASSERTL3(d2>=0 && d2<dm2);
	    return objs[d1][d2];
	}
    
    //////////
    //Returns number of rows
    inline int dim1() const {return dm1;}
    
    //////////
    //Returns number of cols
    inline int dim2() const {return dm2;}
    
    //////////
    //Resize Array
    void newsize(int, int);
    
    //////////
    //Initialize all values in an array
    void initialize(const T&);

    inline T** get_dataptr() {return objs;}

    
    //////////
    //Rigorous Tests
    static void test_rigorous(RigorousTest* __test);

    friend void Pio(Piostream&, Array2<T>&);
    friend void Pio(Piostream&, Array2<T>*&);
    


};


#endif














