
/*
 *  BitArray1.h: 1D bit array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Classlib_BitArray1_h
#define SCI_Classlib_BitArray1_h 1
#include <Tester/RigorousTest.h>
/**************************************

CLASS 
   BitArray1
   
KEYWORDS
   BitArray1

DESCRIPTION
   
 
   BitArray1.h: 1D bit array class
 
   Written by:
    Steven G. Parker
    Department of Computer Science
    University of Utah
    Feb. 1994
 
   Copyright (C) 1994 SCI Group
  
 
PATTERNS
   
WARNING
  
****************************************/

class BitArray1 {
    int size;
    int nbits;
    unsigned char* bits;
public:

    //////////
    //Class constructor (array size, and initial value)
    BitArray1(int size, int initial);
    ~BitArray1();
    
    //////////
    //Check to see if a bit is on or off
    int is_set(int);
    
    //////////
    //Set a bit
    void set(int);

    //////////
    //Clear a bit
    void clear(int);

    //////////
    //Set all bits in the array to 0
    void clear_all();

    //////////
    //Set all bits in the array to 1
    void set_all();

    //////////
    //Rigorous Tests
    static void test_rigorous(RigorousTest* __test);
    

};

#endif







