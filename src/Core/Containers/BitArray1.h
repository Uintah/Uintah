
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

#ifndef SCI_Containers_BitArray1_h
#define SCI_Containers_BitArray1_h 1

#include <SCICore/share/share.h>

#include <SCICore/Tester/RigorousTest.h>

namespace SCICore {
namespace Containers {

using SCICore::Tester::RigorousTest;

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

class SCICORESHARE BitArray1 {
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

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:35  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:12  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:42  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:29  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//

#endif
