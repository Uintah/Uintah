/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/share/share.h>

namespace SCIRun {

class RigorousTest;
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

} // End namespace SCIRun

#endif



