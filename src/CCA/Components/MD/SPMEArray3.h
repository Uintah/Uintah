/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  Array3.h: Interface to dynamic 3D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 */

#ifndef UINTAH_MD_ELECTROSTATICS_SPME_ARRAY3_H
#define UINTAH_MD_ELECTROSTATICS_SPME_ARRAY3_H

#include <Core/Util/Assert.h>

#include <iostream>
#include <stdio.h>

#include <sci_defs/fftw_defs.h>

namespace Uintah {

using namespace SCIRun;

typedef std::complex<double> dblcomplex;

//template<class T> class SPMEArray3;

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

 PATTERNS

 WARNING

 ****************************************/

template<class T> class SPMEArray3 {
    T* objs;
    int dm1;
    int dm2;
    int dm3;
    void allocate();

    // The copy constructor and the assignment operator have been
    // privatized on purpose -- no one should use these.  Instead,
    // use the default constructor and the copy method.
    //////////
    //Copy Constructor
    SPMEArray3(const SPMEArray3&);
    //////////
    //Assignment Operator
    SPMEArray3<T>& operator=(const SPMEArray3&);
  public:
    //////////
    //Default Constructor
    SPMEArray3();

    //////////
    //Constructor
    SPMEArray3(int,
               int,
               int);

    //////////
    //Class Destructor
    virtual ~SPMEArray3();

    //////////
    //Access the nXnXn element of the array
    inline T& operator()(int d1,
                         int d2,
                         int d3) const
    {
      ASSERTL3(d1>=0 && d1<dm1);
      ASSERTL3(d2>=0 && d2<dm2);
      ASSERTL3(d3>=0 && d3<dm3);
      int idx = (d1) + ((d2) * dm1) + ((d3) * dm2 * dm3);
      return objs[idx];
    }

    //////////
    //Array3 Copy Method
    void copy(const SPMEArray3&);

    //////////
    //Returns the number of spaces in dim1
    inline int dim1() const
    {
      return dm1;
    }
    //////////
    //Returns the number of spaces in dim2
    inline int dim2() const
    {
      return dm2;
    }
    //////////
    //Returns the number of spaces in dim3
    inline int dim3() const
    {
      return dm3;
    }

    inline long get_datasize() const
    {
      return dm1 * long(dm2 * dm3 * sizeof(T));
    }

    //////////
    //Re-size the Array
    void resize(int,
                int,
                int);

    //////////
    //Initialize all elements to T
    void initialize(const T&);

    inline T* get_dataptr()
    {
      return objs;
    }

};

}  // End namespace Uintah

#endif

