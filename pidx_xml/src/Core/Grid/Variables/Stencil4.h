/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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


#ifndef Packages_Uintah_Core_Grid_Stencil4_h
#define Packages_Uintah_Core_Grid_Stencil4_h

#include <Core/Disclosure/TypeUtils.h>
#include <Core/Util/FancyAssert.h>
#include <iostream>

namespace Uintah {
  class TypeDescription;
  struct  Stencil4 {
    // The order of this is designed to match the order of faces in Patch
    // Do not change it!
    //     -x +x -y +y -z +z
    double  w, s, b;
    // diagonal term
    double p;
    double& operator[](int index) {
      ASSERTRANGE(index, 0, 3);
      return (&w)[index];
    }
    const double& operator[](int index) const {
      ASSERTRANGE(index, 0, 3);
      return (&w)[index];
    }
  };
  
   std::ostream & operator << (std::ostream &out, const Uintah::Stencil4 &a);
  
}

namespace Uintah {
   void swapbytes( Uintah::Stencil4& );
} // namespace Uintah

#endif
