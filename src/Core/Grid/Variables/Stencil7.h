/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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


#ifndef Packages_Uintah_Core_Grid_Stencil7_h
#define Packages_Uintah_Core_Grid_Stencil7_h

#include <Core/Disclosure/TypeUtils.h>
#include <Core/Util/FancyAssert.h>

#include <sci_defs/gpu_defs.h>

#include <iostream>

namespace Uintah {
  class TypeDescription;
  struct Stencil7 {
    // The order of this is designed to match the order of faces in Patch
    // Do not change it!
    //     -x +x -y +y -z +z
    double  w, e, s, n, b, t;
    // diagonal term
    double p;

    GPU_INLINE_FUNCTION double& operator[](int index) {
      // ASSERTRANGE(index, 0, 7);
      return (&w)[index];
    }

    GPU_INLINE_FUNCTION const double& operator[](int index) const {
      // ASSERTRANGE(index, 0, 7);
      return (&w)[index];
    }

    GPU_INLINE_FUNCTION void initialize(double a) {
      w = a;
      e = a;
      s = a;
      n = a;
      b = a;
      t = a;
      p = a;
    }

    // constructors
    GPU_INLINE_FUNCTION Stencil7() {}

    GPU_INLINE_FUNCTION Stencil7(double init) :
      w(init), e(init),
      s(init), n(init),
      b(init), t(init),
      p(init) {}
  };

  std::ostream & operator << (std::ostream &out, const Uintah::Stencil7 &a);

  void swapbytes( Uintah::Stencil7& );

} // namespace Uintah

#endif
