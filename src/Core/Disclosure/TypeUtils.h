/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#ifndef UINTAH_HOMEBREW_TypeUtils_H
#define UINTAH_HOMEBREW_TypeUtils_H

#include <sci_defs/compile_defs.h> // for STATIC_BUILD

#include <inttypes.h>
#include <cfloat>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>

namespace Uintah {

class  Matrix3;
class  Short27;
class  Int130;
class  TypeDescription;
struct Stencil7;
struct Stencil4;

typedef int64_t long64;

 const TypeDescription* fun_getTypeDescription(bool*);
 const TypeDescription* fun_getTypeDescription(unsigned char*);
 const TypeDescription* fun_getTypeDescription(int*);
 const TypeDescription* fun_getTypeDescription(short int*);
// const TypeDescription* fun_getTypeDescription(long*);
 const TypeDescription* fun_getTypeDescription(long64*);
 const TypeDescription* fun_getTypeDescription(double*);
 const TypeDescription* fun_getTypeDescription(float*);
 const TypeDescription* fun_getTypeDescription(Point*);
 const TypeDescription* fun_getTypeDescription(Vector*);
 const TypeDescription* fun_getTypeDescription(IntVector*);
 const TypeDescription* fun_getTypeDescription(Matrix3*);
 const TypeDescription* fun_getTypeDescription(Short27*);
 const TypeDescription* fun_getTypeDescription(Int130*);
 const TypeDescription* fun_getTypeDescription(Stencil7*);
 const TypeDescription* fun_getTypeDescription(Stencil4*);
 
 const TypeDescription* fun_getTypeDescription(FILE**);



// these functions are for getting safe values of types
// return back the value in the argument (so we don't have to include
// Vector.h here)
 void fun_getLargeValue(double*);
 void fun_getSmallValue(double*);
 void fun_getZeroValue(double*);
 void fun_getZeroValue(bool*);
 void fun_getZeroValue(int64_t*);
 void fun_getZeroValue(Vector*);
 void fun_getZeroValue(IntVector*);

// these functions should never get called - they just exist for
// template completeness
 void fun_getLargeValue(bool*);
 void fun_getSmallValue(bool*);
 void fun_getLargeValue(long64*);
 void fun_getSmallValue(long64*);
 void fun_getLargeValue(Vector*);
 void fun_getSmallValue(Vector*);
 void fun_getLargeValue(IntVector*);
 void fun_getSmallValue(IntVector*);

} // End namespace Uintah
   
#include <Core/Datatypes/TypeName.h>
#include <string>

#if !defined( STATIC_BUILD )

namespace Uintah {

  template<>  const std::string find_type_name(Uintah::long64*);

   const TypeDescription* get_type_description(Uintah::long64*);

} // namespace Uintah 
#endif // STATIC_BUILD

#endif
