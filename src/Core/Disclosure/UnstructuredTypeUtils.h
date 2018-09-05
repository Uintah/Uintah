/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef UINTAH_HOMEBREW_UnstructuredTypeUtils_H
#define UINTAH_HOMEBREW_UnstructuredTypeUtils_H

#include <sci_defs/compile_defs.h> // for STATIC_BUILD

#include <inttypes.h>

#include <cfloat>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>

namespace Uintah {

class  Matrix3;
class  Short27;
class  UnstructuredTypeDescription;
struct Stencil7;
struct Stencil4;

typedef int64_t long64;

 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(bool*);
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(unsigned char*);
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(int*);
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(short int*);
// const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(long*);
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(long64*);
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(double*);
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(float*);
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(Point*);
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(Vector*);
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(IntVector*);

// THIS IS A GUESS -> Because of the order of instantiation of
// templates by the IBM xlC compiler, we can not declare the
// fun_getUnstructuredTypeDescription(Stencil7*) in Stencil7.h (where it probably
// should be.)  Instead we have to put it here.  I believe this is
// true for Matrix3 too.  However, both the fun_getUnstructuredTypeDescription of
// Matrix3 and Stencil7 are inplemented in their respective .cc files.

// since these are defined in Math/Grid, and declared here, we need to export them correctly
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(Matrix3*);
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(Short27*);

 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(Stencil7*);
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(Stencil4*);
 
 const UnstructuredTypeDescription* fun_getUnstructuredTypeDescription(FILE**);


#if 0
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

#endif

} // End namespace Uintah
   
#include <Core/Datatypes/TypeName.h>
#include <string>

#if !defined( STATIC_BUILD )

namespace Uintah {

#if 0
  template<>  const std::string find_type_name(Uintah::long64*);
#endif

   const UnstructuredTypeDescription* get_unstructured_type_description(Uintah::long64*);

} // namespace Uintah 
#endif // STATIC_BUILD

#endif
