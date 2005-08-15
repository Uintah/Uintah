/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  MiscMath.cc
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   June 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <Core/Math/MiscMath.h>
#include <math.h>
#ifdef __sgi
#include <ieeefp.h>
#endif
#ifdef __digital__
#include <fp_class.h>
#endif

namespace SCIRun {


#ifdef __sgi
double MakeReal(double value)
{
  if (!finite(value))
  {
    fpclass_t c = fpclass(value);
    if (c == FP_PINF)
    {
      value = (double)(0x7fefffffffffffffULL);
    }
    else if (c == FP_NINF)
    {
      value = (double)(0x0010000000000000ULL);
    }
    else
    {
      value = 0.0;
    }
  }      
  return value;
}
#else
double MakeReal(double value)
{
  if (!finite(value))
  {
#ifndef __digital__
    const int is_inf  = isinf(value);
#else
    // on dec, we have fp_class which can tell us if a number is +/- infinity
    int fpclass = fp_class(value);
    int is_inf;
    if (fpclass == FP_POS_INF) is_inf = 1;
    if (fpclass == FP_NEG_INF) is_inf = -1;
#endif
    if (is_inf == 1) value = (double)(0x7fefffffffffffffULL);
    if (is_inf == -1) value = (double)(0x0010000000000000ULL);
    else value = 0.0; // Assumed NaN
  }
  return value;
}
#endif

} // namespace SCIRun
