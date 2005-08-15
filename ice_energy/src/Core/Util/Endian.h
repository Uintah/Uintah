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
 *  NotFinished.h:  Consistent way to keep track of holes in the code...
 *
 *  Written by:
 *   Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Endianness_h
#define SCI_Endianness_h

#include <sci_defs/config_defs.h>

#if HAVE_INTTYPES_H
#  include <inttypes.h>
#endif

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun{
using std::string;

#define SWAP_2(u2)/* IronDoc macro to swap two byte quantity */ \
  { unsigned char* _p = (unsigned char*)(&(u2)); \
    unsigned char _c =   *_p; *_p = _p[1]; _p[1] = _c; }
#define SWAP_4(u4)/* IronDoc macro to swap four byte quantity */ \
  { unsigned char* _p = (unsigned char*)(&(u4)); \
    unsigned char  _c =   *_p; *_p = _p[3]; _p[3] = _c; \
                   _c = *++_p; *_p = _p[1]; _p[1] = _c; }
#define SWAP_8(u8)/* IronDoc macro to swap eight byte quantity */ \
  { unsigned char* _p = (unsigned char*)(&(u8)); \
    unsigned char  _c =   *_p; *_p = _p[7]; _p[7] = _c; \
                   _c = *++_p; *_p = _p[5]; _p[5] = _c; \
                   _c = *++_p; *_p = _p[3]; _p[3] = _c; \
                   _c = *++_p; *_p = _p[1]; _p[1] = _c; }

void swapbytes(bool& i);
void swapbytes(int8_t& i);
void swapbytes(uint8_t& i);
void swapbytes(int16_t& i);
void swapbytes(uint16_t& i);
void swapbytes(int32_t& i);
void swapbytes(uint32_t& i);
void swapbytes(int64_t& i);
void swapbytes(uint64_t& i);
// /////////////////////////////////////////////
void swapbytes(float& i);
//void swapbytes(int64_t& i){SWAP_8(i);}
void swapbytes(double& i);
void swapbytes(Point &i);
void swapbytes(Vector &i);

bool isBigEndian();

bool isLittleEndian();
 
string endianness();



} //end namespace SCIRun
#endif
