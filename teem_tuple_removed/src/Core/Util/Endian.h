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

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>
#if HAVE_INTTYPES_H
#include <inttypes.h>
#endif
#if HAVE_STDINT_H
#include <stdint.h>
#endif

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
