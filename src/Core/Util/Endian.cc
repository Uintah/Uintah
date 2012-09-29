/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <Core/Util/Endian.h>

#include <sci_defs/bits_defs.h> // for SCI_32BITS
#include <sci_defs/osx_defs.h>  // for OSX_SNOW_LEOPARD_OR_LATER

namespace SCIRun {

#if !defined( SCI_32BITS )
void swapbytes( long long& i ) { LONG_LONG_SWAP(i); }
#endif

void swapbytes( bool& )       { }
void swapbytes( int8_t& )     { }
void swapbytes( uint8_t& )    { }
void swapbytes( FILE* & )     { }
void swapbytes( int16_t& i )  { SWAP_2(i); }
void swapbytes( uint16_t& i ) { SWAP_2(i); }
void swapbytes( int32_t& i )  { SWAP_4(i); }
void swapbytes( uint32_t& i ) { SWAP_4(i); }
#if !defined( OSX_SNOW_LEOPARD_OR_LATER )
void swapbytes( int64_t& i )  { SWAP_8(i); }
#endif
void swapbytes( uint64_t& i ) { SWAP_8(i); }
void swapbytes( float& i )    { SWAP_4(i); }
void swapbytes( double& i )   { SWAP_8(i); }
void swapbytes( Point &i )    { // probably dangerous, but effective
                              double* p = (double *)(&i);
                              SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p); }
void swapbytes( Vector &i )   { // probably dangerous, but effective
                              double* p = (double *)(&i);
                              SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p); }

bool isBigEndian()
{
  short i = 0x4321;
  if((*(char *)&i) != 0x21 ){
    return true;
  } else {
    return false;
  }
}

bool isLittleEndian()
{
  return !isBigEndian();
}
 

string endianness()
{
  if( isBigEndian() ){
    return string("big_endian");
  } else {
    return string("little_endian");
  }
}

} // end namespace SCIRun
