//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : Utils.cc
//    Author : Milan Ikits
//    Date   : Thu Jul 15 08:03:49 2004

#include <Core/Volume/Utils.h>

namespace SCIRun {

//---------------------------------------------------------------------------
// Power of 2 functions
//---------------------------------------------------------------------------

// Fast way to check for power of two
bool IsPowerOf2(uint n)
{
  return (n & (n-1)) == 0;
}

// Return power of two larger OR equal to n
uint NextPowerOf2(uint n)
{
  // if n is power of 2, return 
  if (IsPowerOf2(n)) return n;
  uint v;
  for(int i=31; i>=0; i--) {
    v = n & (1 << i);
    if (v) {
      v = (1 << (i+1));
      break;
    }
  }
  return v;
}

// Return largest power of two smaller OR equal to n
uint LargestPowerOf2(uint n)
{
  // if n is power of 2, return 
  if(IsPowerOf2(n)) return n;
  uint v;
  for(int i=31; i>=0; i--) {
    v = n & (1 << i);
    if (v) break;
  }
  return v;
}

} // namespace SCIRun
