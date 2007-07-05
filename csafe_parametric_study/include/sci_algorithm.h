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

//    File   : sci_hash_map.h
//    Author : Wayne Witzel
//    Date   : Jul 7 2002
//
// Some ports don't implement the std algorithms we use, so we
// implement them here on our own.

#ifndef SCI_INCLUDE_ALGORITHM
#define SCI_INCLUDE_ALGORITHM

#include <algorithm>

#if !defined(REDSTORM) && (defined(__digital__) || defined(_AIX) || defined(__APPLE__) \
   || defined(__ECC) || defined(_MSC_VER) || HAVE_EXT_ALGORITHM ) // forget about using extension library
// AIX and DEC don't have this...X
namespace std {
  template <class Iter, class Compare>
  bool is_sorted(Iter begin, Iter end, Compare compare)
  {
    if(begin == end)
      return true;
    Iter cur = begin;
    Iter next = cur; next++;
    while(next != end){
      if (compare(*next, *cur))
	return false;
      cur = next;
      next++;
    }
    return true;
  }
  
  template <class Iter>
  bool is_sorted(Iter begin, Iter end)
  {
    if(begin == end)
      return true;
    Iter cur = begin;
    Iter next = cur; next++;
    while(next != end){
      if (*next < *cur)
	return false;
      cur = next;
      next++;
    }
    return true;
  }
}
#endif
#endif
