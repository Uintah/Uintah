//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : sci_hash_map.h
//    Author : Wayne Witzel
//    Date   : Jul 7 2002
//
// Some ports don't implement the std algorithms we use, so we
// implement them here on our own.

#include <algorithm>

#if defined(__digital__) || defined(_AIX) \
   || HAVE_EXT_ALGORITHM // forget about using extension library
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
