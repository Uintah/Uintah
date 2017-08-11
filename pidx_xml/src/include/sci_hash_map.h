/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
//    File   : sci_hash_map.h
//    Author : Martin Cole
//    Date   : Sat Jul 21 10:17:44 2001

// getting  around different include locations for different 
// versions of the STL extension hash_map

#if !defined SCI_Hash_Map_h
#define SCI_Hash_Map_h

#include <sci_defs/hashmap_defs.h>

#define HAVE_HASH_MAP


#if defined(HAVE_C11_HASHMAP)

#  include <unordered_map>

   template<typename A, typename B, typename C> class hash_map : public std::unordered_map<A, B, C> {
      public:
      hash_map(int &n) : std::unordered_map<A,B,C>(n){}
   };
   template<typename A, typename B> class hash_multimap : public std::unordered_multimap<A, B> {};
   template<typename A, typename B> class hashmap : public std::unordered_map<A, B> {};
   using std::hash;

#  undef HAVE_HASH_MAP

#endif
#endif // SCI_Hash_Map_h
