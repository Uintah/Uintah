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
//    File   : sci_hash_map.h
//    Author : Martin Cole
//    Date   : Sat Jul 21 10:17:44 2001

// getting  around different include locations for different 
// versions of the STL extension hash_map

#if !defined SCI_Hash_Map_h
#define SCI_Hash_Map_h

#include <sci_defs/hashmap_defs.h>

#define HAVE_HASH_MAP
#if defined(HAVE_STD_HASHMAP)
#  include <hash_map>
   using std::hash_map;
   using std::hash_multimap;
#  ifndef _MSC_VER
   // msvc hash map is a little different
     using std::hash;
#  else
     using std::hash_compare; // MS VC 7
#  endif
#elif defined(HAVE_EXT_HASHMAP)
#  include <ext/hash_map>
   using std::hash_map;
   using std::hash_multimap;
   using std::hash;
#elif defined(HAVE_TR1_HASHMAP)
#  include <tr1/unordered_map>
template<typename A, typename B, typename C> class hash_map : public std::tr1::unordered_map<A, B, C> {
   public:
   hash_map(int &n) : std::tr1::unordered_map<A,B,C>(n){}
};
   template<typename A, typename B> class hash_multimap : public std::tr1::unordered_multimap<A, B> {};
   using std::tr1::hash;
#elif defined(HAVE_GNU_HASHMAP)
#  include <ext/hash_map>
   using __gnu_cxx::hash_map;
   using __gnu_cxx::hash_multimap;
   using __gnu_cxx::hash;
#elif defined(HAVE_STDEXT_HASHMAP)
#  include <hash_map>
   using stdext::hash_map;
   using stdext::hash_multimap;
   using stdext::hash_compare; // MS VC 8
#else
#  undef HAVE_HASH_MAP
#endif

#endif // SCI_Hash_Map_h
