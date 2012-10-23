/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
//    File   : sci_hash_set.h
//    Author : Dav de St. Germain
//    Date   : Fri May  3 19:38:43 EDT 2002

// getting  around different include locations for different 
// versions of the STL extension hash_set

#if !defined SCI_HASH_SET_H
#define SCI_HASH_SET_H

#include <sci_defs/hashmap_defs.h>

#define HAVE_HASH_SET

#ifdef HAVE_STD_HASHMAP
#  include <hash_set>
   using std::hash_set;
   using std::hash_multiset;
#elif defined(HAVE_EXT_HASHMAP)
#  include <ext/hash_set>
   using std::hash_set;
   using std::hash_multiset;
#elif defined(HAVE_TR1_HASHMAP)
#  include <tr1/unordered_set>
#  define hash_set std::tr1::unordered_set
#  define hash_multiset std::tr1::unordered_multiset
#elif defined(HAVE_GNU_HASHMAP)
#  include <ext/hash_set>
   using __gnu_cxx::hash_set;
   using __gnu_cxx::hash_multiset;
#elif defined(HAVE_STDEXT_HASHMAP)
#  include <hash_set>
   using stdext::hash_set;
   using stdext::hash_multiset;
#else
#  undef HAVE_HASH_SET
#endif

#endif // SCI_HASH_SET_H
