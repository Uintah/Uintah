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
//    File   : sci_hash_set.h
//    Author : Dav de St. Germain
//    Date   : Fri May  3 19:38:43 EDT 2002

// getting  around different include locations for different 
// versions of the STL extension hash_set

#if !defined SCI_HASH_SET_H
#define SCI_HASH_SET_H

#define HAVE_HASH_SET
#ifdef HAVE_STD_HASHMAP
#include <hash_set>
using std::hash_set;
using std::hash_multiset;
#else
#ifdef HAVE_EXT_HASHMAP
#include <ext/hash_set>
using std::hash_set;
using std::hash_multiset;
#else
#ifdef HAVE_GNU_HASHMAP
#include <ext/hash_set>
using __gnu_cxx::hash_set;
using __gnu_cxx::hash_multiset;
#else
#undef HAVE_HASH_SET
#endif
#endif
#endif

#endif // SCI_Hash_set_h
