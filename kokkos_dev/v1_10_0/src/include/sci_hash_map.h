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
//    Author : Martin Cole
//    Date   : Sat Jul 21 10:17:44 2001

// getting  around different include locations for different 
// versions of the STL extension hash_map

#if !defined SCI_Hash_Map_h
#define SCI_Hash_Map_h

#include <sci_defs.h>

#define HAVE_HASH_MAP
#ifdef HAVE_STD_HASHMAP
#include <hash_map>
using std::hash_map;
using std::hash_multimap;
using std::hash;
#else
#ifdef HAVE_EXT_HASHMAP
#include <ext/hash_map>
using std::hash_map;
using std::hash_multimap;
using std::hash;
#else
#ifdef HAVE_GNU_HASHMAP
#include <ext/hash_map>
using __gnu_cxx::hash_map;
using __gnu_cxx::hash_multimap;
using __gnu_cxx::hash;
#else
#undef HAVE_HASH_MAP
#endif
#endif
#endif

#endif // SCI_Hash_Map_h
