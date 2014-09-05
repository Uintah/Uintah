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
 *  pidl_cast.h: The equivalent of dynamic_cast for network based objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_PIDL_pidl_cast_h
#define CCA_PIDL_pidl_cast_h

#include <Core/CCA/PIDL/Object.h>
#include <Core/CCA/PIDL/TypeInfo.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/Exceptions/InternalError.h>

// To get rid of:
//cc-1424 CC: REMARK File = ../../../Core/CCA/PIDL/pidl_cast.h, Line = 36
//  The template parameter "T" is not used in declaring the argument types of
//          function template "pidl_cast".
//  template<class T>
//                 ^

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424
#endif

/**************************************
 
CLASS
   pidl_cast
   
KEYWORDS
   pidl_cast, dynamic_cast
   
DESCRIPTION
   An equivalent to dynamic_cast for (possibly) remote objects.  This
   tries to upcast locally, but will send a remote procedure call
   (in TypeInfo::pidl_cast) if required.
****************************************/

// Some compilers (gcc 3 at least) cannot find this unless it is in the
// global namespace
template<class T, class F>
T
pidl_cast(const F& ptr)
{
  typedef typename T::element_type ptr_type;
  // Try the direct cast before we go remote
  ptr_type* iface =
    dynamic_cast<ptr_type* >(ptr.getPointer());
  if(iface)
    return T(iface);
  const SCIRun::TypeInfo* typeinfo = ptr_type::_static_getTypeInfo();
  SCIRun::Object* result=typeinfo->pidl_cast(ptr.getPointer());
  if(result){
    ptr_type* p=dynamic_cast<ptr_type*>(result);
    if(!p)
      throw SCIRun::InternalError("TypeInfo::pidl_cast returned wrong object!");
#ifdef HAVE_MPI
    if(!(SCIRun::PIDL::sampleProxy)) {
      SCIRun::PIDL::optr = ((SCIRun::Object::pointer*) new T(p));
      SCIRun::PIDL::sampleProxy = true;
    } 
#endif
    return T(p);
  } else {
    return T(0);
  }
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif

#endif

