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

#ifndef Component_PIDL_pidl_cast_h
#define Component_PIDL_pidl_cast_h

// In global namespace for now...
#include <Core/CCA/Component/PIDL/Object.h>
#include <Core/CCA/Component/PIDL/TypeInfo.h>
#include <Core/Exceptions/InternalError.h>

// To get rid of:
//cc-1424 CC: REMARK File = ../../../Core/CCA/Component/PIDL/pidl_cast.h, Line = 36
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

template<class T>
T
pidl_cast(const PIDL::Object& obj)
{
  // Try the direct cast before we go remote
  typename T::interfacetype * iface =
    dynamic_cast< typename T::interfacetype* >(obj.getPointer());
  if(iface)
    return iface;

  const PIDL::TypeInfo* typeinfo = T::_getTypeInfo();
  PIDL::Object_interface* result=typeinfo->pidl_cast(obj.getPointer());
  if(result){
    T p=dynamic_cast<typename T::interfacetype*>(result);
    if(!p)
      throw SCIRun::InternalError("TypeInfo::pidl_cast returned wrong object!");
    return p;
  } else {
    return 0;
  }
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif

#endif

