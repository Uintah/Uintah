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
 *  Reference.h: A serializable "pointer" to an object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Component_PIDL_Reference_h
#define Component_PIDL_Reference_h

#include <Core/CCA/Component/Comm/SpChannel.h>

namespace SCIRun {
/**************************************
 
CLASS
   Reference
   
KEYWORDS
   Reference, PIDL
   
DESCRIPTION
   A remote reference.  This class is internal to PIDL and should not
   be used outside of PIDL or sidl generated code.  It contains a nexus
   startpoint and the vtable base offset.
****************************************/
  struct Reference {
    //////////
    // Empty constructor.  Initalizes the startpoint to nil
    Reference();

    //////////
    // Copy the reference. 
    Reference(const Reference&);

    //////////
    // Copy the reference.  
    Reference& operator=(const Reference&);

    //////////
    // Destructor. 
    ~Reference();

    //////////
    // Return the vtable base
    int getVtableBase() const;

    //////////
    // Proxy's communication class. 
    SpChannel * chan;

    //////////
    // The vtable base offset
    int d_vtable_base;
  };
} // End namespace SCIRun

#endif









