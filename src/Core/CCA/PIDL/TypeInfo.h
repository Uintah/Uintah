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
 *  TypeInfo.h: internal representation for a type.
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_PIDL_TypeInfo_h
#define CCA_PIDL_TypeInfo_h

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

class NexusEpChannel;

namespace SCIRun {

class TypeInfo_internal;
class Object;

/**************************************

  CLASS
	 TypeInfo

  KEYWORDS
	 Introspection, PIDL
   
  DESCRIPTION
	 PIDL private class which contains the type information for PIDLs
	 internal type system.  See TypeInfo_internal for the actual data.
	 This class is used in the public .h files, while the internals
	 are separate - to avoid include file pollution.
	 In the method descriptions below, "this class" refers to the class
	 associated with this TypeInfo record.
****************************************/

class TypeInfo {
public:
  //////////
  // Create the typeinfo object from the internal object.
  // This is used only in sidl generated code
  TypeInfo(TypeInfo_internal*);

  //////////
  // Destructor
  ~TypeInfo();

  //////////
  // Compute the new vtable offset for casting from object of
  // this type to a type of newtype.  Used only in sidl
  // generated code.
  int computeVtableOffset(const TypeInfo* newtype) const;

  //////////
  // The internal implementation of the pidl_cast template
  // function.  This is a remote dynamic cast facility that
  // cast the object obj to an object of this type.  It will
  // return 0 for failure, or will return an object that is
  // castable to a subclass of this class using dynamic_cast.
  Object* pidl_cast(Object* obj) const;

  //////////
  // Return the new vtable offset for casting to the named class.
  // This is used only by the sidl generated isa handler.
  int isa(const std::string& classname, const std::string& uuid) const;

  //////////
  // The number used to represent an invalid vtable offset.
  static const int vtable_invalid = -1234;

  //////////
  // The start of the methods in the vtable.  Do NOT change
  // this number unless the corresponding code generation is
  // changed in the sidl compiler.
  static const int vtable_methods_start = 3;

  //////////
  // The ID of the remote isa handler
  static const int vtable_isa_handler = 0;

  //////////
  // The ID of the remote delete reference handler
  static const int vtable_deleteReference_handler = 1;

protected:
private:

  //////////
  // TypeInfo_internal needs access to the vtable_isa_handler
  // constant
  friend class TypeInfo_internal;

  //////////
  // NexusEpChannel needs access to the private TypeInfo_internal
  friend class NexusEpChannel;

  //////////
  // ProxyBase needs access to the private handler numbers
  friend class ProxyBase;

  //////////
  // A pointer to the actual data.
  TypeInfo_internal* d_priv;
};

} // End namespace SCIRun

#endif

