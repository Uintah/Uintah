
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

#ifndef Component_PIDL_TypeInfo_h
#define Component_PIDL_TypeInfo_h

#include <string>

namespace PIDL {

class TypeInfo_internal;
class Object_interface;

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
  Object_interface* pidl_cast(Object_interface* obj) const;

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

protected:
private:

  //////////
  // The ID of the remote isa handler
  static const int vtable_isa_handler = 0;

  //////////
  // The ID of the remote delete reference handler
  static const int vtable_deleteReference_handler = 1;

  //////////
  // TypeInfo_internal needs access to the vtable_isa_handler
  // constant
  friend class TypeInfo_internal;

  //////////
  // ServerContext needs access to the private TypeInfo_internal
  friend class ServerContext;

  //////////
  // ProxyBase needs access to the private handler numbers
  friend class ProxyBase;

  //////////
  // A pointer to the actual data.
  TypeInfo_internal* d_priv;
};

} // End namespace PIDL

#endif

