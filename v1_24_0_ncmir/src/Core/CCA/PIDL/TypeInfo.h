/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
  int isa(const std::string& classname) const;

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

