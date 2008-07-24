// 
// File:          scijump_TypeMismatchException_Impl.cxx
// Symbol:        scijump.TypeMismatchException-v0.2.1
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for scijump.TypeMismatchException
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_TypeMismatchException_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAExceptionType_hxx
#include "gov_cca_CCAExceptionType.hxx"
#endif
#ifndef included_gov_cca_Type_hxx
#include "gov_cca_Type.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_io_Deserializer_hxx
#include "sidl_io_Deserializer.hxx"
#endif
#ifndef included_sidl_io_Serializer_hxx
#include "sidl_io_Serializer.hxx"
#endif
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException._includes)
// Insert-Code-Here {scijump.TypeMismatchException._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(scijump.TypeMismatchException._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::TypeMismatchException_impl::TypeMismatchException_impl() : StubBase(
  reinterpret_cast< void*>(::scijump::TypeMismatchException::_wrapObj(
  reinterpret_cast< void*>(this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException._ctor2)
  // Insert-Code-Here {scijump.TypeMismatchException._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.TypeMismatchException._ctor2)
}

// user defined constructor
void scijump::TypeMismatchException_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException._ctor)
  type = ::gov::cca::CCAExceptionType_Nonstandard;
  // DO-NOT-DELETE splicer.end(scijump.TypeMismatchException._ctor)
}

// user defined destructor
void scijump::TypeMismatchException_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException._dtor)
  // Insert-Code-Here {scijump.TypeMismatchException._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.TypeMismatchException._dtor)
}

// static class initializer
void scijump::TypeMismatchException_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException._load)
  // Insert-Code-Here {scijump.TypeMismatchException._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.TypeMismatchException._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::TypeMismatchException_impl::initialize_impl (
  /* in */::gov::cca::Type requestedType,
  /* in */::gov::cca::Type actualType ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException.initialize)
  this->requestedType = requestedType;
  this->actualType = actualType;
  // DO-NOT-DELETE splicer.end(scijump.TypeMismatchException.initialize)
}

/**
 * Method:  initialize[Full]
 */
void
scijump::TypeMismatchException_impl::initialize_impl (
  /* in */::gov::cca::CCAExceptionType type,
  /* in */::gov::cca::Type requestedType,
  /* in */::gov::cca::Type actualType ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException.initializeFull)
  this->type = type;
  this->requestedType = requestedType;
  this->actualType = actualType;
  // DO-NOT-DELETE splicer.end(scijump.TypeMismatchException.initializeFull)
}

/**
 * Method:  getCCAExceptionType[]
 */
::gov::cca::CCAExceptionType
scijump::TypeMismatchException_impl::getCCAExceptionType_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException.getCCAExceptionType)
  return type;
  // DO-NOT-DELETE splicer.end(scijump.TypeMismatchException.getCCAExceptionType)
}

/**
 *  @return the enumerated value Type sought 
 */
::gov::cca::Type
scijump::TypeMismatchException_impl::getRequestedType_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException.getRequestedType)
  return requestedType;
  // DO-NOT-DELETE splicer.end(scijump.TypeMismatchException.getRequestedType)
}

/**
 *  @return the enumerated value Type sought 
 */
::gov::cca::Type
scijump::TypeMismatchException_impl::getActualType_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException.getActualType)
  return actualType;
  // DO-NOT-DELETE splicer.end(scijump.TypeMismatchException.getActualType)
}


// DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException._misc)
// Insert-Code-Here {scijump.TypeMismatchException._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.TypeMismatchException._misc)

