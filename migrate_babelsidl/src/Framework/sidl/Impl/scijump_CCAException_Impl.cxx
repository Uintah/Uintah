// 
// File:          scijump_CCAException_Impl.cxx
// Symbol:        scijump.CCAException-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.CCAException
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_CCAException_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAExceptionType_hxx
#include "gov_cca_CCAExceptionType.hxx"
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
// DO-NOT-DELETE splicer.begin(scijump.CCAException._includes)
// Insert-Code-Here {scijump.CCAException._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(scijump.CCAException._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::CCAException_impl::CCAException_impl() : StubBase(reinterpret_cast< 
  void*>(::scijump::CCAException::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.CCAException._ctor2)
  // Insert-Code-Here {scijump.CCAException._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.CCAException._ctor2)
}

// user defined constructor
void scijump::CCAException_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.CCAException._ctor)
  type = ::gov::cca::CCAExceptionType_Unexpected;
  // DO-NOT-DELETE splicer.end(scijump.CCAException._ctor)
}

// user defined destructor
void scijump::CCAException_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.CCAException._dtor)
  // Insert-Code-Here {scijump.CCAException._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.CCAException._dtor)
}

// static class initializer
void scijump::CCAException_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.CCAException._load)
  // Insert-Code-Here {scijump.CCAException._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.CCAException._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::CCAException_impl::initialize_impl (
  /* in */::gov::cca::CCAExceptionType type ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.CCAException.initialize)
  this->type = type;
  // DO-NOT-DELETE splicer.end(scijump.CCAException.initialize)
}

/**
 * Method:  getCCAExceptionType[]
 */
::gov::cca::CCAExceptionType
scijump::CCAException_impl::getCCAExceptionType_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.CCAException.getCCAExceptionType)
  return type;
  // DO-NOT-DELETE splicer.end(scijump.CCAException.getCCAExceptionType)
}


// DO-NOT-DELETE splicer.begin(scijump.CCAException._misc)
// Insert-Code-Here {scijump.CCAException._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.CCAException._misc)

