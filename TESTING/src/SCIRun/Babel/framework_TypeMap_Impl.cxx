// For more information, please see: http://software.sci.utah.edu
//
// The MIT License
//
// Copyright (c) 2004 Scientific Computing and Imaging Institute,
// University of Utah.
//
// 
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// 
// File:          framework_TypeMap_Impl.cxx
// Symbol:        framework.TypeMap-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for framework.TypeMap
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 
#include "framework_TypeMap_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_Type_hxx
#include "gov_cca_Type.hxx"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_gov_cca_TypeMismatchException_hxx
#include "gov_cca_TypeMismatchException.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_RuntimeException_hxx
#include "sidl_RuntimeException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(framework.TypeMap._includes)
// Insert-Code-Here {framework.TypeMap._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(framework.TypeMap._includes)

// user defined constructor
void framework::TypeMap_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(framework.TypeMap._ctor)
  typeMap = new SCIRun::TypeMap();
  // DO-NOT-DELETE splicer.end(framework.TypeMap._ctor)
}

// user defined destructor
void framework::TypeMap_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(framework.TypeMap._dtor)
  delete typeMap;
  // DO-NOT-DELETE splicer.end(framework.TypeMap._dtor)
}

// static class initializer
void framework::TypeMap_impl::_load() {
  // DO-NOT-DELETE splicer.begin(framework.TypeMap._load)
  // Insert-Code-Here {framework.TypeMap._load} (class initialization)
  // DO-NOT-DELETE splicer.end(framework.TypeMap._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  setInternalData[]
 */
void
framework::TypeMap_impl::setInternalData_impl (
  /* in */void* data ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.setInternalData)
  SCIRun::TypeMap* tm = (SCIRun::TypeMap*) data;
  if (tm) {
    typeMap = tm;
  }
  // DO-NOT-DELETE splicer.end(framework.TypeMap.setInternalData)
}

/**
 * Method:  getInternalData[]
 */
void*
framework::TypeMap_impl::getInternalData_impl () 

{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getInternalData)
  return &typeMap;
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getInternalData)
}

/**
 * Create an exact copy of this Map 
 */
UCXX ::gov::cca::TypeMap
framework::TypeMap_impl::cloneTypeMap_impl () 

{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.cloneTypeMap)
  sci::cca::TypeMap::pointer tmp = this->typeMap->cloneTypeMap();
  SCIRun::TypeMap* tm = (SCIRun::TypeMap*) tmp.getPointer();

  UCXX ::framework::TypeMap ftm = UCXX ::framework::TypeMap::_create();
  ftm.setInternalData(this->getInternalData());
  UCXX ::gov::cca::TypeMap gctm = UCXX ::sidl::babel_cast<UCXX ::gov::cca::TypeMap>(ftm);
  return gctm;
  // DO-NOT-DELETE splicer.end(framework.TypeMap.cloneTypeMap)
}

/**
 * Create a new Map with no key/value associations. 
 */
UCXX ::gov::cca::TypeMap
framework::TypeMap_impl::cloneEmpty_impl () 

{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.cloneEmpty)
  UCXX ::framework::TypeMap ftm = UCXX ::framework::TypeMap::_create();
  UCXX ::gov::cca::TypeMap gctm = UCXX ::sidl::babel_cast<UCXX ::gov::cca::TypeMap>(ftm);
  return gctm;
  // DO-NOT-DELETE splicer.end(framework.TypeMap.cloneEmpty)
}

/**
 * Method:  getInt[]
 */
int32_t
framework::TypeMap_impl::getInt_impl (
  /* in */const ::std::string& key,
  /* in */int32_t dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getInt)
  return typeMap->getInt(key, dflt);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getInt)
}

/**
 * Method:  getLong[]
 */
int64_t
framework::TypeMap_impl::getLong_impl (
  /* in */const ::std::string& key,
  /* in */int64_t dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getLong)
  return typeMap->getLong(key, dflt);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getLong)
}

/**
 * Method:  getFloat[]
 */
float
framework::TypeMap_impl::getFloat_impl (
  /* in */const ::std::string& key,
  /* in */float dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getFloat)
  return typeMap->getFloat(key, dflt);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getFloat)
}

/**
 * Method:  getDouble[]
 */
double
framework::TypeMap_impl::getDouble_impl (
  /* in */const ::std::string& key,
  /* in */double dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getDouble)
  return typeMap->getDouble(key, dflt);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getDouble)
}

/**
 * Method:  getFcomplex[]
 */
::std::complex<float>
framework::TypeMap_impl::getFcomplex_impl (
  /* in */const ::std::string& key,
  /* in */const ::std::complex<float>& dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getFcomplex)
  return typeMap->getFcomplex(key, dflt);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getFcomplex)
}

/**
 * Method:  getDcomplex[]
 */
::std::complex<double>
framework::TypeMap_impl::getDcomplex_impl (
  /* in */const ::std::string& key,
  /* in */const ::std::complex<double>& dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getDcomplex)
  return typeMap->getDcomplex(key, dflt);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getDcomplex)
}

/**
 * Method:  getString[]
 */
::std::string
framework::TypeMap_impl::getString_impl (
  /* in */const ::std::string& key,
  /* in */const ::std::string& dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getString)
  return typeMap->getString(key, dflt);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getString)
}

/**
 * Method:  getBool[]
 */
bool
framework::TypeMap_impl::getBool_impl (
  /* in */const ::std::string& key,
  /* in */bool dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getBool)
  return typeMap->getBool(key, dflt);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getBool)
}

/**
 * Method:  getIntArray[]
 */
UCXX ::sidl::array<int32_t>
framework::TypeMap_impl::getIntArray_impl (
  /* in */const ::std::string& key,
  /* in array<int> */UCXX ::sidl::array<int32_t> dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getIntArray)
  //return typeMap->getIntArray(key, dflt);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getIntArray)
}

/**
 * Method:  getLongArray[]
 */
UCXX ::sidl::array<int64_t>
framework::TypeMap_impl::getLongArray_impl (
  /* in */const ::std::string& key,
  /* in array<long> */UCXX ::sidl::array<int64_t> dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getLongArray)
  // Insert-Code-Here {framework.TypeMap.getLongArray} (getLongArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getLongArray)
}

/**
 * Method:  getFloatArray[]
 */
UCXX ::sidl::array<float>
framework::TypeMap_impl::getFloatArray_impl (
  /* in */const ::std::string& key,
  /* in array<float> */UCXX ::sidl::array<float> dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getFloatArray)
  // Insert-Code-Here {framework.TypeMap.getFloatArray} (getFloatArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getFloatArray)
}

/**
 * Method:  getDoubleArray[]
 */
UCXX ::sidl::array<double>
framework::TypeMap_impl::getDoubleArray_impl (
  /* in */const ::std::string& key,
  /* in array<double> */UCXX ::sidl::array<double> dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getDoubleArray)
  // Insert-Code-Here {framework.TypeMap.getDoubleArray} (getDoubleArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getDoubleArray)
}

/**
 * Method:  getFcomplexArray[]
 */
UCXX ::sidl::array< UCXX ::sidl::fcomplex>
framework::TypeMap_impl::getFcomplexArray_impl (
  /* in */const ::std::string& key,
  /* in array<fcomplex> */UCXX ::sidl::array< UCXX ::sidl::fcomplex> dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getFcomplexArray)
  // Insert-Code-Here {framework.TypeMap.getFcomplexArray} (getFcomplexArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getFcomplexArray)
}

/**
 * Method:  getDcomplexArray[]
 */
UCXX ::sidl::array< UCXX ::sidl::dcomplex>
framework::TypeMap_impl::getDcomplexArray_impl (
  /* in */const ::std::string& key,
  /* in array<dcomplex> */UCXX ::sidl::array< UCXX ::sidl::dcomplex> dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getDcomplexArray)
  // Insert-Code-Here {framework.TypeMap.getDcomplexArray} (getDcomplexArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getDcomplexArray)
}

/**
 * Method:  getStringArray[]
 */
UCXX ::sidl::array< ::std::string>
framework::TypeMap_impl::getStringArray_impl (
  /* in */const ::std::string& key,
  /* in array<string> */UCXX ::sidl::array< ::std::string> dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getStringArray)
  // Insert-Code-Here {framework.TypeMap.getStringArray} (getStringArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getStringArray)
}

/**
 * Method:  getBoolArray[]
 */
UCXX ::sidl::array<bool>
framework::TypeMap_impl::getBoolArray_impl (
  /* in */const ::std::string& key,
  /* in array<bool> */UCXX ::sidl::array<bool> dflt ) 
throw ( 
  UCXX ::gov::cca::TypeMismatchException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getBoolArray)
  // Insert-Code-Here {framework.TypeMap.getBoolArray} (getBoolArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getBoolArray)
}

/**
 * Assign a key and value. Any value previously assigned
 * to the same key will be overwritten.
 */
void
framework::TypeMap_impl::putInt_impl (
  /* in */const ::std::string& key,
  /* in */int32_t value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putInt)
  typeMap->putInt(key, value);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putInt)
}

/**
 * Method:  putLong[]
 */
void
framework::TypeMap_impl::putLong_impl (
  /* in */const ::std::string& key,
  /* in */int64_t value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putLong)
  typeMap->putLong(key, value);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putLong)
}

/**
 * Method:  putFloat[]
 */
void
framework::TypeMap_impl::putFloat_impl (
  /* in */const ::std::string& key,
  /* in */float value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putFloat)
  typeMap->putFloat(key, value);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putFloat)
}

/**
 * Method:  putDouble[]
 */
void
framework::TypeMap_impl::putDouble_impl (
  /* in */const ::std::string& key,
  /* in */double value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putDouble)
  typeMap->putDouble(key, value);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putDouble)
}

/**
 * Method:  putFcomplex[]
 */
void
framework::TypeMap_impl::putFcomplex_impl (
  /* in */const ::std::string& key,
  /* in */const ::std::complex<float>& value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putFcomplex)
  typeMap->putFcomplex(key, value);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putFcomplex)
}

/**
 * Method:  putDcomplex[]
 */
void
framework::TypeMap_impl::putDcomplex_impl (
  /* in */const ::std::string& key,
  /* in */const ::std::complex<double>& value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putDcomplex)
  typeMap->putDcomplex(key, value);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putDcomplex)
}

/**
 * Method:  putString[]
 */
void
framework::TypeMap_impl::putString_impl (
  /* in */const ::std::string& key,
  /* in */const ::std::string& value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putString)
  typeMap->putString(key, value);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putString)
}

/**
 * Method:  putBool[]
 */
void
framework::TypeMap_impl::putBool_impl (
  /* in */const ::std::string& key,
  /* in */bool value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putBool)
  typeMap->putBool(key, value);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putBool)
}

/**
 * Method:  putIntArray[]
 */
void
framework::TypeMap_impl::putIntArray_impl (
  /* in */const ::std::string& key,
  /* in array<int> */UCXX ::sidl::array<int32_t> value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putIntArray)
  // Insert-Code-Here {framework.TypeMap.putIntArray} (putIntArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putIntArray)
}

/**
 * Method:  putLongArray[]
 */
void
framework::TypeMap_impl::putLongArray_impl (
  /* in */const ::std::string& key,
  /* in array<long> */UCXX ::sidl::array<int64_t> value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putLongArray)
  // Insert-Code-Here {framework.TypeMap.putLongArray} (putLongArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putLongArray)
}

/**
 * Method:  putFloatArray[]
 */
void
framework::TypeMap_impl::putFloatArray_impl (
  /* in */const ::std::string& key,
  /* in array<float> */UCXX ::sidl::array<float> value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putFloatArray)
  // Insert-Code-Here {framework.TypeMap.putFloatArray} (putFloatArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putFloatArray)
}

/**
 * Method:  putDoubleArray[]
 */
void
framework::TypeMap_impl::putDoubleArray_impl (
  /* in */const ::std::string& key,
  /* in array<double> */UCXX ::sidl::array<double> value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putDoubleArray)
  // Insert-Code-Here {framework.TypeMap.putDoubleArray} (putDoubleArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putDoubleArray)
}

/**
 * Method:  putFcomplexArray[]
 */
void
framework::TypeMap_impl::putFcomplexArray_impl (
  /* in */const ::std::string& key,
  /* in array<fcomplex> */UCXX ::sidl::array< UCXX ::sidl::fcomplex> value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putFcomplexArray)
  // Insert-Code-Here {framework.TypeMap.putFcomplexArray} (putFcomplexArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putFcomplexArray)
}

/**
 * Method:  putDcomplexArray[]
 */
void
framework::TypeMap_impl::putDcomplexArray_impl (
  /* in */const ::std::string& key,
  /* in array<dcomplex> */UCXX ::sidl::array< UCXX ::sidl::dcomplex> value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putDcomplexArray)
  // Insert-Code-Here {framework.TypeMap.putDcomplexArray} (putDcomplexArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putDcomplexArray)
}

/**
 * Method:  putStringArray[]
 */
void
framework::TypeMap_impl::putStringArray_impl (
  /* in */const ::std::string& key,
  /* in array<string> */UCXX ::sidl::array< ::std::string> value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putStringArray)
  // Insert-Code-Here {framework.TypeMap.putStringArray} (putStringArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putStringArray)
}

/**
 * Method:  putBoolArray[]
 */
void
framework::TypeMap_impl::putBoolArray_impl (
  /* in */const ::std::string& key,
  /* in array<bool> */UCXX ::sidl::array<bool> value ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putBoolArray)
  // Insert-Code-Here {framework.TypeMap.putBoolArray} (putBoolArray method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putBoolArray)
}

/**
 * Make the key and associated value disappear from the object. 
 */
void
framework::TypeMap_impl::remove_impl (
  /* in */const ::std::string& key ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.remove)
  typeMap->remove(key);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.remove)
}

/**
 *  Get all the names associated with a particular type
 *  without exposing the data implementation details.  The keys
 *  will be returned in an arbitrary order. If type specified is
 *  None (no specification) all keys of all types are returned.
 */
UCXX ::sidl::array< ::std::string>
framework::TypeMap_impl::getAllKeys_impl (
  /* in */UCXX ::gov::cca::Type t ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getAllKeys)
  // Insert-Code-Here {framework.TypeMap.getAllKeys} (getAllKeys method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getAllKeys)
}

/**
 * Return true if the key exists in this map 
 */
bool
framework::TypeMap_impl::hasKey_impl (
  /* in */const ::std::string& key ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.hasKey)
  return typeMap->hasKey(key);
  // DO-NOT-DELETE splicer.end(framework.TypeMap.hasKey)
}

/**
 * Return the type of the value associated with this key 
 */
UCXX ::gov::cca::Type
framework::TypeMap_impl::typeOf_impl (
  /* in */const ::std::string& key ) 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.typeOf)
  // Insert-Code-Here {framework.TypeMap.typeOf} (typeOf method)
  // DO-NOT-DELETE splicer.end(framework.TypeMap.typeOf)
}


// DO-NOT-DELETE splicer.begin(framework.TypeMap._misc)
// Insert-Code-Here {framework.TypeMap._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(framework.TypeMap._misc)

