// 
// File:          framework_TypeMap_Impl.cc
// Symbol:        framework.TypeMap-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030720 10:32:36 MDT
// Generated:     20030720 10:32:38 MDT
// Description:   Server-side implementation for framework.TypeMap
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 17
// source-url    = file:/home/sci/kzhang/SCIRun/debug/../src/SCIRun/Babel/framework.sidl
// 
#include "framework_TypeMap_Impl.hh"

// DO-NOT-DELETE splicer.begin(framework.TypeMap._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(framework.TypeMap._includes)

// user defined constructor
void framework::TypeMap_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(framework.TypeMap._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(framework.TypeMap._ctor)
}

// user defined destructor
void framework::TypeMap_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(framework.TypeMap._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(framework.TypeMap._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Create an exact copy of this Map 
 */
::gov::cca::TypeMap
framework::TypeMap_impl::cloneTypeMap () 
throw () 

{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.cloneTypeMap)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.cloneTypeMap)
}

/**
 * Create a new Map with no key/value associations. 
 */
::gov::cca::TypeMap
framework::TypeMap_impl::cloneEmpty () 
throw () 

{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.cloneEmpty)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.cloneEmpty)
}

/**
 * Method:  getInt[]
 */
int32_t
framework::TypeMap_impl::getInt (
  /*in*/ const ::std::string& key,
  /*in*/ int32_t dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getInt)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getInt)
}

/**
 * Method:  getLong[]
 */
int64_t
framework::TypeMap_impl::getLong (
  /*in*/ const ::std::string& key,
  /*in*/ int64_t dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getLong)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getLong)
}

/**
 * Method:  getFloat[]
 */
float
framework::TypeMap_impl::getFloat (
  /*in*/ const ::std::string& key,
  /*in*/ float dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getFloat)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getFloat)
}

/**
 * Method:  getDouble[]
 */
double
framework::TypeMap_impl::getDouble (
  /*in*/ const ::std::string& key,
  /*in*/ double dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getDouble)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getDouble)
}

/**
 * Method:  getFcomplex[]
 */
::std::complex<float>
framework::TypeMap_impl::getFcomplex (
  /*in*/ const ::std::string& key,
  /*in*/ const ::std::complex<float>& dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getFcomplex)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getFcomplex)
}

/**
 * Method:  getDcomplex[]
 */
::std::complex<double>
framework::TypeMap_impl::getDcomplex (
  /*in*/ const ::std::string& key,
  /*in*/ const ::std::complex<double>& dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getDcomplex)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getDcomplex)
}

/**
 * Method:  getString[]
 */
::std::string
framework::TypeMap_impl::getString (
  /*in*/ const ::std::string& key,
  /*in*/ const ::std::string& dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getString)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getString)
}

/**
 * Method:  getBool[]
 */
bool
framework::TypeMap_impl::getBool (
  /*in*/ const ::std::string& key,
  /*in*/ bool dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getBool)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getBool)
}

/**
 * Method:  getIntArray[]
 */
::SIDL::array<int>
framework::TypeMap_impl::getIntArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<int> dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getIntArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getIntArray)
}

/**
 * Method:  getLongArray[]
 */
::SIDL::array<long>
framework::TypeMap_impl::getLongArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<long> dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getLongArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getLongArray)
}

/**
 * Method:  getFloatArray[]
 */
::SIDL::array<float>
framework::TypeMap_impl::getFloatArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<float> dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getFloatArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getFloatArray)
}

/**
 * Method:  getDoubleArray[]
 */
::SIDL::array<double>
framework::TypeMap_impl::getDoubleArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<double> dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getDoubleArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getDoubleArray)
}

/**
 * Method:  getFcomplexArray[]
 */
::SIDL::array< ::SIDL::fcomplex>
framework::TypeMap_impl::getFcomplexArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array< ::SIDL::fcomplex> dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getFcomplexArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getFcomplexArray)
}

/**
 * Method:  getDcomplexArray[]
 */
::SIDL::array< ::SIDL::dcomplex>
framework::TypeMap_impl::getDcomplexArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array< ::SIDL::dcomplex> dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getDcomplexArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getDcomplexArray)
}

/**
 * Method:  getStringArray[]
 */
::SIDL::array< ::std::string>
framework::TypeMap_impl::getStringArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array< ::std::string> dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getStringArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getStringArray)
}

/**
 * Method:  getBoolArray[]
 */
::SIDL::array<bool>
framework::TypeMap_impl::getBoolArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<bool> dflt ) 
throw ( 
  ::gov::cca::TypeMismatchException
){
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getBoolArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getBoolArray)
}

/**
 * Assign a key and value. Any value previously assigned
 * to the same key will be overwritten.  
 */
void
framework::TypeMap_impl::putInt (
  /*in*/ const ::std::string& key,
  /*in*/ int32_t value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putInt)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putInt)
}

/**
 * Method:  putLong[]
 */
void
framework::TypeMap_impl::putLong (
  /*in*/ const ::std::string& key,
  /*in*/ int64_t value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putLong)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putLong)
}

/**
 * Method:  putFloat[]
 */
void
framework::TypeMap_impl::putFloat (
  /*in*/ const ::std::string& key,
  /*in*/ float value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putFloat)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putFloat)
}

/**
 * Method:  putDouble[]
 */
void
framework::TypeMap_impl::putDouble (
  /*in*/ const ::std::string& key,
  /*in*/ double value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putDouble)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putDouble)
}

/**
 * Method:  putFcomplex[]
 */
void
framework::TypeMap_impl::putFcomplex (
  /*in*/ const ::std::string& key,
  /*in*/ const ::std::complex<float>& value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putFcomplex)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putFcomplex)
}

/**
 * Method:  putDcomplex[]
 */
void
framework::TypeMap_impl::putDcomplex (
  /*in*/ const ::std::string& key,
  /*in*/ const ::std::complex<double>& value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putDcomplex)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putDcomplex)
}

/**
 * Method:  putString[]
 */
void
framework::TypeMap_impl::putString (
  /*in*/ const ::std::string& key,
  /*in*/ const ::std::string& value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putString)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putString)
}

/**
 * Method:  putBool[]
 */
void
framework::TypeMap_impl::putBool (
  /*in*/ const ::std::string& key,
  /*in*/ bool value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putBool)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putBool)
}

/**
 * Method:  putIntArray[]
 */
void
framework::TypeMap_impl::putIntArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<int> value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putIntArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putIntArray)
}

/**
 * Method:  putLongArray[]
 */
void
framework::TypeMap_impl::putLongArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<long> value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putLongArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putLongArray)
}

/**
 * Method:  putFloatArray[]
 */
void
framework::TypeMap_impl::putFloatArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<float> value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putFloatArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putFloatArray)
}

/**
 * Method:  putDoubleArray[]
 */
void
framework::TypeMap_impl::putDoubleArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<double> value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putDoubleArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putDoubleArray)
}

/**
 * Method:  putFcomplexArray[]
 */
void
framework::TypeMap_impl::putFcomplexArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array< ::SIDL::fcomplex> value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putFcomplexArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putFcomplexArray)
}

/**
 * Method:  putDcomplexArray[]
 */
void
framework::TypeMap_impl::putDcomplexArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array< ::SIDL::dcomplex> value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putDcomplexArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putDcomplexArray)
}

/**
 * Method:  putStringArray[]
 */
void
framework::TypeMap_impl::putStringArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array< ::std::string> value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putStringArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putStringArray)
}

/**
 * Method:  putBoolArray[]
 */
void
framework::TypeMap_impl::putBoolArray (
  /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<bool> value ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.putBoolArray)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.putBoolArray)
}

/**
 * Make the key and associated value disappear from the object. 
 */
void
framework::TypeMap_impl::remove (
  /*in*/ const ::std::string& key ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.remove)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.remove)
}

/**
 *  Get all the names associated with a particular type
 *  without exposing the data implementation details.  The keys
 *  will be returned in an arbitrary order. If type specified is
 *  None (no specification) all keys of all types are returned.
 */
::SIDL::array< ::std::string>
framework::TypeMap_impl::getAllKeys (
  /*in*/ ::gov::cca::Type t ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.getAllKeys)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.getAllKeys)
}

/**
 * Return true if the key exists in this map 
 */
bool
framework::TypeMap_impl::hasKey (
  /*in*/ const ::std::string& key ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.hasKey)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.hasKey)
}

/**
 * Return the type of the value associated with this key 
 */
::gov::cca::Type
framework::TypeMap_impl::typeOf (
  /*in*/ const ::std::string& key ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.TypeMap.typeOf)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.TypeMap.typeOf)
}


// DO-NOT-DELETE splicer.begin(framework.TypeMap._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(framework.TypeMap._misc)

