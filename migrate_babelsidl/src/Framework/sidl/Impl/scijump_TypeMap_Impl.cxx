// 
// File:          scijump_TypeMap_Impl.cxx
// Symbol:        scijump.TypeMap-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.TypeMap
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_TypeMap_Impl.hxx"

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
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(scijump.TypeMap._includes)

#include "scijump_TypeMismatchException.hxx"

// Insert-Code-Here {scijump.TypeMap._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(scijump.TypeMap._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::TypeMap_impl::TypeMap_impl() : StubBase(reinterpret_cast< void*>(
  ::scijump::TypeMap::_wrapObj(reinterpret_cast< void*>(this))),false) , 
  _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap._ctor2)
  // Insert-Code-Here {scijump.TypeMap._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.TypeMap._ctor2)
}

// user defined constructor
void scijump::TypeMap_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap._ctor)
  // Insert-Code-Here {scijump.TypeMap._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(scijump.TypeMap._ctor)
}

// user defined destructor
void scijump::TypeMap_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap._dtor)
  // Insert-Code-Here {scijump.TypeMap._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.TypeMap._dtor)
}

// static class initializer
void scijump::TypeMap_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap._load)
  // Insert-Code-Here {scijump.TypeMap._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.TypeMap._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 *  Create an exact copy of this Map 
 */
::gov::cca::TypeMap
scijump::TypeMap_impl::cloneTypeMap_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.cloneTypeMap)
  scijump::TypeMap_impl tm;
  tm.intMap = this->intMap;
  tm.longMap = this->longMap;
  tm.floatMap = this->floatMap;
  tm.doubleMap = this->doubleMap;
  tm.stringMap = this->stringMap;
  tm.boolMap = this->boolMap;
  tm.fcomplexMap = this->fcomplexMap;
  tm.dcomplexMap = this->dcomplexMap;
  tm.intArrayMap = this->intArrayMap;
  tm.longArrayMap = this->longArrayMap;
  tm.floatArrayMap = this->floatArrayMap;
  tm.doubleArrayMap = this->doubleArrayMap;
  tm.fcomplexArrayMap = this->fcomplexArrayMap;
  tm.dcomplexArrayMap = this->dcomplexArrayMap;
  tm.stringArrayMap = this->stringArrayMap;
  tm.boolArrayMap = this->boolArrayMap;

  return ::sidl::babel_cast< ::gov::cca::TypeMap>(tm);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.cloneTypeMap)
}

/**
 *  Create a new Map with no key/value associations. 
 */
::gov::cca::TypeMap
scijump::TypeMap_impl::cloneEmpty_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.cloneEmpty)
  scijump::TypeMap tm = scijump::TypeMap::_create();
  return tm;
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.cloneEmpty)
}

/**
 * Method:  getInt[]
 */
int32_t
scijump::TypeMap_impl::getInt_impl (
  /* in */const ::std::string& key,
  /* in */int32_t dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getInt)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Int) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Int, t);
    ex.setNote("Type is not ::gov::cca::Type_Int");
    ex.add(__FILE__, __LINE__, "getInt");
    throw ex;
  }
  return intMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getInt)
}

/**
 * Method:  getLong[]
 */
int64_t
scijump::TypeMap_impl::getLong_impl (
  /* in */const ::std::string& key,
  /* in */int64_t dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getLong)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Long) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Long, t);
    ex.setNote("Type is not ::gov::cca::Type_Long");
    ex.add(__FILE__, __LINE__, "getLong");
    throw ex;
  }
  return longMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getLong)
}

/**
 * Method:  getFloat[]
 */
float
scijump::TypeMap_impl::getFloat_impl (
  /* in */const ::std::string& key,
  /* in */float dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getFloat)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Float) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Float, t);
    ex.setNote("Type is not ::gov::cca::Type_Float");
    ex.add(__FILE__, __LINE__, "getFloat");
    throw ex;
  }
  return floatMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getFloat)
}

/**
 * Method:  getDouble[]
 */
double
scijump::TypeMap_impl::getDouble_impl (
  /* in */const ::std::string& key,
  /* in */double dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getDouble)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Double) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Double, t);
    ex.setNote("Type is not ::gov::cca::Type_Double");
    ex.add(__FILE__, __LINE__, "getDouble");
    throw ex;
  }
  return doubleMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getDouble)
}

/**
 * Method:  getFcomplex[]
 */
::std::complex<float>
scijump::TypeMap_impl::getFcomplex_impl (
  /* in */const ::std::string& key,
  /* in */const ::std::complex<float>& dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getFcomplex)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Fcomplex) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Fcomplex, t);
    ex.setNote("Type is not ::gov::cca::Type_Fcomplex");
    ex.add(__FILE__, __LINE__, "getFcomplex");
    throw ex;
  }
  return fcomplexMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getFcomplex)
}

/**
 * Method:  getDcomplex[]
 */
::std::complex<double>
scijump::TypeMap_impl::getDcomplex_impl (
  /* in */const ::std::string& key,
  /* in */const ::std::complex<double>& dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getDcomplex)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Dcomplex) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Dcomplex, t);
    ex.setNote("Type is not ::gov::cca::Type_Dcomplex");
    ex.add(__FILE__, __LINE__, "getDcomplex");
    throw ex;
  }
  return dcomplexMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getDcomplex)
}

/**
 * Method:  getString[]
 */
::std::string
scijump::TypeMap_impl::getString_impl (
  /* in */const ::std::string& key,
  /* in */const ::std::string& dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getString)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_String) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_String, t);
    ex.setNote("Type is not ::gov::cca::Type_String");
    ex.add(__FILE__, __LINE__, "getString");
    throw ex;
  }
  return stringMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getString)
}

/**
 * Method:  getBool[]
 */
bool
scijump::TypeMap_impl::getBool_impl (
  /* in */const ::std::string& key,
  /* in */bool dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getBool)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Bool) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Bool, t);
    ex.setNote("Type is not ::gov::cca::Type_Bool");
    ex.add(__FILE__, __LINE__, "getBool");
    throw ex;
  }
  return boolMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getBool)
}

/**
 * Method:  getIntArray[]
 */
::sidl::array<int32_t>
scijump::TypeMap_impl::getIntArray_impl (
  /* in */const ::std::string& key,
  /* in array<int> */::sidl::array<int32_t>& dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getIntArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_IntArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_IntArray, t);
    ex.setNote("Type is not ::gov::cca::Type_IntArray");
    ex.add(__FILE__, __LINE__, "getIntArray");
    throw ex;
  }
  return intArrayMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getIntArray)
}

/**
 * Method:  getLongArray[]
 */
::sidl::array<int64_t>
scijump::TypeMap_impl::getLongArray_impl (
  /* in */const ::std::string& key,
  /* in array<long> */::sidl::array<int64_t>& dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getLongArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_LongArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_LongArray, t);
    ex.setNote("Type is not ::gov::cca::Type_LongArray");
    ex.add(__FILE__, __LINE__, "getLongArray");
    throw ex;
  }
  return longArrayMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getLongArray)
}

/**
 * Method:  getFloatArray[]
 */
::sidl::array<float>
scijump::TypeMap_impl::getFloatArray_impl (
  /* in */const ::std::string& key,
  /* in array<float> */::sidl::array<float>& dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getFloatArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_FloatArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_FloatArray, t);
    ex.setNote("Type is not ::gov::cca::Type_FloatArray");
    ex.add(__FILE__, __LINE__, "getFloatArray");
    throw ex;
  }
  return floatArrayMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getFloatArray)
}

/**
 * Method:  getDoubleArray[]
 */
::sidl::array<double>
scijump::TypeMap_impl::getDoubleArray_impl (
  /* in */const ::std::string& key,
  /* in array<double> */::sidl::array<double>& dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getDoubleArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_DoubleArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_DoubleArray, t);
    ex.setNote("Type is not ::gov::cca::Type_DoubleArray");
    ex.add(__FILE__, __LINE__, "getDoubleArray");
    throw ex;
  }
  return doubleArrayMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getDoubleArray)
}

/**
 * Method:  getFcomplexArray[]
 */
::sidl::array< ::sidl::fcomplex>
scijump::TypeMap_impl::getFcomplexArray_impl (
  /* in */const ::std::string& key,
  /* in array<fcomplex> */::sidl::array< ::sidl::fcomplex>& dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getFcomplexArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_FcomplexArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_FcomplexArray, t);
    ex.setNote("Type is not ::gov::cca::Type_FcomplexArray");
    ex.add(__FILE__, __LINE__, "getFcomplexArray");
    throw ex;
  }
  return fcomplexArrayMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getFcomplexArray)
}

/**
 * Method:  getDcomplexArray[]
 */
::sidl::array< ::sidl::dcomplex>
scijump::TypeMap_impl::getDcomplexArray_impl (
  /* in */const ::std::string& key,
  /* in array<dcomplex> */::sidl::array< ::sidl::dcomplex>& dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getDcomplexArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_DcomplexArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_DcomplexArray, t);
    ex.setNote("Type is not ::gov::cca::Type_DcomplexArray");
    ex.add(__FILE__, __LINE__, "getDcomplexArray");
    throw ex;
  }
  return dcomplexArrayMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getDcomplexArray)
}

/**
 * Method:  getStringArray[]
 */
::sidl::array< ::std::string>
scijump::TypeMap_impl::getStringArray_impl (
  /* in */const ::std::string& key,
  /* in array<string> */::sidl::array< ::std::string>& dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getStringArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_StringArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_StringArray, t);
    ex.setNote("Type is not ::gov::cca::Type_StringArray");
    ex.add(__FILE__, __LINE__, "getStringArray");
    throw ex;
  }
  return stringArrayMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getStringArray)
}

/**
 * Method:  getBoolArray[]
 */
::sidl::array<bool>
scijump::TypeMap_impl::getBoolArray_impl (
  /* in */const ::std::string& key,
  /* in array<bool> */::sidl::array<bool>& dflt ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getBoolArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_BoolArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_BoolArray, t);
    ex.setNote("Type is not ::gov::cca::Type_BoolArray");
    ex.add(__FILE__, __LINE__, "getBoolArray");
    throw ex;
  }
  return boolArrayMap.get(key, dflt);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getBoolArray)
}

/**
 *  
 * Assign a key and value. Any value previously assigned
 * to the same key will be overwritten so long as it
 * is of the same type. If types conflict, an exception occurs.
 */
void
scijump::TypeMap_impl::putInt_impl (
  /* in */const ::std::string& key,
  /* in */int32_t value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putInt)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Int) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Int, t);
    ex.setNote("Type is not ::gov::cca::Type_Int");
    ex.add(__FILE__, __LINE__, "putInt");
    throw ex;
  }
  return intMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putInt)
}

/**
 * Method:  putLong[]
 */
void
scijump::TypeMap_impl::putLong_impl (
  /* in */const ::std::string& key,
  /* in */int64_t value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putLong)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Long) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Long, t);
    ex.setNote("Type is not ::gov::cca::Type_Long");
    ex.add(__FILE__, __LINE__, "putLong");
    throw ex;
  }
  return longMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putLong)
}

/**
 * Method:  putFloat[]
 */
void
scijump::TypeMap_impl::putFloat_impl (
  /* in */const ::std::string& key,
  /* in */float value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putFloat)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Float) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Float, t);
    ex.setNote("Type is not ::gov::cca::Type_Float");
    ex.add(__FILE__, __LINE__, "putFloat");
    throw ex;
  }
  return floatMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putFloat)
}

/**
 * Method:  putDouble[]
 */
void
scijump::TypeMap_impl::putDouble_impl (
  /* in */const ::std::string& key,
  /* in */double value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putDouble)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Double) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Double, t);
    ex.setNote("Type is not ::gov::cca::Type_Double");
    ex.add(__FILE__, __LINE__, "putDouble");
    throw ex;
  }
  return doubleMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putDouble)
}

/**
 * Method:  putFcomplex[]
 */
void
scijump::TypeMap_impl::putFcomplex_impl (
  /* in */const ::std::string& key,
  /* in */const ::std::complex<float>& value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putFcomplex)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Fcomplex) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Fcomplex, t);
    ex.setNote("Type is not ::gov::cca::Type_Fcomplex");
    ex.add(__FILE__, __LINE__, "putFcomplex");
    throw ex;
  }
  return fcomplexMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putFcomplex)
}

/**
 * Method:  putDcomplex[]
 */
void
scijump::TypeMap_impl::putDcomplex_impl (
  /* in */const ::std::string& key,
  /* in */const ::std::complex<double>& value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putDcomplex)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Dcomplex) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Dcomplex, t);
    ex.setNote("Type is not ::gov::cca::Type_Dcomplex");
    ex.add(__FILE__, __LINE__, "putDcomplex");
    throw ex;
  }
  return dcomplexMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putDcomplex)
}

/**
 * Method:  putString[]
 */
void
scijump::TypeMap_impl::putString_impl (
  /* in */const ::std::string& key,
  /* in */const ::std::string& value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putString)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_String) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_String, t);
    ex.setNote("Type is not ::gov::cca::Type_String");
    ex.add(__FILE__, __LINE__, "putString");
    throw ex;
  }
  return stringMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putString)
}

/**
 * Method:  putBool[]
 */
void
scijump::TypeMap_impl::putBool_impl (
  /* in */const ::std::string& key,
  /* in */bool value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putBool)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_Bool) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_Bool, t);
    ex.setNote("Type is not ::gov::cca::Type_Bool");
    ex.add(__FILE__, __LINE__, "putBool");
    throw ex;
  }
  return boolMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putBool)
}

/**
 * Method:  putIntArray[]
 */
void
scijump::TypeMap_impl::putIntArray_impl (
  /* in */const ::std::string& key,
  /* in array<int> */::sidl::array<int32_t>& value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putIntArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_IntArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_IntArray, t);
    ex.setNote("Type is not ::gov::cca::Type_IntArray");
    ex.add(__FILE__, __LINE__, "putIntArray");
    throw ex;
  }
  return intArrayMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putIntArray)
}

/**
 * Method:  putLongArray[]
 */
void
scijump::TypeMap_impl::putLongArray_impl (
  /* in */const ::std::string& key,
  /* in array<long> */::sidl::array<int64_t>& value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putLongArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_LongArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_LongArray, t);
    ex.setNote("Type is not ::gov::cca::Type_LongArray");
    ex.add(__FILE__, __LINE__, "putLongArray");
    throw ex;
  }
  return longArrayMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putLongArray)
}

/**
 * Method:  putFloatArray[]
 */
void
scijump::TypeMap_impl::putFloatArray_impl (
  /* in */const ::std::string& key,
  /* in array<float> */::sidl::array<float>& value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putFloatArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_FloatArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_FloatArray, t);
    ex.setNote("Type is not ::gov::cca::Type_FloatArray");
    ex.add(__FILE__, __LINE__, "putFloatArray");
    throw ex;
  }
  return floatArrayMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putFloatArray)
}

/**
 * Method:  putDoubleArray[]
 */
void
scijump::TypeMap_impl::putDoubleArray_impl (
  /* in */const ::std::string& key,
  /* in array<double> */::sidl::array<double>& value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putDoubleArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_DoubleArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_DoubleArray, t);
    ex.setNote("Type is not ::gov::cca::Type_DoubleArray");
    ex.add(__FILE__, __LINE__, "putDoubleArray");
    throw ex;
  }
  return doubleArrayMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putDoubleArray)
}

/**
 * Method:  putFcomplexArray[]
 */
void
scijump::TypeMap_impl::putFcomplexArray_impl (
  /* in */const ::std::string& key,
  /* in array<fcomplex> */::sidl::array< ::sidl::fcomplex>& value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putFcomplexArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_FcomplexArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_FcomplexArray, t);
    ex.setNote("Type is not ::gov::cca::Type_FcomplexArray");
    ex.add(__FILE__, __LINE__, "putFcomplexArray");
    throw ex;
  }
  return fcomplexArrayMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putFcomplexArray)
}

/**
 * Method:  putDcomplexArray[]
 */
void
scijump::TypeMap_impl::putDcomplexArray_impl (
  /* in */const ::std::string& key,
  /* in array<dcomplex> */::sidl::array< ::sidl::dcomplex>& value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putDcomplexArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_DcomplexArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_DcomplexArray, t);
    ex.setNote("Type is not ::gov::cca::Type_DcomplexArray");
    ex.add(__FILE__, __LINE__, "putDcomplexArray");
    throw ex;
  }
  return dcomplexArrayMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putDcomplexArray)
}

/**
 * Method:  putStringArray[]
 */
void
scijump::TypeMap_impl::putStringArray_impl (
  /* in */const ::std::string& key,
  /* in array<string> */::sidl::array< ::std::string>& value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putStringArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_StringArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_StringArray, t);
    ex.setNote("Type is not ::gov::cca::Type_StringArray");
    ex.add(__FILE__, __LINE__, "putStringArray");
    throw ex;
  }
  return stringArrayMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putStringArray)
}

/**
 * Method:  putBoolArray[]
 */
void
scijump::TypeMap_impl::putBoolArray_impl (
  /* in */const ::std::string& key,
  /* in array<bool> */::sidl::array<bool>& value ) 
// throws:
//     ::gov::cca::TypeMismatchException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.putBoolArray)
  ::gov::cca::Type t = typeOf(key);
  if (t != ::gov::cca::Type_NoType && t != ::gov::cca::Type_BoolArray) {
    scijump::TypeMismatchException ex = scijump::TypeMismatchException::_create();
    ex.initialize(::gov::cca::Type_BoolArray, t);
    ex.setNote("Type is not ::gov::cca::Type_BoolArray");
    ex.add(__FILE__, __LINE__, "putBoolArray");
    throw ex;
  }
  return boolArrayMap.put(key, value);
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.putBoolArray)
}

/**
 *  Make the key and associated value disappear from the object. 
 */
void
scijump::TypeMap_impl::remove_impl (
  /* in */const ::std::string& key ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.remove)
  if (stringMap.remove(key)) return;
  if (boolMap.remove(key)) return;
  if (intMap.remove(key)) return;
  if (longMap.remove(key)) return;
  if (floatMap.remove(key)) return;
  if (doubleMap.remove(key)) return;
  if (fcomplexMap.remove(key)) return;
  if (dcomplexMap.remove(key)) return;
  if (intArrayMap.remove(key)) return;
  if (longArrayMap.remove(key)) return;
  if (floatArrayMap.remove(key)) return;
  if (doubleArrayMap.remove(key)) return;
  if (stringArrayMap.remove(key)) return;
  if (boolArrayMap.remove(key)) return;
  if (fcomplexArrayMap.remove(key)) return;
  if (dcomplexArrayMap.remove(key)) return;

  // spec doesn't require us to throw an exception if key can't be found
  return;
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.remove)
}

/**
 *  
 * Get all the names associated with a particular type
 * without exposing the data implementation details.  The keys
 * will be returned in an arbitrary order. If type specified is
 * NoType (no specification) all keys of all types are returned.
 */
::sidl::array< ::std::string>
scijump::TypeMap_impl::getAllKeys_impl (
  /* in */::gov::cca::Type t ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.getAllKeys)

  switch(t) {
  case ::gov::cca::Type_String:
    {
      TypeMapBase<std::string>::size_type len = stringMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      stringMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_Bool:
    {
      TypeMapBase<bool>::size_type len = boolMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      boolMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_Int:
    {
      TypeMapBase<int32_t>::size_type len = intMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      intMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_Long:
    {
      TypeMapBase<int64_t>::size_type len = longMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      longMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_Float:
    {
      TypeMapBase<float>::size_type len = floatMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      floatMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_Double:
    {
      TypeMapBase<double>::size_type len = doubleMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      doubleMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_Fcomplex:
    {
      TypeMapBase<std::complex<float> >::size_type len = fcomplexMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      fcomplexMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_Dcomplex:
    {
      TypeMapBase<std::complex<double> >::size_type len = dcomplexMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      dcomplexMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_IntArray:
    {
      TypeMapBase< ::sidl::array<int32_t> >::size_type len = intArrayMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      intArrayMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_LongArray:
    {
      TypeMapBase< ::sidl::array<int64_t> >::size_type len = longArrayMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      longArrayMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_FloatArray:
    {
      TypeMapBase< ::sidl::array<float> >::size_type len = floatArrayMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      floatArrayMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_DoubleArray:
    {
      TypeMapBase< ::sidl::array<double> >::size_type len = doubleArrayMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      doubleArrayMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_StringArray:
    {
      TypeMapBase< ::sidl::array<std::string> >::size_type len = stringArrayMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      stringArrayMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_BoolArray:
    {
      TypeMapBase< ::sidl::array<bool> >::size_type len = boolArrayMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      boolArrayMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_FcomplexArray:
    {
      TypeMapBase< ::sidl::array<std::complex<float> > >::size_type len = fcomplexArrayMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      fcomplexArrayMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_DcomplexArray:
    {
      TypeMapBase< ::sidl::array<std::complex<double> > >::size_type len = dcomplexArrayMap.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      dcomplexArrayMap.getAllKeys(temp);
      return temp;
    }
    break;
  case ::gov::cca::Type_NoType:
    {
      std::list< ::std::string> list(20);
      intMap.getAllKeys(list);
      longMap.getAllKeys(list);
      floatMap.getAllKeys(list);
      doubleMap.getAllKeys(list);
      stringMap.getAllKeys(list);
      boolMap.getAllKeys(list);
      fcomplexMap.getAllKeys(list);
      dcomplexMap.getAllKeys(list);
      intArrayMap.getAllKeys(list);
      longArrayMap.getAllKeys(list);
      floatArrayMap.getAllKeys(list);
      doubleArrayMap.getAllKeys(list);
      stringArrayMap.getAllKeys(list);
      boolArrayMap.getAllKeys(list);
      fcomplexArrayMap.getAllKeys(list);
      dcomplexArrayMap.getAllKeys(list);

      std::list< ::std::string>::size_type len = list.size(); 
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(len);
      int index = 0;
      for (std::list<std::string>::iterator iter = list.begin(); iter != list.end(); iter++) {
        temp.set(index++, *iter);
      }
      return temp;
    }
    break;
  default:
    {
      ::sidl::array< ::std::string> temp = sidl::array< ::std::string>::create1d(0);
      return temp;
    }
    break;
  }
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.getAllKeys)
}

/**
 *  Return true if the key exists in this map 
 */
bool
scijump::TypeMap_impl::hasKey_impl (
  /* in */const ::std::string& key ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.hasKey)
  if (stringMap.hasKey(key)) return true;
  if (boolMap.hasKey(key)) return true;
  if (intMap.hasKey(key)) return true;
  if (longMap.hasKey(key)) return true;
  if (floatMap.hasKey(key)) return true;
  if (doubleMap.hasKey(key)) return true;
  if (fcomplexMap.hasKey(key)) return true;
  if (dcomplexMap.hasKey(key)) return true;
  if (intArrayMap.hasKey(key)) return true;
  if (longArrayMap.hasKey(key)) return true;
  if (floatArrayMap.hasKey(key)) return true;
  if (doubleArrayMap.hasKey(key)) return true;
  if (stringArrayMap.hasKey(key)) return true;
  if (boolArrayMap.hasKey(key)) return true;
  if (fcomplexArrayMap.hasKey(key)) return true;
  if (dcomplexArrayMap.hasKey(key)) return true;
  return false;
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.hasKey)
}

/**
 *  Return the type of the value associated with this key 
 */
::gov::cca::Type
scijump::TypeMap_impl::typeOf_impl (
  /* in */const ::std::string& key ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap.typeOf)
  if (stringMap.hasKey(key)) return ::gov::cca::Type_String;
  if (boolMap.hasKey(key)) return ::gov::cca::Type_Bool;
  if (intMap.hasKey(key)) return ::gov::cca::Type_Int;
  if (longMap.hasKey(key)) return ::gov::cca::Type_Long;
  if (floatMap.hasKey(key)) return ::gov::cca::Type_Float;
  if (doubleMap.hasKey(key)) return ::gov::cca::Type_Double;
  if (fcomplexMap.hasKey(key)) return ::gov::cca::Type_Fcomplex;
  if (dcomplexMap.hasKey(key)) return ::gov::cca::Type_Dcomplex;
  if (intArrayMap.hasKey(key)) return ::gov::cca::Type_IntArray;
  if (longArrayMap.hasKey(key)) return ::gov::cca::Type_LongArray;
  if (floatArrayMap.hasKey(key)) return ::gov::cca::Type_FloatArray;
  if (doubleArrayMap.hasKey(key)) return ::gov::cca::Type_DoubleArray;
  if (stringArrayMap.hasKey(key)) return ::gov::cca::Type_StringArray;
  if (boolArrayMap.hasKey(key)) return ::gov::cca::Type_BoolArray;
  if (fcomplexArrayMap.hasKey(key)) return ::gov::cca::Type_FcomplexArray;
  if (dcomplexArrayMap.hasKey(key)) return ::gov::cca::Type_DcomplexArray;
  return ::gov::cca::Type_NoType;
  // DO-NOT-DELETE splicer.end(scijump.TypeMap.typeOf)
}


// DO-NOT-DELETE splicer.begin(scijump.TypeMap._misc)

template<class T>
scijump::TypeMap_impl::TypeMapBase<T>::TypeMapBase(const TypeMapBase<T>& tmi)
{
  if (this != &tmi) {
    for (MapConstIter iter = tmi.typeMap.begin(); iter != tmi.typeMap.end(); iter++) {
      std::string key(iter->first);
      T value = iter->second;
      this->put(key, value);
    }
  }
}

template<class T>
scijump::TypeMap_impl::TypeMapBase<T>&
scijump::TypeMap_impl::TypeMapBase<T>::operator=(const TypeMapBase<T>& tmi)
{
  if (this == &tmi) {
    return *this;
  }

  for (MapConstIter iter = tmi.typeMap.begin(); iter != tmi.typeMap.end(); iter++) {
    std::string key(iter->first);
    T value = iter->second;
    this->put(key, value);
  }
  return *this;
}

template<class T>
T scijump::TypeMap_impl::TypeMapBase<T>::get(const std::string& key, const T& dflt)
{
    MapIter found = typeMap.find(key);
    if (found != typeMap.end()) {
        return found->second;
    }
    return dflt;
}

template<class T>
void scijump::TypeMap_impl::TypeMapBase<T>::put(const std::string& key, const T& value)
{
  typeMap[key] = value;
}

template<class T>
void scijump::TypeMap_impl::TypeMapBase<T>::getAllKeys(std::list<std::string>& list)
{
  for (MapIter iter = typeMap.begin(); iter != typeMap.end(); iter++) {
    list.push_back(iter->first);
  }
}

template<class T>
void scijump::TypeMap_impl::TypeMapBase<T>::getAllKeys(::sidl::array<std::string>& array, const int startIndex)
{
  int index = startIndex;
  for (MapIter iter = typeMap.begin(); iter != typeMap.end(); iter++) {
    array.set(index++, iter->first);
  }
}

template<class T>
bool scijump::TypeMap_impl::TypeMapBase<T>::hasKey(const std::string& key)
{
    MapIter found = typeMap.find(key);
    if (found != typeMap.end()) {
        return true;
    }
    return false;
}

template<class T>
bool scijump::TypeMap_impl::TypeMapBase<T>::remove(const std::string& key)
{
    MapIter found = typeMap.find(key);
    if (found != typeMap.end()) {
        typeMap.erase(found);
        return true;
    }
    return false;
}

// Insert-Code-Here {scijump.TypeMap._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.TypeMap._misc)

