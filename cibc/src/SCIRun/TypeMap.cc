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
 *  TypeMap.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 */

#include <SCIRun/TypeMap.h>
#include <SCIRun/CCA/TypeMismatchException.h>

namespace SCIRun {

TypeMap::TypeMap()
{
}

TypeMap::~TypeMap()
{
}

int
TypeMap::getInt(const std::string& key, int dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Int) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getInt", sci::cca::Int, t));
  }
  return intMap.get(key, dflt);
}

long
TypeMap::getLong(const std::string& key, long dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Long) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getLong", sci::cca::Long, t));
  }
  return longMap.get(key, dflt);
}

float
TypeMap::getFloat(const std::string& key, float dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Float) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getFloat", sci::cca::Float, t));
  }
  return floatMap.get(key, dflt);
}

double
TypeMap::getDouble(const std::string& key, double dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Double) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getDouble", sci::cca::Double, t));
  }
  return doubleMap.get(key, dflt);
}

std::complex<float>
TypeMap::getFcomplex(const std::string& key, const std::complex<float>& dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Fcomplex) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getFcomplex", sci::cca::Fcomplex, t));
  }
  return fcomplexMap.get(key, dflt);
}

std::complex<double>
TypeMap::getDcomplex(const std::string& key, const std::complex<double>& dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Dcomplex) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getDcomplex", sci::cca::Dcomplex, t));
  }
  return dcomplexMap.get(key, dflt);
}

std::string
TypeMap::getString(const std::string& key, const std::string& dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::String) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getString", sci::cca::String, t));
  }
  return stringMap.get(key, dflt);
}

bool
TypeMap::getBool(const std::string& key, bool dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Bool) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getBool", sci::cca::Bool, t));
  }
  return boolMap.get(key, dflt);
}

SSIDL::array1<int>
TypeMap::getIntArray(const std::string& key, const SSIDL::array1<int>& dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::IntArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getIntArray", sci::cca::IntArray, t));
  }
  return intArrayMap.get(key, dflt);
}

SSIDL::array1<long>
TypeMap::getLongArray(const std::string& key, const SSIDL::array1<long>& dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::LongArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getLongArray", sci::cca::LongArray, t));
  }
  return longArrayMap.get(key, dflt);
}

SSIDL::array1<float>
TypeMap::getFloatArray(const std::string& key, const SSIDL::array1<float>& dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::FloatArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getFloatArray", sci::cca::FloatArray, t));
  }
  return floatArrayMap.get(key, dflt);
}

SSIDL::array1<double>
TypeMap::getDoubleArray(const std::string& key, const SSIDL::array1<double>& dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::DoubleArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getDoubleArray", sci::cca::DoubleArray, t));
  }
  return doubleArrayMap.get(key, dflt);
}

SSIDL::array1<std::complex<float> >
TypeMap::getFcomplexArray(const std::string& key, const SSIDL::array1<std::complex<float> >& dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::FcomplexArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getFcomplexArray", sci::cca::FcomplexArray, t));
  }
  return fcomplexArrayMap.get(key, dflt);
}

SSIDL::array1<std::complex<double> >
TypeMap::getDcomplexArray(const std::string& key, const SSIDL::array1<std::complex<double> >& dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::DcomplexArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getDcomplexArray", sci::cca::DcomplexArray, t));
  }
  return dcomplexArrayMap.get(key, dflt);
}

SSIDL::array1<std::string>
TypeMap::getStringArray(const std::string& key, const SSIDL::array1<std::string>& dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::StringArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getStringArray", sci::cca::StringArray, t));
  }
  return stringArrayMap.get(key, dflt);
}

SSIDL::array1<bool>
TypeMap::getBoolArray(const std::string& key, const SSIDL::array1<bool>& dflt)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::BoolArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getBoolArray", sci::cca::BoolArray, t));
  }
  return boolArrayMap.get(key, dflt);
}

void
TypeMap::putInt(const std::string& key, int value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Int) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getInt", sci::cca::Int, t));
  }
  return intMap.put(key, value);
}

void
TypeMap::putLong(const std::string& key, long value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Long) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getLong", sci::cca::Long, t));
  }
  return longMap.put(key, value);
}

void
TypeMap::putFloat(const std::string& key, float value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Float) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getFloat", sci::cca::Float, t));
  }
  return floatMap.put(key, value);
}

void
TypeMap::putDouble(const std::string& key, double value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Double) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getDouble", sci::cca::Double, t));
  }
  return doubleMap.put(key, value);
}

void
TypeMap::putFcomplex(const std::string& key, const std::complex<float>& value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Fcomplex) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getFcomplex", sci::cca::Fcomplex, t));
  }
  return fcomplexMap.put(key, value);
}

void
TypeMap::putDcomplex(const std::string& key, const std::complex<double>& value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Dcomplex) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getDcomplex", sci::cca::Dcomplex, t));
  }
  return dcomplexMap.put(key, value);
}

void
TypeMap::putString(const std::string& key, const std::string& value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::String) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getString", sci::cca::String, t));
  }
  return stringMap.put(key, value);
}

void
TypeMap::putBool(const std::string& key, bool value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::Bool) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getBool", sci::cca::Bool, t));
  }
  return boolMap.put(key, value);
}

void
TypeMap::putIntArray(const std::string& key, const SSIDL::array1<int>& value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::IntArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getIntArray", sci::cca::IntArray, t));
  }
  return intArrayMap.put(key, value);
}

void
TypeMap::putLongArray(const std::string& key, const SSIDL::array1<long>& value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::LongArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getLongArray", sci::cca::LongArray, t));
  }
  return longArrayMap.put(key, value);
}

void
TypeMap::putFloatArray(const std::string& key, const SSIDL::array1<float>& value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::FloatArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getFloatArray", sci::cca::FloatArray, t));
  }
  return floatArrayMap.put(key, value);
}

void
TypeMap::putDoubleArray(const std::string& key, const SSIDL::array1<double>& value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::DoubleArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getDoubleArray", sci::cca::DoubleArray, t));
  }
  return doubleArrayMap.put(key, value);
}

void
TypeMap::putFcomplexArray(const std::string& key, const SSIDL::array1<std::complex<float> >& value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::FcomplexArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getFcomplexArray", sci::cca::FcomplexArray, t));
  }
  return fcomplexArrayMap.put(key, value);
}

void
TypeMap::putDcomplexArray(const std::string& key, const SSIDL::array1<std::complex<double> >& value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::DcomplexArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getDcomplexArray", sci::cca::DcomplexArray, t));
  }
  return dcomplexArrayMap.put(key, value);
}

void
TypeMap::putStringArray(const std::string& key, const SSIDL::array1<std::string>& value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::StringArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getStringArray", sci::cca::StringArray, t));
  }
  return stringArrayMap.put(key, value);
}

void
TypeMap::putBoolArray(const std::string& key, const SSIDL::array1<bool>& value)
{
  sci::cca::Type t = typeOf(key);
  if (t != sci::cca::None && t != sci::cca::BoolArray) {
    throw sci::cca::TypeMismatchException::pointer(new TypeMismatchException("getBoolArray", sci::cca::BoolArray, t));
  }
  return boolArrayMap.put(key, value);
}

sci::cca::TypeMap::pointer TypeMap::cloneTypeMap()
{
  TypeMap *tm = new TypeMap();
  tm->intMap = this->intMap;
  tm->longMap = this->longMap;
  tm->floatMap = this->floatMap;
  tm->doubleMap = this->doubleMap;
  tm->stringMap = this->stringMap;
  tm->boolMap = this->boolMap;
  tm->fcomplexMap = this->fcomplexMap;
  tm->dcomplexMap = this->dcomplexMap;
  tm->intArrayMap = this->intArrayMap;
  tm->longArrayMap = this->longArrayMap;
  tm->floatArrayMap = this->floatArrayMap;
  tm->doubleArrayMap = this->doubleArrayMap;
  tm->fcomplexArrayMap = this->fcomplexArrayMap;
  tm->dcomplexArrayMap = this->dcomplexArrayMap;
  tm->stringArrayMap = this->stringArrayMap;
  tm->boolArrayMap = this->boolArrayMap;
  return sci::cca::TypeMap::pointer(tm);  
}

sci::cca::TypeMap::pointer TypeMap::cloneEmpty()
{
  return sci::cca::TypeMap::pointer(new TypeMap); 
}

void
TypeMap::remove(const std::string& key)
{
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
}

SSIDL::array1<std::string>
TypeMap::getAllKeys(sci::cca::Type type)
{
  SSIDL::array1<std::string> temp;

  switch(type) {
  case sci::cca::String:
    stringMap.getAllKeys(temp);
    return temp;
  case sci::cca::Bool:
    boolMap.getAllKeys(temp);
    return temp;
  case sci::cca::Int:
    intMap.getAllKeys(temp);
    return temp;
  case sci::cca::Long:
    longMap.getAllKeys(temp);
    return temp;
  case sci::cca::Float:
    floatMap.getAllKeys(temp);
    return temp;
  case sci::cca::Double:
    doubleMap.getAllKeys(temp);
    return temp;
  case sci::cca::Fcomplex:
    fcomplexMap.getAllKeys(temp);
    return temp;
  case sci::cca::Dcomplex:
    dcomplexMap.getAllKeys(temp);
    return temp;
  case sci::cca::IntArray:
    intArrayMap.getAllKeys(temp);
    return temp;
  case sci::cca::LongArray:
    longArrayMap.getAllKeys(temp);
    return temp;
  case sci::cca::FloatArray:
    floatArrayMap.getAllKeys(temp);
    return temp;
  case sci::cca::DoubleArray:
    doubleArrayMap.getAllKeys(temp);
    return temp;
  case sci::cca::StringArray:
    stringArrayMap.getAllKeys(temp);
    return temp;
  case sci::cca::BoolArray:
    boolArrayMap.getAllKeys(temp);
    return temp;
  case sci::cca::FcomplexArray:
    fcomplexArrayMap.getAllKeys(temp);
    return temp;
  case sci::cca::DcomplexArray:
    dcomplexArrayMap.getAllKeys(temp);
    return temp;
  case sci::cca::None: // if type == None, get keys from all types
    intMap.getAllKeys(temp);
    longMap.getAllKeys(temp);
    floatMap.getAllKeys(temp);
    doubleMap.getAllKeys(temp);
    stringMap.getAllKeys(temp);
    boolMap.getAllKeys(temp);
    fcomplexMap.getAllKeys(temp);
    dcomplexMap.getAllKeys(temp);
    intArrayMap.getAllKeys(temp);
    longArrayMap.getAllKeys(temp);
    floatArrayMap.getAllKeys(temp);
    doubleArrayMap.getAllKeys(temp);
    stringArrayMap.getAllKeys(temp);
    boolArrayMap.getAllKeys(temp);
    fcomplexArrayMap.getAllKeys(temp);
    dcomplexArrayMap.getAllKeys(temp);
    return temp;
  default:
    return temp;
  }
}

bool
TypeMap::hasKey(const std::string& key)
{
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
}

sci::cca::Type
TypeMap::typeOf(const std::string& key)
{
  if (stringMap.hasKey(key)) return sci::cca::String;
  if (boolMap.hasKey(key)) return sci::cca::Bool;
  if (intMap.hasKey(key)) return sci::cca::Int;
  if (longMap.hasKey(key)) return sci::cca::Long;
  if (floatMap.hasKey(key)) return sci::cca::Float;
  if (doubleMap.hasKey(key)) return sci::cca::Double;
  if (fcomplexMap.hasKey(key)) return sci::cca::Fcomplex;
  if (dcomplexMap.hasKey(key)) return sci::cca::Dcomplex;
  if (intArrayMap.hasKey(key)) return sci::cca::IntArray;
  if (longArrayMap.hasKey(key)) return sci::cca::LongArray;
  if (floatArrayMap.hasKey(key)) return sci::cca::FloatArray;
  if (doubleArrayMap.hasKey(key)) return sci::cca::DoubleArray;
  if (stringArrayMap.hasKey(key)) return sci::cca::StringArray;
  if (boolArrayMap.hasKey(key)) return sci::cca::BoolArray;
  if (fcomplexArrayMap.hasKey(key)) return sci::cca::FcomplexArray;
  if (dcomplexArrayMap.hasKey(key)) return sci::cca::DcomplexArray;
  return sci::cca::None;
}

template<class T>
TypeMap::TypeMapImpl<T>::TypeMapImpl(const TypeMapImpl<T>& tmi)
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
TypeMap::TypeMapImpl<T>&
TypeMap::TypeMapImpl<T>::operator=(const TypeMapImpl<T>& tmi)
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
T TypeMap::TypeMapImpl<T>::get(const std::string& key, const T& dflt)
{
    MapIter found = typeMap.find(key);
    if (found != typeMap.end()) {
        return found->second;
    }
    return dflt;
}

template<class T>
void TypeMap::TypeMapImpl<T>::put(const std::string& key, const T& value)
{
  typeMap[key] = value;
}

template<class T>
void TypeMap::TypeMapImpl<T>::getAllKeys(SSIDL::array1<std::string>& array)
{
  for (MapIter iter = typeMap.begin(); iter != typeMap.end(); iter++) {
      array.push_back(iter->first);
  }
}

template<class T>
bool TypeMap::TypeMapImpl<T>::hasKey(const std::string& key)
{
    MapIter found = typeMap.find(key);
    if (found != typeMap.end()) {
        return true;
    }
    return false;
}

template<class T>
bool TypeMap::TypeMapImpl<T>::remove(const std::string& key)
{
    MapIter found = typeMap.find(key);
    if (found != typeMap.end()) {
        typeMap.erase(found);
        return true;
    }
    return false;
}

} // end namespace SCIRun
