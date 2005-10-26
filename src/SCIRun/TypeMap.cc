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

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/TypeMap.h>
#include <Core/Util/NotFinished.h>

namespace SCIRun {

TypeMap::TypeMap()
{
}

TypeMap::~TypeMap()
{
}

// .sci.cca.TypeMap .sci.cca.TypeMap.cloneTypeMap()
sci::cca::TypeMap::pointer TypeMap::cloneTypeMap()
{
  NOT_FINISHED("method not implemented");
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

// .sci.cca.TypeMap .sci.cca.TypeMap.cloneEmpty()
sci::cca::TypeMap::pointer TypeMap::cloneEmpty()
{
  return sci::cca::TypeMap::pointer(new TypeMap); 
}

// void .sci.cca.TypeMap.remove(in string key)
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

// array1<string, 1> .sci.cca.TypeMap.getAllKeys(in .sci.cca.Type t)
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

// bool .sci.cca.TypeMap.hasKey(in string key)
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

// .sci.cca.Type .sci.cca.TypeMap.typeOf(in string key)
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
