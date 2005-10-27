/* For more information, please see: http://software.sci.utah.edu

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
 *  TypeMapImpl.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 */

#include <Core/CCA/spec/sci_sidl.h>
#include <SCIRun/Core/TypeMapImpl.h>
#include <Core/Util/NotFinished.h>

namespace SCIRun {

  using namespace sci::cca;

  TypeMapImpl::TypeMapImpl()
  {
  }
  
  TypeMapImpl::~TypeMapImpl()
  {
  }

  // .sci.cca.TypeMapImpl .sci.cca.TypeMapImpl.cloneTypeMapImpl()
  TypeMap::pointer TypeMapImpl::cloneTypeMap()
  {
    NOT_FINISHED("method not implemented");
    TypeMapImpl *tm = new TypeMapImpl();
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
    return TypeMap::pointer(tm);  
  }

  // .sci.cca.TypeMapImpl .sci.cca.TypeMapImpl.cloneEmpty()
  TypeMap::pointer TypeMapImpl::cloneEmpty()
  {
    return TypeMap::pointer(new TypeMapImpl); 
  }

  // void .sci.cca.TypeMapImpl.remove(in string key)
  void
  TypeMapImpl::remove(const std::string& key)
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

  // array1<string, 1> .sci.cca.TypeMapImpl.getAllKeys(in .sci.cca.Type t)
  SSIDL::array1<std::string>
  TypeMapImpl::getAllKeys(Type type)
  {
    SSIDL::array1<std::string> temp;

    switch(type) {
    case String:
      stringMap.getAllKeys(temp);
      return temp;
    case Bool:
      boolMap.getAllKeys(temp);
      return temp;
    case Int:
      intMap.getAllKeys(temp);
      return temp;
    case Long:
      longMap.getAllKeys(temp);
      return temp;
    case Float:
      floatMap.getAllKeys(temp);
      return temp;
    case Double:
      doubleMap.getAllKeys(temp);
      return temp;
    case Fcomplex:
      fcomplexMap.getAllKeys(temp);
      return temp;
    case Dcomplex:
      dcomplexMap.getAllKeys(temp);
      return temp;
    case IntArray:
      intArrayMap.getAllKeys(temp);
      return temp;
    case LongArray:
      longArrayMap.getAllKeys(temp);
      return temp;
    case FloatArray:
      floatArrayMap.getAllKeys(temp);
      return temp;
    case DoubleArray:
      doubleArrayMap.getAllKeys(temp);
      return temp;
    case StringArray:
      stringArrayMap.getAllKeys(temp);
      return temp;
    case BoolArray:
      boolArrayMap.getAllKeys(temp);
      return temp;
    case FcomplexArray:
      fcomplexArrayMap.getAllKeys(temp);
      return temp;
    case DcomplexArray:
      dcomplexArrayMap.getAllKeys(temp);
      return temp;
    case None: // if type == None, get keys from all types
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

  // bool .sci.cca.TypeMapImpl.hasKey(in string key)
  bool
  TypeMapImpl::hasKey(const std::string& key)
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

  // .sci.cca.Type .sci.cca.TypeMapImpl.typeOf(in string key)
  Type
  TypeMapImpl::typeOf(const std::string& key)
  {
    if (stringMap.hasKey(key)) return String;
    if (boolMap.hasKey(key)) return Bool;
    if (intMap.hasKey(key)) return Int;
    if (longMap.hasKey(key)) return Long;
    if (floatMap.hasKey(key)) return Float;
    if (doubleMap.hasKey(key)) return Double;
    if (fcomplexMap.hasKey(key)) return Fcomplex;
    if (dcomplexMap.hasKey(key)) return Dcomplex;
    if (intArrayMap.hasKey(key)) return IntArray;
    if (longArrayMap.hasKey(key)) return LongArray;
    if (floatArrayMap.hasKey(key)) return FloatArray;
    if (doubleArrayMap.hasKey(key)) return DoubleArray;
    if (stringArrayMap.hasKey(key)) return StringArray;
    if (boolArrayMap.hasKey(key)) return BoolArray;
    if (fcomplexArrayMap.hasKey(key)) return FcomplexArray;
    if (dcomplexArrayMap.hasKey(key)) return DcomplexArray;
    return None;
  }
  
  template<class T>
  TypeMapImpl::TypeMapBase<T>::TypeMapBase(const TypeMapBase<T>& tmi)
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
  TypeMapImpl::TypeMapBase<T>&
  TypeMapImpl::TypeMapBase<T>::operator=(const TypeMapBase<T>& tmi)
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
  T TypeMapImpl::TypeMapBase<T>::get(const std::string& key, const T& dflt)
  {
    MapIter found = typeMap.find(key);
    if (found != typeMap.end()) {
      return found->second;
    }
    return dflt;
  }

  template<class T>
  void TypeMapImpl::TypeMapBase<T>::put(const std::string& key, const T& value)
  {
    typeMap[key] = value;
  }

  template<class T>
  void TypeMapImpl::TypeMapBase<T>::getAllKeys(SSIDL::array1<std::string>& array)
  {
    for (MapIter iter = typeMap.begin(); iter != typeMap.end(); iter++) {
      array.push_back(iter->first);
    }
  }

  template<class T>
  bool TypeMapImpl::TypeMapBase<T>::hasKey(const std::string& key)
  {
    MapIter found = typeMap.find(key);
    if (found != typeMap.end()) {
      return true;
    }
    return false;
  }

  template<class T>
  bool TypeMapImpl::TypeMapBase<T>::remove(const std::string& key)
  {
    MapIter found = typeMap.find(key);
    if (found != typeMap.end()) {
      typeMap.erase(found);
      return true;
    }
    return false;
  }

} // end namespace SCIRun
