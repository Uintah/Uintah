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
 *  TypeMap.h:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 */

#ifndef SCIRun_TypeMap_h
#define SCIRun_TypeMap_h

#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/SSIDL/sidl_sidl.h>
#include <vector>
#include <map>
#include <list>
#include <string>

namespace SCIRun {

class TypeMap : public sci::cca::TypeMap {
public:
  TypeMap();
  virtual ~TypeMap();
    
  virtual TypeMap::pointer cloneTypeMap();
  virtual TypeMap::pointer cloneEmpty();

  // get, put methods throw sci.cca.TypeMismatchException which is not handled here!
  virtual int
  getInt(const std::string& key, int dflt) { return intMap.get(key, dflt); }

  virtual long
  getLong(const std::string& key, long dflt) { return longMap.get(key, dflt); }

  virtual float
  getFloat(const std::string& key, float dflt) { return floatMap.get(key, dflt); }

  virtual double
  getDouble(const std::string& key, double dflt) { return doubleMap.get(key, dflt); }

  virtual std::complex<float>
  getFcomplex(const std::string& key, const std::complex<float>& dflt)
  {
    return fcomplexMap.get(key, dflt);
  }

  virtual std::complex<double>
  getDcomplex(const std::string& key, const std::complex<double>& dflt)
  {
    return dcomplexMap.get(key, dflt);
  }

  virtual std::string
  getString(const std::string& key, const std::string& dflt) { return stringMap.get(key, dflt); }

  virtual bool
  getBool(const std::string& key, bool dflt) { return boolMap.get(key, dflt); }

  virtual SSIDL::array1<int>
  getIntArray(const std::string& key, const SSIDL::array1<int>& dflt)
  {
    return intArrayMap.get(key, dflt);
  }

  virtual SSIDL::array1<long>
  getLongArray(const std::string& key, const SSIDL::array1<long>& dflt)
  {
    return longArrayMap.get(key, dflt);
  }

  virtual SSIDL::array1<float>
  getFloatArray(const std::string& key, const SSIDL::array1<float>& dflt)
  {
    return floatArrayMap.get(key, dflt);
  }

  virtual SSIDL::array1<double>
  getDoubleArray(const std::string& key, const SSIDL::array1<double>& dflt)
  {
    return doubleArrayMap.get(key, dflt);
  }

  virtual SSIDL::array1<std::complex<float> >
  getFcomplexArray(const std::string& key, const SSIDL::array1<std::complex<float> >& dflt)
  {
    return fcomplexArrayMap.get(key, dflt);
  }

  virtual SSIDL::array1<std::complex<double> >
  getDcomplexArray(const std::string& key, const SSIDL::array1<std::complex<double> >& dflt)
  {
    return dcomplexArrayMap.get(key, dflt);
  }

  virtual SSIDL::array1<std::string>
  getStringArray(const std::string& key, const SSIDL::array1<std::string>& dflt)
  {
    return stringArrayMap.get(key, dflt);
  }

  virtual SSIDL::array1<bool>
  getBoolArray(const std::string& key, const SSIDL::array1<bool>& dflt)
  {
    return boolArrayMap.get(key, dflt);
  }

  virtual void
  putInt(const std::string& key, int value) { return intMap.put(key, value); }

  virtual void
  putLong(const std::string& key, long value) { return longMap.put(key, value); }

  virtual void
  putFloat(const std::string& key, float value) { return floatMap.put(key, value); }

  virtual void
  putDouble(const std::string& key, double value) { return doubleMap.put(key, value); }

  virtual void
  putFcomplex(const std::string& key, const std::complex<float>& value)
  {
    return fcomplexMap.put(key, value);
  }

  virtual void
  putDcomplex(const std::string& key, const std::complex<double>& value)
  {
    return dcomplexMap.put(key, value);
  }

  virtual void
  putString(const std::string& key, const std::string& value) { return stringMap.put(key, value); }

  virtual void
  putBool(const std::string& key, bool value) { return boolMap.put(key, value); }

  virtual void
  putIntArray(const std::string& key, const SSIDL::array1<int>& value)
  {
    return intArrayMap.put(key, value);
  }

  virtual void
  putLongArray(const std::string& key, const SSIDL::array1<long>& value)
  {
    return longArrayMap.put(key, value);
  }

  virtual void
  putFloatArray(const std::string& key, const SSIDL::array1<float>& value)
  {
    return floatArrayMap.put(key, value);
  }

  virtual void
  putDoubleArray(const std::string& key, const SSIDL::array1<double>& value)
  {
    return doubleArrayMap.put(key, value);
  }

  virtual void
  putFcomplexArray(const std::string& key, const SSIDL::array1<std::complex<float> >& value)
  {
    return fcomplexArrayMap.put(key, value);
  }

  virtual void
  putDcomplexArray(const std::string& key, const SSIDL::array1<std::complex<double> >& value)
  {
    return dcomplexArrayMap.put(key, value);
  }

  virtual void
  putStringArray(const std::string& key, const SSIDL::array1<std::string>& value)
  {
    return stringArrayMap.put(key, value);
  }

  virtual void
  putBoolArray(const std::string& key, const SSIDL::array1<bool>& value)
  {
    return boolArrayMap.put(key, value);
  }

  virtual void remove(const std::string& key);

  virtual SSIDL::array1<std::string> getAllKeys(sci::cca::Type type);

  virtual bool hasKey(const std::string& key);

  virtual sci::cca::Type typeOf(const std::string& key);

private:
  template<class T>
  class TypeMapImpl  {
  public:
    TypeMapImpl() {}
    ~TypeMapImpl() {}
    TypeMapImpl(const TypeMapImpl<T>&);
    TypeMapImpl<T>& operator=(const TypeMapImpl<T>&);

    T get(const std::string& key, const T& dflt);
    void put(const std::string& key, const T& value);
    void getAllKeys(SSIDL::array1<std::string>& array);
    bool hasKey(const std::string& key);
    bool remove(const std::string& key);

  private:
    typedef typename std::map<std::string, T>::iterator MapIter;
    typedef typename std::map<std::string, T>::const_iterator MapConstIter;
    std::map<std::string, T> typeMap;
  };
  TypeMapImpl<int> intMap;
  TypeMapImpl<long> longMap;
  TypeMapImpl<float> floatMap;
  TypeMapImpl<double> doubleMap;
  TypeMapImpl<std::string> stringMap;
  TypeMapImpl<bool> boolMap;
  TypeMapImpl<std::complex<float> > fcomplexMap;
  TypeMapImpl<std::complex<double> > dcomplexMap;
  TypeMapImpl<SSIDL::array1<int> > intArrayMap;
  TypeMapImpl<SSIDL::array1<long> > longArrayMap;
  TypeMapImpl<SSIDL::array1<float> > floatArrayMap;
  TypeMapImpl<SSIDL::array1<double> > doubleArrayMap;
  TypeMapImpl<SSIDL::array1<std::string> > stringArrayMap;
  TypeMapImpl<SSIDL::array1<bool> > boolArrayMap;
  TypeMapImpl<SSIDL::array1<std::complex<float> > > fcomplexArrayMap;
  TypeMapImpl<SSIDL::array1<std::complex<double> > > dcomplexArrayMap;
};

} //SCIRun namespace


#endif
