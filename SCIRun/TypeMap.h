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
 *  SCIRunFramework.h: An instance of the SCIRun framework
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
#include <string>

namespace SCIRun
{

/**
 * \class Map
 *
 * An implementation of the CCA TypeMap, which maps a string key to a value.
 * The value may be of any type enumerated in CCA::Type, which included native
 * C++ types as well as strings and SIDL arrays of native types.  Values must
 * be stored and retrieved using the method that corresponds to the value type.
 * For example, an integer value must be stored using putInt and retrieved
 * using putInt.
 *
 * \todo Many of the get and put methods are not yet implemented.
 */
class TypeMap : public sci::cca::TypeMap
{
public:
  typedef std::map<std::string, std::string> StringMap;
  typedef std::map<std::string, int> IntMap;
  
  TypeMap();
  virtual ~TypeMap();
  
  /** Returns an exact copy of this TypeMap. */
  virtual TypeMap::pointer cloneTypeMap();
  
  /** Returns a new, empty TypeMap, i.e. one with no key/value pairs. */
  virtual TypeMap::pointer cloneEmpty();
  
  /** Returns the value associated with string \em key.  If \em key is
      not found, return the default value \em dflt. */
  virtual int getInt(const std::string& key, int dflt);
  virtual long getLong(const std::string& key, long dflt);
  virtual float getFloat(const std::string& key, float dflt);
  virtual double getDouble(const std::string& key, double dflt);
  virtual std::complex<float>  getFcomplex(const std::string& key,
                                             std::complex<float>  dflt);
  virtual std::complex<double>  getDcomplex(const std::string& key,
                                              std::complex<double>  dflt);
  virtual std::string getString(const std::string& key,
                                  const std::string& dflt);
  virtual bool getBool(const std::string& key, bool dflt);
  virtual SSIDL::array1< int> getIntArray(const std::string& key,
                                             const SSIDL::array1< int>& dflt);
  virtual SSIDL::array1< long> getLongArray(const std::string& key,
                                            const SSIDL::array1< long>& dflt);
  virtual SSIDL::array1< float> getFloatArray(const std::string& key,
                                           const SSIDL::array1< float>& dflt);
  virtual SSIDL::array1< double> getDoubleArray(const std::string& key,
                                          const SSIDL::array1< double>& dflt);
  virtual SSIDL::array1< std::complex<float> >
     getFcomplexArray( const std::string& key,
                       const SSIDL::array1< std::complex<float> >& dflt);
  virtual SSIDL::array1< std::complex<double> >
     getDcomplexArray(const std::string& key,
                      const SSIDL::array1< std::complex<double> >& dflt);
  virtual SSIDL::array1< std::string> getStringArray(const std::string& key,
                                    const SSIDL::array1< std::string>& dflt);
  virtual SSIDL::array1< bool> getBoolArray(const std::string& key,
                                              const SSIDL::array1< bool>& dflt);

  /** Assign a value to a key \em key.  If \em key already exists, the current
      value will be overwritten by the new value. */
  virtual void putInt(const std::string& key, int value);
  virtual void putLong(const std::string& key, long value);
  virtual void putFloat(const std::string& key, float value);
  virtual void putDouble(const std::string& key, double value);
  virtual void putFcomplex(const std::string& key, std::complex<float> value);
  virtual void putDcomplex(const std::string& key, std::complex<double>  value);
  virtual void putString(const std::string& key, const std::string& value);
  virtual void putBool(const std::string& key, bool value);
  virtual void putIntArray(const std::string& key,
                           const SSIDL::array1< int>& value);
  virtual void putLongArray(const std::string& key,
                            const SSIDL::array1< long>& value);
  virtual void putFloatArray(const std::string& key,
                             const SSIDL::array1< float>& value);
  virtual void putDoubleArray(const std::string& key,
                              const SSIDL::array1< double>& value);
  virtual void putFcomplexArray(const std::string& key,
                          const SSIDL::array1< std::complex<float> >& value);
  virtual void putDcomplexArray(const std::string& key, const SSIDL::array1< std::complex<double> >& value);
  virtual void putStringArray(const std::string& key, const SSIDL::array1< std::string>& value);
  virtual void putBoolArray(const std::string& key, const SSIDL::array1< bool>& value);

  virtual void remove(const std::string& key);

  virtual SSIDL::array1< std::string> getAllKeys(sci::cca::Type t);

  virtual bool hasKey(const std::string& key);
  
  virtual sci::cca::Type typeOf(const std::string& key);

private:
  StringMap stringMap;
  IntMap intMap;  
};

} //SCIRun namespace


#endif
