/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
using namespace std;
namespace SCIRun{

  class TypeMap : public sci::cca::TypeMap{
  public:
    typedef map<string, string> StringMap;
    

    TypeMap();
    virtual ~TypeMap();
    
    // .sci.cca.TypeMap .sci.cca.TypeMap.cloneTypeMap()
    virtual TypeMap::pointer cloneTypeMap();

    // .sci.cca.TypeMap .sci.cca.TypeMap.cloneEmpty()
    virtual TypeMap::pointer cloneEmpty();

    // int .sci.cca.TypeMap.getInt(in string key, in int dflt)throws .sci.cca.TypeMismatchException
    virtual int getInt(const ::std::string& key, int dflt);
    
      // long .sci.cca.TypeMap.getLong(in string key, in long dflt)throws .sci.cca.TypeMismatchException
    virtual long getLong(const ::std::string& key, long dflt);
    
      // float .sci.cca.TypeMap.getFloat(in string key, in float dflt)throws .sci.cca.TypeMismatchException
    virtual float getFloat(const ::std::string& key, float dflt);
    
    // double .sci.cca.TypeMap.getDouble(in string key, in double dflt)throws .sci.cca.TypeMismatchException
    virtual double getDouble(const ::std::string& key, double dflt);
    
    // ::std::complex<float>  .sci.cca.TypeMap.getFcomplex(in string key, in ::std::complex<float>  dflt)throws .sci.cca.TypeMismatchException
    virtual ::std::complex<float>  getFcomplex(const ::std::string& key, ::std::complex<float>  dflt);
    
      // ::std::complex<double>  .sci.cca.TypeMap.getDcomplex(in string key, in ::std::complex<double>  dflt)throws .sci.cca.TypeMismatchException
    virtual ::std::complex<double>  getDcomplex(const ::std::string& key, ::std::complex<double>  dflt);
    
    // string .sci.cca.TypeMap.getString(in string key, in string dflt)throws .sci.cca.TypeMismatchException
    virtual ::std::string getString(const ::std::string& key, const ::std::string& dflt);
    
    // bool .sci.cca.TypeMap.getBool(in string key, in bool dflt)throws .sci.cca.TypeMismatchException
    virtual bool getBool(const ::std::string& key, bool dflt);
    
    // array1< int, 1> .sci.cca.TypeMap.getIntArray(in string key, in array1< int, 1> dflt)throws .sci.cca.TypeMismatchException
    virtual ::SSIDL::array1< int> getIntArray(const ::std::string& key, const ::SSIDL::array1< int>& dflt);
    
    // array1< long, 1> .sci.cca.TypeMap.getLongArray(in string key, in array1< long, 1> dflt)throws .sci.cca.TypeMismatchException
    virtual ::SSIDL::array1< long> getLongArray(const ::std::string& key, const ::SSIDL::array1< long>& dflt);
    
    // array1< float, 1> .sci.cca.TypeMap.getFloatArray(in string key, in array1< float, 1> dflt)throws .sci.cca.TypeMismatchException
    virtual ::SSIDL::array1< float> getFloatArray(const ::std::string& key, const ::SSIDL::array1< float>& dflt);
    
    // array1< double, 1> .sci.cca.TypeMap.getDoubleArray(in string key, in array1< double, 1> dflt)throws .sci.cca.TypeMismatchException
    virtual ::SSIDL::array1< double> getDoubleArray(const ::std::string& key, const ::SSIDL::array1< double>& dflt);
      
    // array1< ::std::complex<float> , 1> .sci.cca.TypeMap.getFcomplexArray(in string key, in array1< ::std::complex<float> , 1> dflt)throws .sci.cca.TypeMismatchException
    virtual ::SSIDL::array1< ::std::complex<float> > getFcomplexArray(const ::std::string& key, const ::SSIDL::array1< ::std::complex<float> >& dflt);
    
    // array1< ::std::complex<double> , 1> .sci.cca.TypeMap.getDcomplexArray(in string key, in array1< ::std::complex<double> , 1> dflt)throws .sci.cca.TypeMismatchException
    virtual ::SSIDL::array1< ::std::complex<double> > getDcomplexArray(const ::std::string& key, const ::SSIDL::array1< ::std::complex<double> >& dflt);
    
    // array1< string, 1> .sci.cca.TypeMap.getStringArray(in string key, in array1< string, 1> dflt)throws .sci.cca.TypeMismatchException
    virtual ::SSIDL::array1< ::std::string> getStringArray(const ::std::string& key, const ::SSIDL::array1< ::std::string>& dflt);
    
    // array1< bool, 1> .sci.cca.TypeMap.getBoolArray(in string key, in array1< bool, 1> dflt)throws .sci.cca.TypeMismatchException
    virtual ::SSIDL::array1< bool> getBoolArray(const ::std::string& key, const ::SSIDL::array1< bool>& dflt);
    
    // void .sci.cca.TypeMap.putInt(in string key, in int value)
    virtual void putInt(const ::std::string& key, int value);
    
    // void .sci.cca.TypeMap.putLong(in string key, in long value)
    virtual void putLong(const ::std::string& key, long value);
    
    // void .sci.cca.TypeMap.putFloat(in string key, in float value)
    virtual void putFloat(const ::std::string& key, float value);
    
    // void .sci.cca.TypeMap.putDouble(in string key, in double value)
    virtual void putDouble(const ::std::string& key, double value);
    
    // void .sci.cca.TypeMap.putFcomplex(in string key, in ::std::complex<float>  value)
    virtual void putFcomplex(const ::std::string& key, ::std::complex<float>  value);
    
    // void .sci.cca.TypeMap.putDcomplex(in string key, in ::std::complex<double>  value)
    virtual void putDcomplex(const ::std::string& key, ::std::complex<double>  value);

    // void .sci.cca.TypeMap.putString(in string key, in string value)
    virtual void putString(const ::std::string& key, const ::std::string& value);
    
    // void .sci.cca.TypeMap.putBool(in string key, in bool value)
    virtual void putBool(const ::std::string& key, bool value);
    
    // void .sci.cca.TypeMap.putIntArray(in string key, in array1< int, 1> value)
    virtual void putIntArray(const ::std::string& key, const ::SSIDL::array1< int>& value);

    // void .sci.cca.TypeMap.putLongArray(in string key, in array1< long, 1> value)
    virtual void putLongArray(const ::std::string& key, const ::SSIDL::array1< long>& value);
    
    // void .sci.cca.TypeMap.putFloatArray(in string key, in array1< float, 1> value)
    virtual void putFloatArray(const ::std::string& key, const ::SSIDL::array1< float>& value);
    
    // void .sci.cca.TypeMap.putDoubleArray(in string key, in array1< double, 1> value)
    virtual void putDoubleArray(const ::std::string& key, const ::SSIDL::array1< double>& value);
    
    // void .sci.cca.TypeMap.putFcomplexArray(in string key, in array1< ::std::complex<float> , 1> value)
    virtual void putFcomplexArray(const ::std::string& key, const ::SSIDL::array1< ::std::complex<float> >& value);
    
    // void .sci.cca.TypeMap.putDcomplexArray(in string key, in array1< ::std::complex<double> , 1> value)
    virtual void putDcomplexArray(const ::std::string& key, const ::SSIDL::array1< ::std::complex<double> >& value);
    
    // void .sci.cca.TypeMap.putStringArray(in string key, in array1< string, 1> value)
    virtual void putStringArray(const ::std::string& key, const ::SSIDL::array1< ::std::string>& value);
    
    // void .sci.cca.TypeMap.putBoolArray(in string key, in array1< bool, 1> value)
    virtual void putBoolArray(const ::std::string& key, const ::SSIDL::array1< bool>& value);
    
    // void .sci.cca.TypeMap.remove(in string key)
    virtual void remove(const ::std::string& key);

    // array1< string, 1> .sci.cca.TypeMap.getAllKeys(in .sci.cca.Type t)
    virtual ::SSIDL::array1< ::std::string> getAllKeys(sci::cca::Type t);
    
    // bool .sci.cca.TypeMap.hasKey(in string key)
    virtual bool hasKey(const ::std::string& key);
    
    // .sci.cca.Type .sci.cca.TypeMap.typeOf(in string key)
    virtual sci::cca::Type typeOf(const ::std::string& key);
  private:
    StringMap stringMap;
    



  };

} //SCIRun namespace


#endif
