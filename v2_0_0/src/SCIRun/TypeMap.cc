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

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/TypeMap.h>
#include <map>
#include <string>

using namespace SCIRun;

TypeMap::TypeMap(){
}

TypeMap::~TypeMap(){
}

// .sci.cca.TypeMap .sci.cca.TypeMap.cloneTypeMap()
sci::cca::TypeMap::pointer
TypeMap::cloneTypeMap(){
  cerr<<"method not implemented\n";  
  return sci::cca::TypeMap::pointer(0);  
}

// .sci.cca.TypeMap .sci.cca.TypeMap.cloneEmpty()
sci::cca::TypeMap::pointer
TypeMap::cloneEmpty(){
  cerr<<"method not implemented\n";  
  return sci::cca::TypeMap::pointer(0); 
}


// int .sci.cca.TypeMap.getInt(in string key, in int dflt)throws .sci.cca.TypeMismatchException
int
TypeMap::getInt(const ::std::string& key, int dflt){
  IntMap::iterator found=intMap.find(key);
  if(found!=intMap.end()) return found->second;
  return dflt;
}

    
// long .sci.cca.TypeMap.getLong(in string key, in long dflt)throws .sci.cca.TypeMismatchException
long
TypeMap::getLong(const ::std::string& key, long dflt){
  cerr<<"method not implemented\n";  
  return dflt;  
}

    
// float .sci.cca.TypeMap.getFloat(in string key, in float dflt)throws .sci.cca.TypeMismatchException
float
TypeMap::getFloat(const ::std::string& key, float dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}


// double .sci.cca.TypeMap.getDouble(in string key, in double dflt)throws .sci.cca.TypeMismatchException
double
TypeMap::getDouble(const ::std::string& key, double dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}


// ::std::complex<float>  .sci.cca.TypeMap.getFcomplex(in string key, in ::std::complex<float>  dflt)throws .sci.cca.TypeMismatchException
::std::complex<float> 
TypeMap::getFcomplex(const ::std::string& key, ::std::complex<float>  dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}


      // ::std::complex<double>  .sci.cca.TypeMap.getDcomplex(in string key, in ::std::complex<double>  dflt)throws .sci.cca.TypeMismatchException
::std::complex<double> 
TypeMap::getDcomplex(const ::std::string& key, ::std::complex<double>  dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}

    
// string .sci.cca.TypeMap.getString(in string key, in string dflt)throws .sci.cca.TypeMismatchException
::std::string
TypeMap::getString(const ::std::string& key, const ::std::string& dflt){
  cerr<<"method not implemented\n";  
  StringMap::iterator found=stringMap.find(key);
  if(found!=stringMap.end()) return found->second;
  return dflt;
}


// bool .sci.cca.TypeMap.getBool(in string key, in bool dflt)throws .sci.cca.TypeMismatchException
bool
TypeMap::getBool(const ::std::string& key, bool dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}


// array1< int, 1> .sci.cca.TypeMap.getIntArray(in string key, in array1< int, 1> dflt)throws .sci.cca.TypeMismatchException
::SSIDL::array1< int>
TypeMap::getIntArray(const ::std::string& key, const ::SSIDL::array1< int>& dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}


// array1< long, 1> .sci.cca.TypeMap.getLongArray(in string key, in array1< long, 1> dflt)throws .sci.cca.TypeMismatchException
::SSIDL::array1< long>
TypeMap::getLongArray(const ::std::string& key, const ::SSIDL::array1< long>& dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}


// array1< float, 1> .sci.cca.TypeMap.getFloatArray(in string key, in array1< float, 1> dflt)throws .sci.cca.TypeMismatchException
::SSIDL::array1< float>
TypeMap::getFloatArray(const ::std::string& key, const ::SSIDL::array1< float>& dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}


// array1< double, 1> .sci.cca.TypeMap.getDoubleArray(in string key, in array1< double, 1> dflt)throws .sci.cca.TypeMismatchException
::SSIDL::array1< double>
TypeMap::getDoubleArray(const ::std::string& key, const ::SSIDL::array1< double>& dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}


// array1< ::std::complex<float> , 1> .sci.cca.TypeMap.getFcomplexArray(in string key, in array1< ::std::complex<float> , 1> dflt)throws .sci.cca.TypeMismatchException
::SSIDL::array1< ::std::complex<float> >
TypeMap::getFcomplexArray(const ::std::string& key, const ::SSIDL::array1< ::std::complex<float> >& dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}


// array1< ::std::complex<double> , 1> .sci.cca.TypeMap.getDcomplexArray(in string key, in array1< ::std::complex<double> , 1> dflt)throws .sci.cca.TypeMismatchException
::SSIDL::array1< ::std::complex<double> >
TypeMap::getDcomplexArray(const ::std::string& key, const ::SSIDL::array1< ::std::complex<double> >& dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}


// array1< string, 1> .sci.cca.TypeMap.getStringArray(in string key, in array1< string, 1> dflt)throws .sci.cca.TypeMismatchException
::SSIDL::array1< ::std::string>
TypeMap::getStringArray(const ::std::string& key, const ::SSIDL::array1< ::std::string>& dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}


// array1< bool, 1> .sci.cca.TypeMap.getBoolArray(in string key, in array1< bool, 1> dflt)throws .sci.cca.TypeMismatchException
::SSIDL::array1< bool>
TypeMap::getBoolArray(const ::std::string& key, const ::SSIDL::array1< bool>& dflt){
  cerr<<"method not implemented\n";  
  return dflt;
}


// void .sci.cca.TypeMap.putInt(in string key, in int value)
void
TypeMap::putInt(const ::std::string& key, int value){
  intMap.insert(IntMap::value_type(key, value));
  return;
}


// void .sci.cca.TypeMap.putLong(in string key, in long value)
void
TypeMap::putLong(const ::std::string& key, long value){
  cerr<<"method not implemented\n";  
  return;
}


// void .sci.cca.TypeMap.putFloat(in string key, in float value)
void
TypeMap::putFloat(const ::std::string& key, float value){
  cerr<<"method not implemented\n";  
  return;
}


// void .sci.cca.TypeMap.putDouble(in string key, in double value)
void
TypeMap::putDouble(const ::std::string& key, double value){
  cerr<<"method not implemented\n";  
  return;
}

    
// void .sci.cca.TypeMap.putFcomplex(in string key, in ::std::complex<float>  value)
void
TypeMap::putFcomplex(const ::std::string& key, ::std::complex<float>  value){
  cerr<<"method not implemented\n";  
  return;
}


// void .sci.cca.TypeMap.putDcomplex(in string key, in ::std::complex<double>  value)
void
TypeMap::putDcomplex(const ::std::string& key, ::std::complex<double>  value){
  cerr<<"method not implemented\n";  
  return;
}


// void .sci.cca.TypeMap.putString(in string key, in string value)
void
TypeMap::putString(const ::std::string& key, const ::std::string& value){
  stringMap.insert(StringMap::value_type(key, value));
  return;
}


// void .sci.cca.TypeMap.putBool(in string key, in bool value)
void
TypeMap::putBool(const ::std::string& key, bool value){
  cerr<<"method not implemented\n";  
  return;
}

    
// void .sci.cca.TypeMap.putIntArray(in string key, in array1< int, 1> value)
void
TypeMap::putIntArray(const ::std::string& key, const ::SSIDL::array1< int>& value){
  cerr<<"method not implemented\n";  
  return;
}


// void .sci.cca.TypeMap.putLongArray(in string key, in array1< long, 1> value)
void
TypeMap::putLongArray(const ::std::string& key, const ::SSIDL::array1< long>& value){
  cerr<<"method not implemented\n";  
  return;
}



// void .sci.cca.TypeMap.putFloatArray(in string key, in array1< float, 1> value)
void
TypeMap::putFloatArray(const ::std::string& key, const ::SSIDL::array1< float>& value){
  cerr<<"method not implemented\n";  
  return;
}


// void .sci.cca.TypeMap.putDoubleArray(in string key, in array1< double, 1> value)
void
TypeMap::putDoubleArray(const ::std::string& key, const ::SSIDL::array1< double>& value){
  cerr<<"method not implemented\n";  
  return;
}


// void .sci.cca.TypeMap.putFcomplexArray(in string key, in array1< ::std::complex<float> , 1> value)
void
TypeMap::putFcomplexArray(const ::std::string& key, const ::SSIDL::array1< ::std::complex<float> >& value){
  cerr<<"method not implemented\n";  
  return;
}


// void .sci.cca.TypeMap.putDcomplexArray(in string key, in array1< ::std::complex<double> , 1> value)
void
TypeMap::putDcomplexArray(const ::std::string& key, const ::SSIDL::array1< ::std::complex<double> >& value){
  cerr<<"method not implemented\n";  
  return;
}

    
// void .sci.cca.TypeMap.putStringArray(in string key, in array1< string, 1> value)
void
TypeMap::putStringArray(const ::std::string& key, const ::SSIDL::array1< ::std::string>& value){
  cerr<<"method not implemented\n";  
  return;
}

    
// void .sci.cca.TypeMap.putBoolArray(in string key, in array1< bool, 1> value)
void
TypeMap::putBoolArray(const ::std::string& key, const ::SSIDL::array1< bool>& value){
  cerr<<"method not implemented\n";  
  return;
}

    
// void .sci.cca.TypeMap.remove(in string key)
void
TypeMap:: remove(const ::std::string& key){
  cerr<<"method not implemented\n";  
  return;
}


// array1< string, 1> .sci.cca.TypeMap.getAllKeys(in .sci.cca.Type t)
::SSIDL::array1< ::std::string>
TypeMap::getAllKeys(sci::cca::Type t){
  cerr<<"method not implemented\n";  
  ::SSIDL::array1< ::std::string> temp;
  return temp;
}


// bool .sci.cca.TypeMap.hasKey(in string key)
bool
TypeMap::hasKey(const ::std::string& key){
  cerr<<"method not implemented\n";  
  return true;
}

// .sci.cca.Type .sci.cca.TypeMap.typeOf(in string key)
sci::cca::Type
TypeMap::typeOf(const ::std::string& key){
  cerr<<"method not implemented\n";
  return (sci::cca::Type)0;
}


