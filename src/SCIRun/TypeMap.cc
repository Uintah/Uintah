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

#include <Core/CCA/spec/sci_sidl.h>
#include <SCIRun/TypeMap.h>
#include <map>
#include <string>

namespace SCIRun {

TypeMap::TypeMap()
{ }

TypeMap::~TypeMap()
{ }

// .sci.cca.TypeMap .sci.cca.TypeMap.cloneTypeMap()
sci::cca::TypeMap::pointer TypeMap::cloneTypeMap()
{
  std::cerr<<"method not implemented" << std::endl;  
  return sci::cca::TypeMap::pointer(0);  
}

// .sci.cca.TypeMap .sci.cca.TypeMap.cloneEmpty()
sci::cca::TypeMap::pointer TypeMap::cloneEmpty()
{
  std::cerr<<"method not implemented" << std::endl;  
  return sci::cca::TypeMap::pointer(0); 
}

// int .sci.cca.TypeMap.getInt(in string key, in int dflt)throws .sci.cca.TypeMismatchException
int
TypeMap::getInt(const std::string& key, int dflt)
{
  IntMap::iterator found = intMap.find(key);
  if (found != intMap.end()) {
    return found->second;
  }
  return dflt;
}
    
// long .sci.cca.TypeMap.getLong(in string key, in long dflt)throws .sci.cca.TypeMismatchException
long
TypeMap::getLong(const std::string& key, long dflt)
{
  LongMap::iterator found = longMap.find(key);
  if (found != longMap.end()) {
    return found->second;
  }
  return dflt;  
}
   
// float .sci.cca.TypeMap.getFloat(in string key, in float dflt)throws .sci.cca.TypeMismatchException
float
TypeMap::getFloat(const std::string& key, float dflt)
{
    FloatMap::iterator found = floatMap.find(key);
    if (found != floatMap.end()) {
        return found->second;
    }
    return dflt;
}

// double .sci.cca.TypeMap.getDouble(in string key, in double dflt)throws .sci.cca.TypeMismatchException
double
TypeMap::getDouble(const std::string& key, double dflt)
{
    DoubleMap::iterator found = doubleMap.find(key);
    if (found != doubleMap.end()) {
        return found->second;
    }
    return dflt;
}

// std::complex<float>  .sci.cca.TypeMap.getFcomplex(in string key, in std::complex<float>  dflt)throws .sci.cca.TypeMismatchException
std::complex<float>
TypeMap::getFcomplex(const std::string& key, std::complex<float>  dflt)
{
  std::cerr<<"method not implemented" << std::endl;  
  return dflt;
}

// std::complex<double>  .sci.cca.TypeMap.getDcomplex(in string key, in std::complex<double>  dflt)throws .sci.cca.TypeMismatchException
std::complex<double> 
TypeMap::getDcomplex(const std::string& key, std::complex<double>  dflt)
{
  std::cerr<<"method not implemented" << std::endl;  
  return dflt;
}
    
// string .sci.cca.TypeMap.getString(in string key, in string dflt)throws .sci.cca.TypeMismatchException
std::string
TypeMap::getString(const std::string& key, const std::string& dflt)
{
  StringMap::iterator found = stringMap.find(key);
  if (found != stringMap.end()) {
    return found->second;
  }
  return dflt;
}

// bool .sci.cca.TypeMap.getBool(in string key, in bool dflt)throws .sci.cca.TypeMismatchException
bool
TypeMap::getBool(const std::string& key, bool dflt)
{
    BoolMap::iterator found = boolMap.find(key);
    if (found != boolMap.end()) {
        return found->second;
    }
    return dflt;
}

// array1< int, 1> .sci.cca.TypeMap.getIntArray(in string key, in array1< int, 1> dflt)throws .sci.cca.TypeMismatchException
::SSIDL::array1< int>
TypeMap::getIntArray(const std::string& key, const ::SSIDL::array1< int>& dflt){
  IntArrayMap::iterator found = intArrayMap.find(key);
  if (found != intArrayMap.end()) {
    return found->second;
  }
  return dflt;
}

// array1< long, 1> .sci.cca.TypeMap.getLongArray(in string key, in array1< long, 1> dflt)throws .sci.cca.TypeMismatchException
SSIDL::array1< long>
TypeMap::getLongArray(const std::string& key, const SSIDL::array1< long>& dflt)
{
  LongArrayMap::iterator found = longArrayMap.find(key);
  if (found != longArrayMap.end()) {
    return found->second;
  }
  return dflt;
}

// array1< float, 1> .sci.cca.TypeMap.getFloatArray(in string key, in array1< float, 1> dflt)throws .sci.cca.TypeMismatchException
SSIDL::array1< float>
TypeMap::getFloatArray(const std::string& key, const SSIDL::array1< float>& dflt)
{
  std::cerr<<"method not implemented" << std::endl;  
  return dflt;
}

// array1< double, 1> .sci.cca.TypeMap.getDoubleArray(in string key, in array1< double, 1> dflt)throws .sci.cca.TypeMismatchException
SSIDL::array1< double>
TypeMap::getDoubleArray(const std::string& key, const SSIDL::array1< double>& dflt)
{
  std::cerr<<"method not implemented" << std::endl;  
  return dflt;
}

// array1< std::complex<float> , 1> .sci.cca.TypeMap.getFcomplexArray(in string key, in array1< std::complex<float> , 1> dflt)throws .sci.cca.TypeMismatchException
SSIDL::array1< std::complex<float> >
TypeMap::getFcomplexArray(const std::string& key,
                          const SSIDL::array1< std::complex<float> >& dflt)
{
  std::cerr<<"method not implemented" << std::endl;  
  return dflt;
}

// array1< std::complex<double> , 1> .sci.cca.TypeMap.getDcomplexArray(in string key, in array1< std::complex<double> , 1> dflt)throws .sci.cca.TypeMismatchException
SSIDL::array1< std::complex<double> >
TypeMap::getDcomplexArray(const std::string& key,
                          const SSIDL::array1< std::complex<double> >& dflt)
{
  std::cerr<<"method not implemented" << std::endl;  
  return dflt;
}

// array1<string, 1> .sci.cca.TypeMap.getStringArray(in string key, in array1<string, 1> dflt)throws .sci.cca.TypeMismatchException
SSIDL::array1<std::string>
TypeMap::getStringArray(const std::string& key, const SSIDL::array1<std::string>& dflt)
{
    StringArrayMap::iterator found = stringArrayMap.find(key);
    if (found != stringArrayMap.end()) {
        return found->second;
    }
    return dflt;
}

// array1< bool, 1> .sci.cca.TypeMap.getBoolArray(in string key, in array1< bool, 1> dflt)throws .sci.cca.TypeMismatchException
SSIDL::array1< bool>
TypeMap::getBoolArray(const std::string& key, const SSIDL::array1< bool>& dflt)
{
  std::cerr<<"method not implemented" << std::endl;  
  return dflt;
}

// void .sci.cca.TypeMap.putInt(in string key, in int value)
void
TypeMap::putInt(const std::string& key, int value)
{
  // insert new value for key or
  // change existing value for key
  intMap[key] = value;
  return;
}

// void .sci.cca.TypeMap.putLong(in string key, in long value)
void
TypeMap::putLong(const std::string& key, long value)
{
  longMap[key] = value;
  return;
}

// void .sci.cca.TypeMap.putFloat(in string key, in float value)
void
TypeMap::putFloat(const std::string& key, float value)
{
  std::cerr<<"method not implemented" << std::endl;  
  return;
}

// void .sci.cca.TypeMap.putDouble(in string key, in double value)
void
TypeMap::putDouble(const std::string& key, double value)
{
  std::cerr<<"method not implemented" << std::endl;  
  return;
}

// void .sci.cca.TypeMap.putFcomplex(in string key, in std::complex<float>  value)
void
TypeMap::putFcomplex(const std::string& key, std::complex<float> value)
{
  std::cerr<<"method not implemented" << std::endl;  
  return;
}

// void .sci.cca.TypeMap.putDcomplex(in string key, in std::complex<double>  value)
void
TypeMap::putDcomplex(const std::string& key, std::complex<double> value)
{
  std::cerr<<"method not implemented" << std::endl;  
  return;
}

// void .sci.cca.TypeMap.putString(in string key, in string value)
void
TypeMap::putString(const std::string& key, const std::string& value)
{
  // insert new value for key or
  // change existing value for key
  stringMap[key] = value;
  return;
}

// void .sci.cca.TypeMap.putBool(in string key, in bool value)
void
TypeMap::putBool(const std::string& key, bool value)
{
    boolMap[key] = value;
    return;
}
   
// void .sci.cca.TypeMap.putIntArray(in string key, in array1< int, 1> value)
void
TypeMap::putIntArray(const std::string& key, const ::SSIDL::array1<int>& value) {
  //intArrayMap.insert(IntArrayMap::value_type(key, value));
  intArrayMap[key] = value;
  return;
}

// void .sci.cca.TypeMap.putLongArray(in string key, in array1< long, 1> value)
void
TypeMap::putLongArray(const std::string& key, const SSIDL::array1< long>& value)
{
  longArrayMap[key] = value;
  return;
}

// void .sci.cca.TypeMap.putFloatArray(in string key, in array1< float, 1> value)
void
TypeMap::putFloatArray(const std::string& key, const SSIDL::array1< float>& value)
{
  std::cerr<<"method not implemented" << std::endl;  
  return;
}

// void .sci.cca.TypeMap.putDoubleArray(in string key, in array1< double, 1> value)
void
TypeMap::putDoubleArray(const std::string& key, const SSIDL::array1< double>& value)
{
  std::cerr<<"method not implemented" << std::endl;  
  return;
}


// void .sci.cca.TypeMap.putFcomplexArray(in string key, in array1< std::complex<float> , 1> value)
void
TypeMap::putFcomplexArray(const std::string& key,
                          const SSIDL::array1< std::complex<float> >& value)
{
  std::cerr<<"method not implemented" << std::endl;  
  return;
}

// void .sci.cca.TypeMap.putDcomplexArray(in string key, in array1< std::complex<double> , 1> value)
void
TypeMap::putDcomplexArray(const std::string& key,
                          const SSIDL::array1< std::complex<double> >& value)
{
  std::cerr<<"method not implemented" << std::endl;  
  return;
}
    
// void .sci.cca.TypeMap.putStringArray(in string key, in array1< string, 1> value)
void
TypeMap::putStringArray(const std::string& key,
                        const SSIDL::array1< std::string>& value)
{
  //stringArrayMap.insert(StringArrayMap::value_type(key, value));
  stringArrayMap[key] = value;
  return;
}
    
// void .sci.cca.TypeMap.putBoolArray(in string key, in array1< bool, 1> value)
void
TypeMap::putBoolArray(const std::string& key, const SSIDL::array1< bool>& value)
{
  std::cerr<<"method not implemented" << std::endl;  
  return;
}

// void .sci.cca.TypeMap.remove(in string key)
void
TypeMap::remove(const std::string& key)
{
  std::cerr<<"method not implemented" << std::endl;
  return;
}

// array1< string, 1> .sci.cca.TypeMap.getAllKeys(in .sci.cca.Type t)
SSIDL::array1< std::string>
TypeMap::getAllKeys(sci::cca::Type t)
{
  std::cerr<<"method not implemented" << std::endl;  
  SSIDL::array1< std::string> temp;
  return temp;
}

// bool .sci.cca.TypeMap.hasKey(in string key)
bool
TypeMap::hasKey(const std::string& key)
{
  std::cerr<<"method not implemented" << std::endl;  
  return true;
}

// .sci.cca.Type .sci.cca.TypeMap.typeOf(in string key)
sci::cca::Type
TypeMap::typeOf(const std::string& key)
{
  std::cerr<<"method not implemented" << std::endl;
  return (sci::cca::Type)0;
}

} // end namespace SCIRun
