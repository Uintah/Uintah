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

#include <Core/CCA/SSIDL/sidl_sidl.h>
#include <Core/Util/NotFinished.h>

using SSIDL::io::Serializer;

void Serializer::packBool(const std::string& key, bool value)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packBool(in string key, in bool value)throws .SSIDL.io.IOException");
}

void Serializer::packChar(const std::string& key, char value)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packChar(in string key, in char value)throws .SSIDL.io.IOException");
}

void Serializer::packInt(const std::string& key, int value)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packInt(in string key, in int value)throws .SSIDL.io.IOException");
}

void Serializer::packLong(const std::string& key, long value)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packLong(in string key, in long value)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packOpaque(in string key, in void* value)throws .SSIDL.io.IOException
void Serializer::packOpaque(const std::string& key, void* value)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packOpaque(in string key, in void* value)throws .SSIDL.io.IOException");
}

void Serializer::packFloat(const std::string& key, float value)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packFloat(in string key, in float value)throws .SSIDL.io.IOException");
}

void Serializer::packDouble(const std::string& key, double value)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packDouble(in string key, in double value)throws .SSIDL.io.IOException");
}

void Serializer::packFcomplex(const std::string& key, const std::complex<float>&  value)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packFcomplex(in string key, in complex<float>  value)throws .SSIDL.io.IOException");
}

void Serializer::packDcomplex(const std::string& key, const std::complex<double>&  value)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packDcomplex(in string key, in complex<double>  value)throws .SSIDL.io.IOException");
}

void Serializer::packString(const std::string& key, const std::string& value)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packString(in string key, in string value)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packSerializable(in string key, in .SSIDL.io.Serializable value)throws .SSIDL.io.IOException
void Serializer::packSerializable(const std::string& key, const Serializable::pointer& value)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packSerializable(in string key, in .SSIDL.io.Serializable value)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packBoolArray(in string key, in array1< bool, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException
void Serializer::packBoolArray(const std::string& key, const ::SSIDL::array1< bool>& value, int ordering, int dimen, bool reuse_array)
{
  NOT_FINISHED(" void .SSIDL.io.Serializer.packBoolArray(in string key, in array1< bool, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packCharArray(in string key, in array1< char, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException
void Serializer::packCharArray(const std::string& key, const ::SSIDL::array1< char>& value, int ordering, int dimen, bool reuse_array)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packCharArray(in string key, in array1< char, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packIntArray(in string key, in array1< int, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException
void Serializer::packIntArray(const std::string& key, const ::SSIDL::array1< int>& value, int ordering, int dimen, bool reuse_array)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packIntArray(in string key, in array1< int, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packLongArray(in string key, in array1< long, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException
void Serializer::packLongArray(const std::string& key, const ::SSIDL::array1< long>& value, int ordering, int dimen, bool reuse_array)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packLongArray(in string key, in array1< long, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packOpaqueArray(in string key, in array1< void*, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException
void Serializer::packOpaqueArray(const std::string& key, const ::SSIDL::array1< void*>& value, int ordering, int dimen, bool reuse_array)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packOpaqueArray(in string key, in array1< void*, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packFloatArray(in string key, in array1< float, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException
void Serializer::packFloatArray(const std::string& key, const ::SSIDL::array1< float>& value, int ordering, int dimen, bool reuse_array)
{
  NOT_FINISHED(" void .SSIDL.io.Serializer.packFloatArray(in string key, in array1< float, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packDoubleArray(in string key, in array1< double, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException
void Serializer::packDoubleArray(const std::string& key, const ::SSIDL::array1< double>& value, int ordering, int dimen, bool reuse_array)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packDoubleArray(in string key, in array1< double, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packFcomplexArray(in string key, in array1< complex<float> , 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException
void Serializer::packFcomplexArray(const std::string& key, const ::SSIDL::array1< std::complex<float> >& value, int ordering, int dimen, bool reuse_array)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packFcomplexArray(in string key, in array1< complex<float> , 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packDcomplexArray(in string key, in array1< complex<double> , 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException
void Serializer::packDcomplexArray(const std::string& key, const ::SSIDL::array1< std::complex<double> >& value, int ordering, int dimen, bool reuse_array)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packDcomplexArray(in string key, in array1< complex<double> , 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packStringArray(in string key, in array1< string, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException
void Serializer::packStringArray(const std::string& key, const ::SSIDL::array1< std::string>& value, int ordering, int dimen, bool reuse_array)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packStringArray(in string key, in array1< string, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException");
}

// void .SSIDL.io.Serializer.packSerializableArray(in string key, in array1< .SSIDL.io.Serializable, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException
void Serializer::packSerializableArray(const std::string& key, const ::SSIDL::array1< SSIDL::io::Serializable::pointer >& value, int ordering, int dimen, bool reuse_array)
{
  NOT_FINISHED("void .SSIDL.io.Serializer.packSerializableArray(in string key, in array1< .SSIDL.io.Serializable, 1> value, in int ordering, in int dimen, in bool reuse_array)throws .SSIDL.io.IOException");
}
