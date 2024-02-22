/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Geometry/IntVector.h>
#include <Core/Util/XMLUtils.h>

#include <iostream>
#include <string>

using namespace Uintah::UintahXML;

namespace Uintah {

IntVector
IntVector::fromString( const std::string & source )
{
  IntVector result;

  // Parse out the [num,num,num]

  std::string::size_type i1 = source.find("[");
  std::string::size_type i2 = source.find_first_of(",");
  std::string::size_type i3 = source.find_last_of(",");
  std::string::size_type i4 = source.find("]");
  
  std::string x_val(source,i1+1,i2-i1-1);
  std::string y_val(source,i2+1,i3-i2-1);
  std::string z_val(source,i3+1,i4-i3-1);

  validateType( x_val, INT_TYPE );
  validateType( y_val, INT_TYPE );
  validateType( z_val, INT_TYPE );
          
  result.x( atoi(x_val.c_str()) );
  result.y( atoi(y_val.c_str()) );
  result.z( atoi(z_val.c_str()) );

  return result;
}

std::ostream&
operator<<(std::ostream& out, const Uintah::IntVector& v)
{
  out << "[int " << v.x() << ", " << v.y() << ", " << v.z() << ']';
  return out;
}

} //end namespace Uintah

