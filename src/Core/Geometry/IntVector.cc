/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
#include <Core/Persistent/Persistent.h>
#include <Core/Util/TypeDescription.h>
#include <iostream>
using std::ostream;

namespace SCIRun{

void
Pio(Piostream& stream, IntVector& p)
{
  stream.begin_cheap_delim();
  Pio(stream, p.value_[0]);
  Pio(stream, p.value_[1]);
  Pio(stream, p.value_[2]);
  stream.end_cheap_delim();
}



const string& 
IntVector::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

const TypeDescription*
get_type_description(IntVector*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("IntVector", IntVector::get_h_file_path(), "SCIRun");
  }
  return td;
}

ostream&
operator<<(std::ostream& out, const SCIRun::IntVector& v)
{
  out << "[int " << v.x() << ", " << v.y() << ", " << v.z() << ']';
  return out;
}

} //end namespace SCIRun

