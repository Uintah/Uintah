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


#include <Core/Geometry/IntVector.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Util/TypeDescription.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
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

const TypeDescription* get_type_description(IntVector*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("IntVector", IntVector::get_h_file_path(), "SCIRun");
  }
  return td;
}


} //end namespace SCIRun

ostream& operator<<(std::ostream& out, const SCIRun::IntVector& v)
{
  out << "[int " << v.x() << ", " << v.y() << ", " << v.z() << ']';
  return out;
}
