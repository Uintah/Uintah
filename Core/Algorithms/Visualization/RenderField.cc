//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : RenderField.cc
//    Author : Martin Cole
//    Date   : Tue May 22 10:57:12 2001

#include <Core/Algorithms/Visualization/RenderField.h>
#include <Core/Geom/GeomArrows.h>

namespace SCIRun {

RenderFieldBase::~RenderFieldBase()
{}

CompileInfo *
RenderFieldBase::get_compile_info(const TypeDescription *td) {
  CompileInfo *rval = scinew CompileInfo(dyn_file_name(td), 
					 base_class_name(), 
					 template_class_name(), 
					 td->get_name());
  rval->add_include(get_h_file_path());
  td->fill_compile_info(rval);
  return rval;
}

const string& 
RenderFieldBase::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

template <>
bool
to_double(const double&in, double &out)
{
  out = in;
  return true;
}

template <>
bool
to_double(const int&in, double &out)
{
  out = in;
  return true;
}

template <>
bool
to_double(const short&in, double &out)
{
  out = in;
  return true;
}

template <>
bool
to_double(const unsigned char&in, double &out)
{
  out = in;
  return true;
}

template <>
bool
to_double(const Vector &in, double &out)
{
  out = in.length();
  return true;
}

template <>
bool 
add_data(const Point &p, const Tensor &d, GeomArrows *arrows, 
	 GeomSwitch *dat_sw,
	 MaterialHandle &mat, const string &s, double sf, bool normalize)
{
  return false;
}

template <>
bool 
add_data(const Point &p, const Vector &d, GeomArrows *arrows, 
	 GeomSwitch *dat_sw,
	 MaterialHandle &mat, const string &s, double sf, bool normalize)
{
  Vector v(d);
  if (normalize) { v.normalize(); }
  arrows->add(p, v*sf, mat, mat, mat);
  return true;
}
} // end namespace SCIRun
