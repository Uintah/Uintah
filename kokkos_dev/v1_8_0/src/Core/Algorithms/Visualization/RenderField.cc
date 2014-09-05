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

RenderFieldBase::RenderFieldBase() :
  node_switch_(0),
  edge_switch_(0),
  face_switch_(0),
  data_switch_(0),
  vec_node_(0),
  def_mat_handle_(0),
  color_handle_(0),
  res_(0),
  mats_(0)
{}

RenderFieldBase::~RenderFieldBase()
{}

CompileInfo *
RenderFieldBase::get_compile_info(const TypeDescription *ftd,
				  const TypeDescription *ltd)
{
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("RenderField");
  static const string base_class_name("RenderFieldBase");

  CompileInfo *rval = scinew CompileInfo(template_class_name + "." +
					 ftd->get_filename() + "." +
					 ltd->get_filename() + ".",
					 base_class_name, 
					 template_class_name, 
					 ftd->get_name() + ", " +
					 ltd->get_name());

  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  return rval;
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
to_vector(const Vector &in, Vector &out)
{
  out = in;
  return true;
}

template <>
bool 
add_data(const Point &, const Tensor &, GeomArrows *, 
	 GeomSwitch *,
	 MaterialHandle &, const string &, double, bool, bool)
{
  return false;
}

template <>
bool 
add_data(const Point &p, const Vector &d, GeomArrows *arrows, 
	 GeomSwitch *,
	 MaterialHandle &mat, const string &, double sf, bool normalize,
	 bool bidirectional)
{
  Vector v(d);
  if (normalize) { v.safe_normalize(); }
  arrows->add(p, v*sf, mat, mat, mat);
  if (bidirectional) arrows->add(p, -v*sf, mat, mat, mat);
  return true;
}

} // end namespace SCIRun
