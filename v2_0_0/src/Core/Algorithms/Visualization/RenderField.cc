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
#include <Core/Geom/GeomCone.h>
#include <Core/Geom/GeomBox.h>
#include <Core/Geom/GeomTransform.h>

namespace SCIRun {

RenderFieldBase::RenderFieldBase() :
  node_switch_(0),
  edge_switch_(0),
  face_switch_(0)
{}

RenderFieldBase::~RenderFieldBase()
{}

CompileInfoHandle
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


void 
RenderFieldBase::add_axis(const Point &p0, double scale, GeomLines *lines)
{
  static const Vector x(1., 0., 0.);
  static const Vector y(0., 1., 0.);
  static const Vector z(0., 0., 1.);

  Point p1 = p0 + x * scale;
  Point p2 = p0 - x * scale;
  lines->add(p1, p2);

  p1 = p0 + y * scale;
  p2 = p0 - y * scale;
  lines->add(p1, p2);

  p1 = p0 + z * scale;
  p2 = p0 - z * scale;
  lines->add(p1, p2);
}


void 
RenderFieldBase::add_axis(const Point &p0, double scale, GeomLines *lines,
			  double val)
{
  static const Vector x(1., 0., 0.);
  static const Vector y(0., 1., 0.);
  static const Vector z(0., 0., 1.);

  Point p1 = p0 + x * scale;
  Point p2 = p0 - x * scale;
  lines->add(p1, val, p2, val);

  p1 = p0 + y * scale;
  p2 = p0 - y * scale;
  lines->add(p1, val, p2, val);

  p1 = p0 + z * scale;
  p2 = p0 - z * scale;
  lines->add(p1, val, p2, val);
}


void 
RenderFieldBase::add_axis(const Point &p0, double scale, GeomLines *lines,
			  const MaterialHandle &vcol)
{
  static const Vector x(1., 0., 0.);
  static const Vector y(0., 1., 0.);
  static const Vector z(0., 0., 1.);

  Point p1 = p0 + x * scale;
  Point p2 = p0 - x * scale;
  lines->add(p1, vcol, p2, vcol);

  p1 = p0 + y * scale;
  p2 = p0 - y * scale;
  lines->add(p1, vcol, p2, vcol);

  p1 = p0 + z * scale;
  p2 = p0 - z * scale;
  lines->add(p1, vcol, p2, vcol);
}



RenderVectorFieldBase::RenderVectorFieldBase()
{}

RenderVectorFieldBase::~RenderVectorFieldBase()
{}

CompileInfoHandle
RenderVectorFieldBase::get_compile_info(const TypeDescription *vftd,
					const TypeDescription *cftd,
					const TypeDescription *ltd)
{
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("RenderVectorField");
  static const string base_class_name("RenderVectorFieldBase");

  CompileInfo *rval = scinew CompileInfo(template_class_name + "." +
					 vftd->get_filename() + "." +
					 cftd->get_filename() + "." +
					 ltd->get_filename() + ".",
					 base_class_name, 
					 template_class_name, 
					 vftd->get_name() + ", " +
					 cftd->get_name() + ", " +
					 ltd->get_name());

  rval->add_include(include_path);
  vftd->fill_compile_info(rval);
  cftd->fill_compile_info(rval);
  return rval;
}


RenderTensorFieldBase::RenderTensorFieldBase()
{}

RenderTensorFieldBase::~RenderTensorFieldBase()
{}

CompileInfoHandle
RenderTensorFieldBase::get_compile_info(const TypeDescription *vftd,
					const TypeDescription *cftd,
					const TypeDescription *ltd)
{
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("RenderTensorField");
  static const string base_class_name("RenderTensorFieldBase");

  CompileInfo *rval = scinew CompileInfo(template_class_name + "." +
					 vftd->get_filename() + "." +
					 cftd->get_filename() + "." +
					 ltd->get_filename() + ".",
					 base_class_name, 
					 template_class_name, 
					 vftd->get_name() + ", " +
					 cftd->get_name() + ", " +
					 ltd->get_name());

  rval->add_include(include_path);
  vftd->fill_compile_info(rval);
  cftd->fill_compile_info(rval);
  return rval;
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
to_double(const Tensor &in, double &out)
{
  return false;
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
	 MaterialHandle &, const string &, double, bool, bool)
{
  return false;
}

template <>
bool 
add_data(const Point &p, const Vector &d, GeomArrows *arrows, 
	 MaterialHandle &mat, const string &, double sf, bool normalize,
	 bool bidirectional)
{
  Vector v(d);
  if (v.length2() > 1.e-12)
  {
    if (normalize) { v.safe_normalize(); }
    arrows->add(p, v*sf, mat, mat, mat);
    if (bidirectional) arrows->add(p, -v*sf, mat, mat, mat);
    return true;
  }
  return false;
}


void 
RenderTensorFieldBase::add_item(GeomGroup *g,
				GeomHandle glyph,
				const Point &p, Tensor &t,
				double scale,
				bool colorize)
{
  Vector ecolor;
  GeomTransform *gt = 0;

  // don't render glyphs that are too small
  if ((t.mat_[0][0] + t.mat_[1][1] + t.mat_[2][2]) < 0.001) return;

  if (t.have_eigens())
  {
    const Vector &e1 = t.get_eigenvector1();
    const Vector &e2 = t.get_eigenvector2();
    const Vector &e3 = t.get_eigenvector3();

    double v1, v2, v3;
    t.get_eigenvalues(v1, v2, v3);

    static const Point origin(0.0, 0.0, 0.0);
    Transform trans(origin, e1, e2, e3);
    trans.post_scale(Vector(fabs(v1), fabs(v2), fabs(v3)) * scale);
    trans.pre_translate(p.asVector());

    gt = scinew GeomTransform(glyph, trans);
    ecolor = e1;
  }
  else
  {
    const Vector v0 = Vector(t.mat_[0][0], t.mat_[0][1], t.mat_[0][2]) * scale;
    const Vector v1 = Vector(t.mat_[1][0], t.mat_[1][1], t.mat_[1][2]) * scale;
    const Vector v2 = Vector(t.mat_[2][0], t.mat_[2][1], t.mat_[2][2]) * scale;

    static const Point origin(0.0, 0.0, 0.0);
    Transform trans(origin, v0, v1, v2);
    trans.pre_translate(p.asVector());

    gt = scinew GeomTransform(glyph, trans);
    colorize = false;
  }

  if (colorize)
  {
    ecolor.normalize();
    double rr = fabs(ecolor.x());
    double gg = fabs(ecolor.y());
    double bb = fabs(ecolor.z());

    Color rgb1(rr, gg, bb);
    HSVColor hsv(rgb1);
    hsv[1] = 0.7;
    hsv[2] = 1.0;
    Color rgb2(hsv);
    MaterialHandle mh = scinew Material(rgb2);
    g->add(scinew GeomMaterial(gt, mh));
  }
  else
  {
    g->add(gt);
  }
}



void 
RenderTensorFieldBase::add_super_quadric(GeomGroup *g,
					 MaterialHandle mat,
					 const Point &p, Tensor &t,
					 double scale, int reso,
					 bool colorize)
{
  double v1, v2, v3;
  t.get_eigenvalues(v1, v2, v3);

  const double cl = (v1 - v2) / (v1 + v2 + v3);
  const double cp = 2.0 * (v2 - v3) / (v1 + v2 + v3);

  double qA, qB;
  int axis;
  if (cl > cp)
  {
    axis = 0;
    qA = pow((1.0 - cp), 3.5);  // Magic 3.5, adds emphasis to edges.
    qB = pow((1.0 - cl), 3.5);
  }
  else
  {
    axis = 2;
    qA = pow((1.0 - cl), 3.5);
    qB = pow((1.0 - cp), 3.5);
  }

  GeomHandle glyph = scinew GeomSuperquadric(axis, qA, qB, reso, reso);

  if (mat.get_rep())
  {
    glyph = scinew GeomMaterial(glyph, mat);
  }

  add_item(g, glyph, p, t, scale, colorize);
}



RenderScalarFieldBase::RenderScalarFieldBase()
{}

RenderScalarFieldBase::~RenderScalarFieldBase()
{}

CompileInfoHandle
RenderScalarFieldBase::get_compile_info(const TypeDescription *vftd,
					const TypeDescription *cftd,
					const TypeDescription *ltd)
{
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("RenderScalarField");
  static const string base_class_name("RenderScalarFieldBase");

  CompileInfo *rval = scinew CompileInfo(template_class_name + "." +
					 vftd->get_filename() + "." +
					 cftd->get_filename() + "." +
					 ltd->get_filename() + ".",
					 base_class_name, 
					 template_class_name, 
					 vftd->get_name() + ", " +
					 cftd->get_name() + ", " +
					 ltd->get_name());

  rval->add_include(include_path);
  vftd->fill_compile_info(rval);
  cftd->fill_compile_info(rval);
  return rval;
}

} // end namespace SCIRun
