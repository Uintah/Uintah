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

//    File   : RenderField.cc
//    Author : Martin Cole
//    Date   : Tue May 22 10:57:12 2001

#include <sci_defs/teem_defs.h>

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

  string template_class_name0 = template_class_name;
  const string fldname = ftd->get_name();
  if (fldname.size() > 10 && fldname.substr(0, 10) == "ImageField")
  {
    template_class_name0 += "Image";
  }

  CompileInfo *rval = scinew CompileInfo(template_class_name0 + "." +
					 ftd->get_filename() + "." +
					 ltd->get_filename() + ".",
					 base_class_name, 
					 template_class_name0, 
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
to_double(const Tensor &/*in*/, double &/*out*/)
{
  return false;
}

template <>
bool
to_double(const string &/*in*/, double &/*out*/)
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
void
value_to_string(std::ostringstream &buffer, const unsigned char &value)
{
  buffer << (int)value;
}

template <>
void
value_to_string(std::ostringstream &buffer, const char &value)
{
  buffer << (int)value;
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

  Vector e1, e2, e3;
  t.get_eigenvectors(e1, e2, e3);
  double v1, v2, v3;
  t.get_eigenvalues(v1, v2, v3);
  // don't render glyphs that are too small
  if (v1 + v2 + v3 < 0.001) return;

  static const Point origin(0.0, 0.0, 0.0);
  Transform trans(origin, e1, e2, e3);
  trans.post_scale(Vector(fabs(v1), fabs(v2), fabs(v3)) * scale);
  trans.pre_translate(p.asVector());
  
  gt = scinew GeomTransform(glyph, trans);
  ecolor = e1;
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


double
RenderTensorFieldBase::map_emphasis(double old)
{
  if (old < 0.0) old = 0.0;
  else if (old > 1.0) old = 1.0;
  return tan(old * (M_PI / 2.0 * 0.999));
  // Map old 3.5 value onto new 0.825 value.
  //return tan(old * (atan(3.5) / (0.825 * 4.0)));
}


// emphasis was 3.5 and looked reasonable.  It should currently be
// computed via 'tan(VAL * M_PI / 2 * 0.999)', where VAL is [0,1].

void 
RenderTensorFieldBase::add_super_quadric(GeomGroup *g,
					 MaterialHandle mat,
					 const Point &p, Tensor &t,
					 double scale, int reso,
					 bool colorize,
					 double emphasis)
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
    qA = pow((1.0 - cp), emphasis);
    qB = pow((1.0 - cl), emphasis);
  }
  else
  {
    axis = 2;
    qA = pow((1.0 - cl), emphasis);
    qB = pow((1.0 - cp), emphasis);
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
