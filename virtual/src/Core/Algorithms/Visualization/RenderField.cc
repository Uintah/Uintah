/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
#include <Core/Thread/Time.h>
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
  if ( fldname.find( "ImageMesh" ) != string::npos )
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
    ecolor.safe_normalize();
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


bool 
render_field(FieldHandle fld_handle, RenderParams &params) 
{
  const TypeDescription *ftd = fld_handle->get_type_description();
  const TypeDescription *ltd = fld_handle->order_type_description();
  // description for just the data in the field

  // Get the Algorithm.
  CompileInfoHandle ci = RenderFieldBase::get_compile_info(ftd, ltd);

  if (!DynamicCompilation::compile(ci, params.dalgo_)) {
    return false;
  }
  
  params.renderer_ = (RenderFieldBase*)params.dalgo_.get_rep();
  if (! params.renderer_) {
    cerr << "Error: could not get algorithm in render_field" 
	 << endl;
    return false;
  }

  if (params.faces_normals_) {
    fld_handle->mesh()->synchronize(Mesh::NORMALS_E);
  }

  params.renderer_->render(fld_handle,
			   params.do_nodes_, 
			   params.do_edges_, 
			   params.do_faces_, 
			   params.color_map_, 
			   params.def_material_,
			   params.ndt_, 
			   params.edt_, 
			   params.ns_, 
			   params.es_, 
			   params.vscale_, 
			   params.normalize_vectors_,
			   params.node_resolution_, 
			   params.edge_resolution_,
			   params.faces_normals_,
			   params.nodes_transparency_,
			   params.edges_transparency_,
			   params.faces_transparency_,
			   params.nodes_usedefcolor_,
			   params.edges_usedefcolor_,
			   params.faces_usedefcolor_,
			   params.approx_div_,
			   params.faces_usetexture_);

  if (params.do_text_) {
    params.text_geometry_ = 
      params.renderer_->render_text(fld_handle,
				    params.color_map_.get_rep(),
				    params.text_use_default_color_,
				    params.text_backface_cull_,
				    params.text_fontsize_,
				    params.text_precision_,
				    params.text_render_locations_,
				    params.text_show_data_,
				    params.text_show_nodes_,
				    params.text_show_edges_,
				    params.text_show_faces_,
				    params.text_show_cells_);
  }
  return true;
}






GeomHandle 
RenderVectorFieldVirtual::render_data(FieldHandle vfld_handle,
						FieldHandle cfld_handle,
						ColorMapHandle cmap,
						MaterialHandle default_material,
						bool force_def_color,
						const string &display_mode,
						double scale,
						double linewidth,
						bool normalize,
						bool bidirectional,
						int resolution)
{
  Field *vfld = vfld_handle.get_rep();
  Field *cfld = cfld_handle.get_rep();

  GeomLines *lines = 0;
  GeomCones *cones = 0;
  GeomCappedCylinders *disks = 0;
  GeomSpheres *spheres = 0;
  const bool lines_p = (display_mode == "Lines");
  const bool needles_p = (display_mode == "Needles");
  const bool cones_p = (display_mode == "Cones");
  const bool arrows_p = (display_mode == "Arrows");
  const bool disks_p = (display_mode == "Disks");
  GeomHandle data_switch;
  if (lines_p)
  {
    lines = scinew GeomLines();
    data_switch = scinew GeomDL(lines);
  }
  else if (needles_p)
  {
    lines = scinew GeomTranspLines();
    data_switch = lines;
  }
  else if (cones_p)
  {
    cones = scinew GeomCones(resolution, scale/6.0);
    spheres = scinew GeomSpheres(scale/6.0, resolution, resolution);
    GeomGroup *grp = scinew GeomGroup();
    grp->add(cones);
    grp->add(spheres);
    data_switch = scinew GeomDL(grp);
  }
  else if (arrows_p)
  {
    lines = scinew GeomLines();
    cones = scinew GeomCones(resolution, scale * 0.15);
    spheres = scinew GeomSpheres(scale * 0.15, resolution, resolution);
    GeomGroup *grp = scinew GeomGroup();
    grp->add(lines);
    grp->add(cones);
    grp->add(spheres);
    data_switch = scinew GeomDL(grp);
  }
  else if (disks_p)
  {
    disks = scinew GeomCappedCylinders(resolution, scale);
    spheres = scinew GeomSpheres(scale * 0.75, resolution, resolution);
    GeomGroup *grp = scinew GeomGroup();
    grp->add(disks);
    grp->add(spheres);
    data_switch = scinew GeomDL(grp);
  }

  if (lines && linewidth)
  {
    lines->setLineWidth(linewidth);
  }

  // Use a default color?
  bool def_color = !(cmap.get_rep()) || force_def_color;
  bool vec_color = false;
  MaterialHandle vcol(0);
  
  FieldInformation cfi(cfld);
  if (def_color && cfi.is_vector() && !force_def_color)
  {
    def_color = false;
    vec_color = true;
    vcol = scinew Material();
    vcol->transparency = 1.0;
  }

  MaterialHandle opaque, transparent;
  if (needles_p)
  {
    if (def_color)
    {
      opaque = scinew Material(default_material->diffuse);
      opaque->transparency = default_material->transparency;
      transparent = scinew Material(default_material->diffuse);
      transparent->transparency = 0.0;
    }
    else
    {
      opaque = scinew Material(Color(1.0, 1.0, 1.0));
      opaque->transparency = 1.0;
      transparent = scinew Material(Color(1.0, 1.0, 1.0));
      transparent->transparency = 0.0;
    }
  }

  Mesh *mesh = vfld->mesh().get_rep();

  int field_order = vfld->basis_order();
  Mesh::size_type datasize = vfld->data_size();

  for (Mesh::index_type i=0;i < datasize; i++)
  {
    Vector val;
    vfld->get_value(val, i);

    Point p;
    if (field_order == 0)
    {
      mesh->get_center(p, Mesh::VElem::index_type(i));
    }
    else
    {
      mesh->get_center(p, Mesh::VNode::index_type(i));      
    }
    
    if (disks_p)
    {
      if (val.length2() > 1.0e-10)
      {
        if (normalize) { val.safe_normalize(); }
        val *= (scale / 6.0);
        
        if (def_color)
        {
          if (normalize)
          {
            disks->add(p+val, p-val);
          }
          else
          {
            disks->add_radius(p+val, p+val, 6*val.length());
          }
        }
        else if (vec_color)
        {
          Vector vtmp;
          cfld->get_value(vtmp, i);
          sciVectorToColor(vcol->diffuse, vtmp);
          if (normalize)
          {
            disks->add(p+val, vcol, p-val, vcol);
          }
          else
          {
            disks->add_radius(p+val, vcol, p-val, vcol, 6*val.length());
          }
        }
        else
        {
          double ctmpd;
          cfld->get_value(ctmpd, i);
          if (normalize)
          {
            disks->add(p+val, ctmpd, p-val, ctmpd);
          }
          else
          {
            disks->add_radius(p+val, ctmpd, p+val, ctmpd, 6*val.length());
          }
        }
      }
      else if (normalize)
      {
        if (def_color)
        {
          spheres->add(p);
        }
        else if (vec_color)
        {
          Vector vtmp;
          cfld->get_value(vtmp, i);
          sciVectorToColor(vcol->diffuse, vtmp);
          spheres->add(p, vcol);
        }
        else
        {
          double ctmpd;
          cfld->get_value(ctmpd, i);
          spheres->add(p, (float)ctmpd);
        }
      }
    }
    else if (cones_p)
    {
      if (val.length2() > 1.0e-10)
      {
        if (normalize) { val.safe_normalize(); }
        val *= scale;

        if (def_color)
        {
          if (normalize)
          {
            cones->add(p, p+val);
            if (bidirectional)
            {
              cones->add(p, p-val);
            }
          }
          else
          {
            const double len = val.length() / 6.0;
            cones->add_radius(p, p+val, len);
            if (bidirectional)
            {
              cones->add_radius(p, p-val, len);
            }
          }
        }
        else if (vec_color)
        {
          Vector vtmp;
          cfld->get_value(vtmp, i);
          sciVectorToColor(vcol->diffuse, vtmp);
          if (normalize)
          {
            cones->add(p, p+val, vcol);
            if (bidirectional)
            {
              cones->add(p, p-val, vcol);
            }
          }
          else
          {
            const double len = val.length() / 6.0;
            cones->add_radius(p, p+val, vcol, len);
            if (bidirectional)
            {
              cones->add_radius(p, p-val, vcol, len);
            }
          }
        }
        else
        {
          double ctmpd;
          cfld->get_value(ctmpd, i);
          if (normalize)
          {
            cones->add(p, p+val, ctmpd);
            if (bidirectional)
            {
              cones->add(p, p-val, ctmpd);
            }
          }
          else
          {
            const double len = val.length() / 6.0;
            cones->add_radius(p, p+val, ctmpd, len);
            if (bidirectional)
            {
              cones->add_radius(p, p-val, ctmpd, len);
            }
          }
        }
      }
      else if (normalize)
      {
        if (def_color)
        {
          spheres->add(p);
        }
        else if (vec_color)
        {
          Vector vtmp;
          cfld->get_value(vtmp, i);
          sciVectorToColor(vcol->diffuse, vtmp);
          spheres->add(p, vcol);
        }
        else
        {
          double ctmpd;
          cfld->get_value(ctmpd, i);
          spheres->add(p, (float)ctmpd);
        }
      }
    }
    else if (arrows_p)
    {
      if (val.length2() > 1.0e-10)
      {
        if (normalize) { val.safe_normalize(); }
        val *= scale;
        const Vector ltmp = val * 0.6;

        if (def_color)
        {
          if (normalize)
          {
            cones->add(p+ltmp, p+val);
            if (bidirectional)
            {
              cones->add(p-ltmp, p-val);
            }
          }
          else
          {
            const double len = val.length() * 0.15;
            cones->add_radius(p+ltmp, p+val, len);
            if (bidirectional)
            {
              cones->add_radius(p-ltmp, p-val, len);
            }
          }
        }
        else if (vec_color)
        {
          Vector vtmp;
          cfld->get_value(vtmp, i);
          sciVectorToColor(vcol->diffuse, vtmp);
          if (normalize)
          {
            cones->add(p+ltmp, p+val, vcol);
            if (bidirectional)
            {
              cones->add(p-ltmp, p-val, vcol);
            }
          }
          else
          {
            const double len = val.length() * 0.15;
            cones->add_radius(p+ltmp, p+val, vcol, len);
            if (bidirectional)
            {
              cones->add_radius(p-ltmp, p-val, vcol, len);
            }
          }
        }
        else
        {
          double ctmpd;
          cfld->get_value(ctmpd, i);
          if (normalize)
          {
            cones->add(p+ltmp, p+val, ctmpd);
            if (bidirectional)
            {
              cones->add(p-ltmp, p-val, ctmpd);
            }
          }
          else
          {
            const double len = val.length() * 0.15;
            cones->add_radius(p+ltmp, p+val, ctmpd, len);
            if (bidirectional)
            {
              cones->add_radius(p-ltmp, p-val, ctmpd, len);
            }
          }
        }

        if (bidirectional)
        {
          if (def_color)
          {
            lines->add(p - ltmp, p + ltmp);
          }
          else if (vec_color)
          {
            Vector vtmp;
            cfld->get_value(vtmp, i);
            sciVectorToColor(vcol->diffuse, vtmp);
            lines->add(p - ltmp, vcol, p + ltmp, vcol);
          }
          else
          {
            double ctmpd;
            cfld->get_value(ctmpd, i);
            lines->add(p - ltmp, ctmpd, p + ltmp, ctmpd);
          }
        }
        else
        {
          if (def_color)
          {
            lines->add(p, p + ltmp);
          }
          else if (vec_color)
          {
            Vector vtmp;
            cfld->get_value(vtmp, i);
            sciVectorToColor(vcol->diffuse, vtmp);
            lines->add(p, vcol, p + ltmp, vcol);
          }
          else
          {
            double ctmpd;
            cfld->get_value(ctmpd, i);
            lines->add(p, ctmpd, p + ltmp, ctmpd);
          }
        }
      }
      else if (normalize)
      {
        if (def_color)
        {
          spheres->add(p);
        }
        else if (vec_color)
        {
          Vector vtmp;
          cfld->get_value(vtmp, i);
          sciVectorToColor(vcol->diffuse, vtmp);
          spheres->add(p, vcol);
        }
        else
        {
          double ctmpd;
          cfld->get_value(ctmpd, i);
          spheres->add(p, (float)ctmpd);
        }
      }
    }
    else if (lines_p)
    {
      if (normalize) { val.safe_normalize(); }
      val *= scale;

      if (bidirectional)
      {
        if (def_color)
        {
          lines->add(p - val, p + val);
        }
        else if (vec_color)
        {
          Vector vtmp;
          cfld->get_value(vtmp, i);
          sciVectorToColor(vcol->diffuse, vtmp);
          lines->add(p - val, vcol, p + val, vcol);
        }
        else
        {
          double ctmpd;
          cfld->get_value(ctmpd, i);
          lines->add(p - val, ctmpd, p + val, ctmpd);
        }
      }
      else
      {
        if (def_color)
        {
          lines->add(p, p + val);
        }
        else if (vec_color)
        {
          Vector vtmp;
          cfld->get_value(vtmp, i);
          sciVectorToColor(vcol->diffuse, vtmp);
          lines->add(p, vcol, p + val, vcol);
        }
        else
        {
          double ctmpd;
          cfld->get_value(ctmpd, i);
          lines->add(p, ctmpd, p + val, ctmpd);
        }
      }
    }
    else // Needles
    {
      if (normalize) { val.safe_normalize(); }
      val *= scale;
  
      if (def_color)
      {
        lines->add(p, opaque, p + val, transparent);
        if (bidirectional)
        {
          lines->add(p, opaque, p - val, transparent);
        }
      }
      else if (vec_color)
      {
        Vector vtmp;
        cfld->get_value(vtmp, i);
        sciVectorToColor(vcol->diffuse, vtmp);
        transparent->diffuse = vcol->diffuse;
        
        lines->add(p, vcol, p + val, transparent);
        if (bidirectional)
        {
          lines->add(p, vcol, p - val, transparent);
        }
      }
      else
      {
        double ctmpd;
        cfld->get_value(ctmpd, i);
        lines->add(p, opaque, ctmpd, p + val, transparent, ctmpd);
        if (bidirectional)
        {
          lines->add(p, opaque, ctmpd, p - val, transparent, ctmpd);
        }
      }
    }
  }
  return data_switch;
}


GeomHandle 
RenderTensorFieldVirtual::render_data(FieldHandle vfld_handle,
						FieldHandle cfld_handle,
						ColorMapHandle cmap,
						MaterialHandle def_mat,
						bool force_def_color,
						const string &display_mode,
						double scale, 
						int resolution,
						double zo_emphasis)
{
  Field *vfld = vfld_handle.get_rep();
  Field *cfld = cfld_handle.get_rep(); 

  const bool box_p = (display_mode == "Boxes");
  const bool sphere_p = (display_mode == "Ellipsoids");
  const bool squad_p = (display_mode == "Superquadrics");
  const bool cbox_p = (display_mode == "Colored Boxes");

  const double emph = map_emphasis(zo_emphasis);

  GeomHandle glyph;
  if (box_p)
  {
    glyph = scinew GeomSimpleBox(Point(-1.0, -1.0, -1.0),Point(1.0, 1.0, 1.0));
  }
  else if (sphere_p)
  {
    glyph = scinew GeomSphere(Point(0.0, 0.0, 0.0), 1.0,resolution, resolution);
  }
  else if (squad_p)
  {
    glyph = 0;
  }
  else // cbox_p, default
  {
    glyph = scinew GeomCBox(Point(-1.0, -1.0, -1.0),Point(1.0, 1.0, 1.0));
  }

  GeomGroup *objs = scinew GeomGroup(); 
  GeomHandle data_switch = scinew GeomSwitch(scinew GeomMaterial(scinew GeomDL(objs), def_mat));

  FieldInformation cfi(cfld);

  int colorstyle = 0;
  if (cmap.get_rep())
  {
    colorstyle = 1;
  }
  else if (cfld && cfi.is_vector())
  {
    colorstyle = 2;
  }
  if (force_def_color || cbox_p)
  {
    colorstyle = 3;
  }

  // Use a default color?
  bool def_color = !(cmap.get_rep()) || force_def_color;
  bool vec_color = false;
  if (def_color && cfld && cfi.is_vector()
      && !force_def_color)
  {
    def_color = false;
    vec_color = true;
  }

  Mesh* mesh = vfld->mesh().get_rep();

  int field_order = vfld->basis_order();
  Mesh::size_type datasize= vfld->data_size();
  
  for(Mesh::index_type i; i < datasize; i++)
  {
    Tensor val;
    vfld->get_value(val, i);

    Point p;
    if (field_order == 0)
    {
      mesh->get_center(p, Mesh::VElem::index_type(i));
    }
    else
    {
      mesh->get_center(p, Mesh::VNode::index_type(i));      
    }
    if (colorstyle == 0)
    {
      if (squad_p)
      {
        add_super_quadric(objs, 0, p, val, scale, resolution, true, emph);
      }
      else
      {
        add_item(objs, glyph, p, val, scale, true);
      }
    }
    else if (colorstyle == 1)
    {
      double ctmpd;
      cfld->get_value(ctmpd, i);
      if (squad_p)
      {
        add_super_quadric(objs, cmap->lookup(ctmpd),
              p, val, scale, resolution, false, emph);
      }
      else
      {
        add_item(objs, scinew GeomMaterial(glyph, cmap->lookup(ctmpd)), p, val, scale, false);
      }
    }
    else if (colorstyle == 2)
    {
      Vector ctmpv;
      cfld->get_value(ctmpv, i);

      MaterialHandle vcol = scinew Material();
      sciVectorToColor(vcol->diffuse, ctmpv);
      if (squad_p)
      {
        add_super_quadric(objs, vcol, p, val, scale, resolution, false, emph);
      }
      else
      {
        add_item(objs, scinew GeomMaterial(glyph, vcol), p, val, scale, false);
      }
    }
    else
    {
      if (squad_p)
      {
        add_super_quadric(objs, 0, p, val, scale, resolution, false, emph);
      }
      else
      {
        add_item(objs, glyph, p, val, scale, false);
      }
    }
  }
  return data_switch;
}


GeomHandle 
RenderScalarFieldVirtual::render_data(FieldHandle sfld_handle,
						FieldHandle cfld_handle,
						ColorMapHandle cmap,
						MaterialHandle def_mat,
						bool force_def_color,
						const string &display_mode,
						double scale, 
						int resolution,
						bool transparent_p)
{
  // Get properties of field so we can query them
  Field *sfld = sfld_handle.get_rep();
  Field *cfld = cfld_handle.get_rep(); 

  int field_basis = sfld->basis_order();
  bool sized_p = (display_mode == "Scaled Spheres");
  bool points_p = (display_mode == "Points");

  GeomHandle data_switch = 0;
  GeomPoints *points = 0;
  GeomSpheres *spheres = 0;

  if (points_p)
  {
    if (transparent_p)
    {
      points = scinew GeomTranspPoints();
      data_switch = points;
    }
    else
    {
      points = scinew GeomPoints();
      data_switch = scinew GeomDL(points);
    }
  }
  else if (sized_p)
  {
    points = scinew GeomPoints();
    spheres = scinew GeomSpheres(scale, resolution, resolution);
    GeomGroup *tmp = scinew GeomGroup();
    tmp->add(spheres);
    tmp->add(points);
    data_switch = scinew GeomDL(tmp);
  }
  else
  {
    spheres = scinew GeomSpheres(scale, resolution, resolution);
    data_switch = scinew GeomDL(spheres);
  }
  
  Mesh* mesh = sfld->mesh().get_rep();

  // Use a default color?
  bool def_color = !(cmap.get_rep()) || force_def_color;
  bool vec_color = false;
  MaterialHandle vcol(0);
 
  FieldInformation cfi(cfld);
  if (def_color && cfld && cfi.is_vector() && !force_def_color)
  {
    def_color = false;
    vec_color = true;
    vcol = scinew Material();
    vcol->transparency = 1.0;
  }

  Mesh::size_type datasize= sfld->data_size(); 

  for (Mesh::index_type i=0; i< datasize; i++)
  {
    double tmp;
    sfld->get_value(tmp, i);

    Point p;
    if (field_basis == 0)
    {
      mesh->get_center(p, Mesh::VElem::index_type(i));
    }
    else
    {
      mesh->get_center(p, Mesh::VNode::index_type(i));      
    }

    if (points_p)
    {
      if (def_color)
      {
        points->add(p);
      }
      else if (vec_color)
      {
        Vector val;
        cfld->get_value(val, i);
        sciVectorToColor(vcol->diffuse, val);
        points->add(p, vcol);
      }
      else
      {
        double val;
        cfld->get_value(val, i);
        points->add(p, val);
      }
    }
    else
    {
      if (def_color)
      {
        if (sized_p)
        {
          const double dtmp = fabs((double)tmp * scale);
          if (!spheres->add_radius(p, dtmp))
          {
            points->add(p);
          }
        }
        else
        {
          spheres->add(p);
        }
      }
      else if (vec_color)
      {
        Vector val;
        cfld->get_value(val, i);
        sciVectorToColor(vcol->diffuse, val);
        if (sized_p)
        {
          const double dtmp = fabs((double)tmp * scale);
          if (!spheres->add_radius(p, dtmp, vcol))
          {
            points->add(p, vcol);
          }
        }
        else
        {
          spheres->add(p, vcol);
        }
      }
      else
      {
        double val;
        cfld->get_value(val, i);

        if (sized_p)
        {
          const double dtmp = fabs((double)tmp * scale);
          if (!spheres->add_radius(p, dtmp, val))
          {
            points->add(p, val);
          }
        }
        else
        {
          spheres->add(p, (float)val);
        }
      }
    }
  }
  return data_switch;
}



void 
RenderFieldVirtual::render(FieldHandle fh,  bool nodes, 
			      bool edges, bool faces,
			      ColorMapHandle color_handle,
			      MaterialHandle def_mat,
			      const string &ndt, const string &edt,
			      double ns, double es, double vs, bool normalize, 
			      int sphere_res, int cyl_res,
			      bool use_normals,
			      bool n_transp, bool e_transp, bool f_transp,
			      bool nfdc, bool efdc, bool ffdc, 
			      unsigned div, bool fut)

{
  Field *fld = fh.get_rep();
  
  Mesh *mesh = fld->mesh().get_rep();

  bool do_linear = (fld->basis_order() < 2 &&  mesh->basis_order() < 2 && div == 1);
    

  if (nodes)
  {
    node_switch_ = render_nodes(fld, ndt, color_handle, def_mat, nfdc, ns, sphere_res, n_transp);
  }
  
  if (edges)
  {
    if (do_linear) 
    {
      edge_switch_ = render_edges_linear(fld, edt, color_handle, def_mat, efdc, es, cyl_res, e_transp);
    } 
    else 
    {
      edge_switch_ = render_edges(fld, edt, color_handle, def_mat, efdc, es, cyl_res, e_transp, div);
    }
  }

  if (faces)
  {
    if (do_linear) 
    {
      face_switch_ = render_faces_linear(fld, color_handle, def_mat, ffdc, use_normals, f_transp, fut);
    } 
    else 
    {
      face_switch_ = render_faces(fld, color_handle, def_mat, ffdc, use_normals, f_transp, div, fut);
    }
  }
}


GeomHandle
RenderFieldVirtual::render_nodes(Field *sfld, 
				    const string &node_display_mode,
				    ColorMapHandle color_handle,
				    MaterialHandle def_mat,
				    bool force_def_color,
				    double node_scale,
				    int node_resolution,
				    bool use_transparency)
{
  Mesh *mesh = sfld->mesh().get_rep();

  GeomCappedCylinders *discs = 0;
  GeomPoints *points = 0;
  GeomSpheres *spheres = 0;
  GeomBoxes *boxes = 0;
  GeomLines *lines = 0;
  GeomHandle display_list(0);

  // 0 Points 1 Spheres 2 Axes 3 Disks
  int mode = 0;
  if (node_display_mode == "Points")       { mode = 0; }
  else if (node_display_mode == "Spheres") { mode = 1; }
  else if (node_display_mode == "Axes")    { mode = 2; }
  else if (node_display_mode == "Disks")   { mode = 3; }
  else if (node_display_mode == "Boxes")   { mode = 4; }

  if (mode == 0) // Points
  {
    if (use_transparency)
    {
      points = scinew GeomTranspPoints();
      display_list = points;
    }
    else
    {
      points = scinew GeomPoints();
      display_list = scinew GeomDL(points);
    }
  }
  else if (mode == 1) // Spheres
  {
    spheres = scinew GeomSpheres(node_scale, node_resolution, node_resolution);
    display_list = scinew GeomDL(spheres);
  }
  else if (mode == 2) // Axis
  {
    if (use_transparency)
    {
      lines = scinew GeomTranspLines();
      display_list = lines;
    }
    else
    {
      lines = scinew GeomLines();
      display_list = scinew GeomDL(lines);
    }
    lines->setLineWidth(3);
  }
  else if (mode == 3)
  {
    discs = scinew GeomCappedCylinders(node_resolution, node_scale);
    spheres = scinew GeomSpheres(node_scale * 0.75,
				 node_resolution, node_resolution);
    GeomGroup *grp = scinew GeomGroup();
    grp->add(discs);
    grp->add(spheres);
    display_list = scinew GeomDL(grp);
  }
  else if (mode == 4) // Boxes
  {
    boxes = scinew GeomBoxes(node_scale, node_resolution, node_resolution);
    display_list = scinew GeomDL(boxes);
  }

  // Use a default color?
  bool def_color = !(color_handle.get_rep()) || force_def_color;
  bool vec_color = false;
  MaterialHandle vcol(0);
  
  
  FieldInformation sfi(sfld);
  
  if (def_color && sfi.is_vector() && !force_def_color)
  {
    def_color = false;
    vec_color = true;
    vcol = scinew Material();
    vcol->transparency = 1.0;
  }

  // First pass: over the nodes
  mesh->synchronize(Mesh::NODES_E);
  
  Mesh::VNode::iterator niter;  mesh->begin(niter);  
  Mesh::VNode::iterator niter_end;  mesh->end(niter_end);  
  
  while (niter != niter_end) 
  {
    Point p;
    mesh->get_center(p, *niter);

    unsigned int n_idx = *niter;
    // val is double because the color index field must be scalar.
    Vector vec(0,0,0);
    double val = 0.0;

    if ((sfld->basis_order() > 0) || (sfld->basis_order() == 0 && mesh->dimensionality() == 0)) 
    { 
      sfld->get_value(val, *niter);
 
      if (vec_color)
      {
        sfld->get_value(vec, *niter);      
        sciVectorToColor(vcol->diffuse, vec);
      }
    } 
    else 
    {
      def_color = true;
    }

    switch (mode)
    {
    case 0: // Points
      if (def_color)
      {
        points->add(p, n_idx);
      }
      else if (vec_color)
      {
        points->add(p, vcol, n_idx);
      }
      else
      {
        points->add(p, val, n_idx);
      }
      break;

    case 1: // Spheres
      if (def_color)
      {
        spheres->add(p, n_idx);
      }
      else if (vec_color)
      {
        spheres->add(p, vcol, n_idx);
      }
      else
      {
        spheres->add(p, val, n_idx);
      }
      break;

    case 2: // Axes
      if (def_color)
      {
        add_axis(p, node_scale, lines);
      }
      else if (vec_color)
      {
        add_axis(p, node_scale, lines, vcol);
      }
      else
      {
        add_axis(p, node_scale, lines, val);
      }
      break;

    case 3: // Disks
    default:
      if (vec.safe_normalize() < 1.0e-5)
      {
        if (def_color)
        {
          spheres->add(p, n_idx);
        }
        else if (vec_color)
        {
          spheres->add(p, vcol, n_idx);
        }
        else
        {
          spheres->add(p, val, n_idx);
        }
      }
      else
      {
        vec *= node_scale/6.0;
        if (def_color)
        {
          discs->add(p-vec, p+vec);
        }
        else if (vec_color)
        {
          discs->add(p-vec, vcol, p+vec, vcol);
        }
        else
        {
          discs->add(p-vec, val, p+vec, val);
        }
      }
      break;
    case 4: // Boxes
      if (def_color)
      {
        boxes->add(p);
      }
      else if (vec_color)
      {
        boxes->add(p, vcol);
      }
      else
      {
        boxes->add(p, val);
      }
      break;
    }
    ++niter;
  }

  return display_list;
}



GeomHandle
RenderFieldVirtual::render_edges(Field *sfld,
				    const string &edge_display_mode,
				    ColorMapHandle color_handle,
				    MaterialHandle def_mat,
				    bool force_def_color,
				    double edge_scale,
				    int cylinder_resolution,
				    bool transparent_p, unsigned div) 
{
  Mesh* mesh = sfld->mesh().get_rep();

  const bool cyl = edge_display_mode == "Cylinders";

  GeomLines* lines = NULL;
  GeomCylinders* cylinders = NULL;
  GeomHandle display_list;
  if (cyl)
  {
    cylinders = scinew GeomCylinders(cylinder_resolution, edge_scale);
    display_list = scinew GeomDL(cylinders);
  }
  else
  {
    if (transparent_p)
    {
      lines = scinew GeomTranspLines;
      display_list = lines;
    }
    else
    {
      lines = scinew GeomLines;
      display_list = scinew GeomDL(lines);
    }
    lines->setLineWidth(1.0);
  }

  // Use a default color?
  bool def_color = !(color_handle.get_rep()) || force_def_color;
  bool vec_color = false;
  MaterialHandle vcol0(0), vcol1(0);

  FieldInformation sfi(sfld);
  
  bool is_tensor_val = sfi.is_tensor();
  bool is_vector_val = sfi.is_vector();
  
  if (def_color && sfi.is_vector() && !force_def_color)
  {
    def_color = false;
    vec_color = true;
    vcol0 = scinew Material();
    vcol0->transparency = 1.0;
    vcol1 = scinew Material();
    vcol1->transparency = 1.0;
  }

  bool constant = sfld->basis_order() == 0;
  bool curve_fld = mesh->dimensionality() == 1;
  
  if (constant && !curve_fld) def_color = true;

#if defined(_MSC_VER) || defined(__ECC)
  typedef hash_set<string> edge_ht_t;
#else
  typedef hash_set<string, str_hasher, equal_str> edge_ht_t;
#endif
  edge_ht_t rendered_edges; 

  // Second pass: over the edges
  mesh->synchronize(Mesh::EDGES_E | Mesh::FACES_E | Mesh::CELLS_E);

  Mesh::VElem::iterator eiter; mesh->begin(eiter);  
  Mesh::VElem::iterator eiter_end; mesh->end(eiter_end);  
  while (eiter != eiter_end) 
  {  
    Mesh::VEdge::array_type edges;
    mesh->get_edges(edges, *eiter);

    Mesh::VEdge::array_type::iterator edge_iter;
    edge_iter = edges.begin();
    int ecount = 0;
    while (edge_iter != edges.end()) 
    {

      Mesh::VNode::array_type nodes;
      Mesh::VEdge::index_type eidx = *edge_iter++;

      Point cntr;
      mesh->get_center(cntr, eidx);
      ostringstream pstr;  
      pstr << setiosflags(ios::scientific);
      pstr << setprecision(7); 
      pstr << cntr.x() << cntr.y() << cntr.z();
      
      edge_ht_t::const_iterator it = rendered_edges.find(pstr.str());

      if (it != rendered_edges.end()) 
      {
        ++ecount;
        continue;
      } 
      else 
      {
        rendered_edges.insert(pstr.str());
      }
      // following print is useful for debugging edge ordering
      //      cout << "elem: " << *eiter << " count " << ecount 
      //	   << " edge" << eidx << std::endl;
      vector<vector<double> > coords;
      mesh->pwl_approx_edge(coords, *eiter, ecount++, div);
      vector<vector<double> >::iterator coord_iter = coords.begin();
      do 
      {
        vector<double> &c0 = *coord_iter++;
        if (coord_iter == coords.end()) break;
        vector<double> &c1 = *coord_iter;
        Point p0, p1; 
        
        if (is_vector_val)
        {
          Vector val0, val1;
          
          // get the geometry at the approx.
          mesh->interpolate(p0, c0, *eiter);
          mesh->interpolate(p1, c1, *eiter);

          // get the field variables values at the approx (if they exist)
          if (sfld->basis_order() >= 0) 
          {
            sfld->interpolate(val0, c0, *eiter);
            sfld->interpolate(val1, c1, *eiter);
          }

          // add the geom_obj for this part....
          add_edge_geom(lines, cylinders, p0, p1, val0, val1, def_color, vec_color, cyl, vcol0, vcol1);        
        }
        else if (is_tensor_val)
        {
          Tensor val0, val1;
          
          // get the geometry at the approx.
          mesh->interpolate(p0, c0, *eiter);
          mesh->interpolate(p1, c1, *eiter);

          // get the field variables values at the approx (if they exist)
          if (sfld->basis_order() >= 0) {
            sfld->interpolate(val0, c0, *eiter);
            sfld->interpolate(val1, c1, *eiter);
          }

          // add the geom_obj for this part....
          add_edge_geom(lines, cylinders, p0, p1, val0, val1, def_color, vec_color, cyl, vcol0, vcol1);          
        }
        else
        {
          double val0, val1;
          
          // get the geometry at the approx.
          mesh->interpolate(p0, c0, *eiter);
          mesh->interpolate(p1, c1, *eiter);

          // get the field variables values at the approx (if they exist)
          if (sfld->basis_order() >= 0) {
            sfld->interpolate(val0, c0, *eiter);
            sfld->interpolate(val1, c1, *eiter);
          }

          // add the geom_obj for this part....
          add_edge_geom(lines, cylinders, p0, p1, val0, val1, def_color, vec_color, cyl, vcol0, vcol1);          
        }

	
      } while (coords.size() > 1 && coord_iter != coords.end()); 
    }
    
    ++eiter;
  }
  
  return display_list;
}



GeomHandle
RenderFieldVirtual::render_edges_linear(Field *sfld,
					   const string &edge_display_mode,
					   ColorMapHandle color_handle,
					   MaterialHandle def_mat,
					   bool force_def_color,
					   double edge_scale,
					   int cylinder_resolution,
					   bool transparent_p) 
{
  Mesh* mesh = sfld->mesh().get_rep();

  const bool cyl = edge_display_mode == "Cylinders";

  GeomLines* lines = NULL;
  GeomCylinders* cylinders = NULL;
  GeomHandle display_list;
  if (cyl)
  {
    cylinders = scinew GeomCylinders(cylinder_resolution, edge_scale);
    display_list = scinew GeomDL(cylinders);
  }
  else
  {
    if (transparent_p)
    {
      lines = scinew GeomTranspLines;
      display_list = lines;
    }
    else
    {
      lines = scinew GeomLines;
      display_list = scinew GeomDL(lines);
    }
    lines->setLineWidth(1.0);
  }

  // Use a default color?
  bool def_color = !(color_handle.get_rep()) || force_def_color;
  bool vec_color = false;
  MaterialHandle vcol0(0), vcol1(0);
  
  FieldInformation sfi(sfld);
  bool is_vector_val = sfi.is_vector();
  bool is_tensor_val = sfi.is_tensor();
  
  if (def_color && sfi.is_vector() && !force_def_color)
  {
    def_color = false;
    vec_color = true;
    vcol0 = scinew Material();
    vcol0->transparency = 1.0;
    vcol1 = scinew Material();
    vcol1->transparency = 1.0;
  }

  // Second pass: over the edges
  mesh->synchronize(Mesh::EDGES_E);
  
  Mesh::VEdge::iterator eiter; mesh->begin(eiter);  
  Mesh::VEdge::iterator eiter_end; mesh->end(eiter_end);  

  Mesh::VNode::array_type nodes;
  Vector v0,v1;
  Point p1, p2;
  double dval0, dval1;
  
  int basis_order = sfld->basis_order();
  int mesh_dimensionality = mesh->dimensionality();
  
  
  while (eiter != eiter_end) 
  {  
    mesh->get_nodes(nodes, *eiter);
      
    mesh->get_point(p1, nodes[0]);
    mesh->get_point(p2, nodes[1]);
    if (basis_order == 1)
    {
      if (def_color)
      {
        if (cyl)
        {
          cylinders->add(p1, p2);
        }
        else
        {
          lines->add(p1, p2);
        }
      }
      else if (vec_color)
      {
        
        sfld->get_value(v0, nodes[0]);
        sfld->get_value(v1, nodes[1]);
        sciVectorToColor(vcol0->diffuse, v0);
        sciVectorToColor(vcol1->diffuse, v1);
        if (cyl)
        {
          cylinders->add(p1, vcol0, p2, vcol1);
        }
        else
        {
          lines->add(p1, vcol0, p2, vcol1);
        }
      }
      else
      {
        sfld->get_value(dval0, nodes[0]);
        sfld->get_value(dval1, nodes[1]);
        if (cyl)
        {
          cylinders->add(p1, dval0, p2, dval1);
        }
        else
        {
          lines->add(p1, dval0, p2, dval1);
        }
      }
    }
    else if (mesh_dimensionality == 1)
    {
      if (def_color)
      {
        if (cyl)
        {
          cylinders->add(p1, p2);
        }
        else
        {
          lines->add(p1, p2);
        }
      }
      else if (vec_color)
      {
        sfld->get_value(v0, *eiter);
        sciVectorToColor(vcol0->diffuse, v0);
        if (cyl)
        {
          cylinders->add(p1, vcol0, p2, vcol0);
        }
        else
        {
          lines->add(p1, vcol0, p2, vcol0);
        }
      }
      else
      {
        double dval;
        sfld->get_value(dval, *eiter);
        if (cyl)
        {
          cylinders->add(p1, dval, p2, dval);
        }
        else
        {
          lines->add(p1, dval, p2, dval);
        }
      }
    }
    else 
    {
      if (cyl)
      {
        cylinders->add(p1, p2);
      }
      else
      {
        lines->add(p1, p2);
      }
    }
    ++eiter;
  }

  return display_list;
}


GeomHandle
RenderFieldVirtual::render_texture_face(Field *sfld,
                                           ColorMapHandle color_handle,
                                           MaterialHandle def_mat,
                                           bool force_def_color,
                                           bool use_normals,
                                           bool use_transparency)
{
  return 0;
}



GeomHandle 
RenderFieldVirtual::render_faces(Field *sfld,
				    ColorMapHandle color_handle,
				    MaterialHandle def_mat,
				    bool force_def_color,
				    bool use_normals,
				    bool use_transparency, unsigned div,
				    bool use_texture_for_face)
{
  unsigned int i;

  Mesh* mesh = sfld->mesh().get_rep();

  FieldInformation sfi(sfld);
  
  // if we have an ImageMesh, we can render it as a single polygon
  // with a textured face, providing the flag is true.


  if(sfi.is_image())
  {
    if( use_texture_for_face )
    {
      return render_texture_face(sfld, color_handle, def_mat,
				 force_def_color, use_normals,use_transparency);
    }
  }
  
  const bool with_normals = (use_normals && mesh->has_normals());

  GeomHandle face_switch;
  GeomFastTriangles* faces;
  GeomFastQuads* qfaces;
  if (use_transparency)
  {
    faces = scinew GeomTranspTriangles;
    qfaces = scinew GeomTranspQuads;
    GeomGroup *tmp = scinew GeomGroup;
    tmp->add(faces);
    tmp->add(qfaces);
    face_switch = tmp;
  }
  else
  {
    faces = scinew GeomFastTriangles;
    qfaces = scinew GeomFastQuads;
    GeomGroup *tmp = scinew GeomGroup;
    tmp->add(faces);
    tmp->add(qfaces);
    GeomDL *dl = scinew GeomDL(tmp);
    face_switch = dl;
  }

  // Use a default color?
  bool def_color = !(color_handle.get_rep()) || force_def_color;
  bool vec_color = false;
  vector<MaterialHandle> vcol(20, (Material *)NULL);
  vector<Vector> vvals(20);
  vector<double> dvals(20);
  
  
  if (def_color && sfi.is_vector() && !force_def_color)
  {
    def_color = false;
    vec_color = true;
    for (i = 0; i < 20; i++)
    {
      vcol[i] = scinew Material();
      vcol[i]->transparency = 1.0;
    }
  }

#if defined(_MSC_VER) || defined(__ECC)
  typedef hash_set<string> face_ht_t;
#else
  typedef hash_set<string, str_hasher, equal_str> face_ht_t;
#endif
  face_ht_t rendered_faces; 
  
  mesh->synchronize(Mesh::FACES_E | Mesh::EDGES_E | Mesh::CELLS_E);
  Mesh::VElem::iterator eiter; mesh->begin(eiter);  
  Mesh::VElem::iterator eiter_end; mesh->end(eiter_end);  
  while (eiter != eiter_end) 
  {  
    Mesh::VFace::array_type face_indecies;
    mesh->get_faces(face_indecies, *eiter);

    Mesh::VFace::array_type::iterator face_iter;
    face_iter = face_indecies.begin();
    int fcount = 0;
    
    bool is_tri = sfi.is_tri_element();
    bool is_vector_val = sfi.is_vector();
    bool is_tensor_val = sfi.is_tensor();

    Mesh::VNode::array_type nodes;
    vector<vector<vector<double> > > coords;
    
    while (face_iter != face_indecies.end()) 
    {
      Mesh::VFace::index_type fidx = *face_iter++;

      Point cntr;
      mesh->get_center(cntr, fidx);
      ostringstream pstr;
      pstr << setiosflags(ios::scientific);
      pstr << setprecision(7); 
      pstr << cntr.x() << cntr.y() << cntr.z();
      
      face_ht_t::const_iterator it = rendered_faces.find(pstr.str());

      if (it != rendered_faces.end()) 
      {
        ++fcount;
        continue;
      } 
      else 
      {
        rendered_faces.insert(pstr.str());
      }

      //coords organized as scanlines of quad/tri strips.
      mesh->pwl_approx_face(coords, *eiter, fcount, div);

      vector<vector<vector<double> > >::iterator coord_iter = coords.begin();
      while (coord_iter != coords.end()) 
      {
        vector<vector<double> > &sl = *coord_iter++;
        vector<vector<double> >::iterator sliter = sl.begin();

        for (unsigned int i = 0; i < sl.size() - 2; i++) 
        {
          if (is_tri) 
          {  // TRI STRIPS
            vector<vector<double> >::iterator it0,it1;
            
            vector<double> &c0 = !i%2 ? sl[i] : sl[i+1];
            vector<double> &c1 = !i%2 ? sl[i+1] : sl[i];
            vector<double> &c2 = sl[i+2];

            vector<Point> pnts(3);
            vector<Vector> norms(3);
	  
            mesh->interpolate(pnts[0], c0, *eiter);
            mesh->interpolate(pnts[1], c1, *eiter);
            mesh->interpolate(pnts[2], c2, *eiter);

            if (with_normals) 
            {	      
              mesh->get_normal(norms[0], c0, *eiter, fcount);
              mesh->get_normal(norms[1], c1, *eiter, fcount);
              mesh->get_normal(norms[2], c2, *eiter, fcount);
            }
          
            // get the field variables values at the approx (if they exist)
            if (is_vector_val)
            {
              vector<Vector> vals(3);

              if (sfld->basis_order() >= 0) 
              {
                sfld->interpolate(vals[0], c0, *eiter);
                sfld->interpolate(vals[1], c1, *eiter);
                sfld->interpolate(vals[2], c2, *eiter);
              } 
              else 
              {
                def_color = true;
              }
              // add the geom_obj for this part....
      
              add_face_geom(faces, qfaces, pnts, norms, vals,
                def_color, vec_color, with_normals, 
                vcol, vvals, dvals);
            }
            else if (is_tensor_val)
            {
               vector<Tensor> vals(3);

              if (sfld->basis_order() >= 0) 
              {
                sfld->interpolate(vals[0], c0, *eiter);
                sfld->interpolate(vals[1], c1, *eiter);
                sfld->interpolate(vals[2], c2, *eiter);
              } 
              else 
              {
                def_color = true;
              }
              // add the geom_obj for this part....
      
              add_face_geom(faces, qfaces, pnts, norms, vals,
                def_color, vec_color, with_normals, 
                vcol, vvals, dvals);           
            }
            else
            {
               vector<double> vals(3);

              if (sfld->basis_order() >= 0) 
              {
                sfld->interpolate(vals[0], c0, *eiter);
                sfld->interpolate(vals[1], c1, *eiter);
                sfld->interpolate(vals[2], c2, *eiter);
              } 
              else 
              {
                def_color = true;
              }
              // add the geom_obj for this part....
      
              add_face_geom(faces, qfaces, pnts, norms, vals,
                def_color, vec_color, with_normals, 
                vcol, vvals, dvals);           
            }
          } 
          else 
          { // QUADS

            vector<double> &c0 = *sliter++;
            if (sliter == sl.end()) break;
            vector<double> &c1 = *sliter++;
            if (sliter == sl.end()) break;
            vector<double> &c2 = *sliter;
            vector<double> &c3 = *(sliter + 1);

            vector<Point> pnts(4);
            vector<Vector> norms(4);

            // get the geometry at the approx.
            mesh->interpolate(pnts[0], c2, *eiter);
            mesh->interpolate(pnts[1], c3, *eiter);
            mesh->interpolate(pnts[2], c1, *eiter);
            mesh->interpolate(pnts[3], c0, *eiter);

            if (with_normals) 
            {	      
              mesh->get_normal(norms[0], c2, *eiter, fcount);
              mesh->get_normal(norms[1], c3, *eiter, fcount);
              mesh->get_normal(norms[2], c1, *eiter, fcount);
              mesh->get_normal(norms[3], c0, *eiter, fcount);
            }

            if (is_vector_val)
            {
              vector<Vector> vals(4);
            
              // get the field variables values at the approx (if they exist)
              if (sfld->basis_order() >= 0) {
                sfld->interpolate(vals[0], c2, *eiter);
                sfld->interpolate(vals[1], c3, *eiter);
                sfld->interpolate(vals[2], c1, *eiter);
                sfld->interpolate(vals[3], c0, *eiter);
              } 
              else 
              {
                def_color = true;
              }
              // add the geom_obj for this part....
            
              add_face_geom(faces, qfaces, pnts, norms, vals,
                def_color, vec_color, with_normals, 
                vcol, vvals, dvals);
            }
            else if (is_tensor_val)
            {
              vector<Tensor> vals(4);
            
              // get the field variables values at the approx (if they exist)
              if (sfld->basis_order() >= 0) {
                sfld->interpolate(vals[0], c2, *eiter);
                sfld->interpolate(vals[1], c3, *eiter);
                sfld->interpolate(vals[2], c1, *eiter);
                sfld->interpolate(vals[3], c0, *eiter);
              } 
              else 
              {
                def_color = true;
              }
              // add the geom_obj for this part....
            
              add_face_geom(faces, qfaces, pnts, norms, vals,
                def_color, vec_color, with_normals, 
                vcol, vvals, dvals);            
            }
            else
            {
              vector<double> vals(4);
            
              // get the field variables values at the approx (if they exist)
              if (sfld->basis_order() >= 0) {
                sfld->interpolate(vals[0], c2, *eiter);
                sfld->interpolate(vals[1], c3, *eiter);
                sfld->interpolate(vals[2], c1, *eiter);
                sfld->interpolate(vals[3], c0, *eiter);
              } 
              else 
              {
                def_color = true;
              }
              // add the geom_obj for this part....
            
              add_face_geom(faces, qfaces, pnts, norms, vals,
                def_color, vec_color, with_normals, 
                vcol, vvals, dvals);               
            }
          }
        }
      }
      ++fcount;
    }
    ++eiter;
  }
  return face_switch;
}


GeomHandle 
RenderFieldVirtual::render_faces_linear(Field *sfld,
					   ColorMapHandle color_handle,
					   MaterialHandle def_mat,
					   bool force_def_color,
					   bool use_normals,
					   bool use_transparency,
					   bool use_texture_for_face)
{
  unsigned int i;

  Mesh* mesh = sfld->mesh().get_rep();

  FieldInformation sfi(sfld);
  
  if((sfi.is_image()))
  {
    if( use_texture_for_face )
    {
      return render_texture_face(sfld, color_handle, def_mat,force_def_color, use_normals,use_transparency);
    }
  }
  
  const bool with_normals = (use_normals && mesh->has_normals());

  GeomHandle face_switch;
  GeomFastTriangles* faces;
  GeomFastQuads* qfaces;
  GeomFastTrianglesTwoSided* tfaces;
  GeomFastQuadsTwoSided* tqfaces;

  bool def_color = !(color_handle.get_rep()) || force_def_color;
  
  if ((sfld->basis_order() == 0) && (mesh->dimensionality() == 3))
  {
    mesh->synchronize(Mesh::FACES_E);
    Mesh::VFace::iterator face_iter; mesh->begin(face_iter); 
    Mesh::VFace::iterator face_iter_end; mesh->end(face_iter_end); 
    Mesh::VElem::array_type cells;
    if (face_iter != face_iter_end)
    {
      mesh->get_elems(cells,*face_iter);  
      if (cells.size() == 0) def_color = true;
    }
    else
    {
      def_color = true;
    }
  }
  
  if (use_transparency)
  {
    faces = scinew GeomTranspTriangles;
    qfaces = scinew GeomTranspQuads;
    GeomGroup *tmp = scinew GeomGroup;
    tmp->add(faces);
    tmp->add(qfaces);
    face_switch = tmp;
    if (sfld->basis_order() == 0 && mesh->dimensionality() == 3)
    {
      def_color = true;
    }
  }
  else if ((sfld->basis_order() == 0) && (mesh->dimensionality() == 3) && !def_color)
  {
    tfaces = scinew GeomFastTrianglesTwoSided;
    tqfaces = scinew GeomFastQuadsTwoSided;
    GeomGroup *tmp = scinew GeomGroup;
    tmp->add(tfaces);
    tmp->add(tqfaces);
    GeomDL *dl = scinew GeomDL(tmp);
    face_switch = dl;
  }
  else
  {
    faces = scinew GeomFastTriangles;
    qfaces = scinew GeomFastQuads;
    GeomGroup *tmp = scinew GeomGroup;
    tmp->add(faces);
    tmp->add(qfaces);
    GeomDL *dl = scinew GeomDL(tmp);
    face_switch = dl;
  }

  bool vec_color = false;
  vector<MaterialHandle> vcol(20, (Material *)NULL);
  vector<Vector> vvals(20);
  
  vector<double> dvals(20);
  if (def_color && sfi.is_vector() && !force_def_color)
  {
    def_color = false;
    vec_color = true;
    for (i = 0; i < 20; i++)
    {
      vcol[i] = scinew Material();
      vcol[i]->transparency = 1.0;
    }
  }
  // Third pass: over the faces
  if (with_normals) mesh->synchronize(Mesh::NORMALS_E);
  mesh->synchronize(Mesh::FACES_E);
  
  Mesh::VFace::iterator fiter; mesh->begin(fiter);  
  Mesh::VFace::iterator fiter_end; mesh->end(fiter_end);  
  Mesh::VNode::array_type nodes;
  Mesh::VElem::array_type cells;

  while (fiter != fiter_end) 
  {
    mesh->get_nodes(nodes, *fiter); 
 
    vector<Point> points(nodes.size());
    vector<Vector> normals(nodes.size());
    for (i = 0; i < nodes.size(); i++)
    {
      mesh->get_point(points[i], nodes[i]);
    }

    if (with_normals) 
    {
      for (i = 0; i < nodes.size(); i++) 
      {
        mesh->get_normal(normals[i], nodes[i]);
      }
    }

    if (sfld->basis_order() == 1 && !def_color)
    {
      if (vec_color)
      {
        for (i = 0; i < nodes.size(); i++)
        {
          Vector v;
          sfld->get_value(v,nodes[i]);
          sciVectorToColor(vcol[i]->diffuse, v);
        }
        if (nodes.size() == 4)
        {
          if (with_normals)
          {
            qfaces->add(points[0], normals[0], vcol[0],
                        points[1], normals[1], vcol[1],
                        points[2], normals[2], vcol[2],
                        points[3], normals[3], vcol[3]);
          }
          else
          {
            qfaces->add(points[0], vcol[0],
                        points[1], vcol[1],
                        points[2], vcol[2],
                        points[3], vcol[3]);
          }
        }
        else
        {
          for (i=2; i<nodes.size(); i++)
          {
            if (with_normals)
            {
              faces->add(points[0], normals[0], vcol[0],
                         points[i-1], normals[i-1], vcol[i-1],
                         points[i], normals[i], vcol[i]);
            }
            else
            {
              faces->add(points[0], vcol[0],
                         points[i-1], vcol[i-1],
                         points[i], vcol[i]);
            }
          }
        }
      }
      else
      {
        for (i = 0; i < nodes.size(); i++)
        {
          sfld->get_value(dvals[i],nodes[i]);
        }
        if (nodes.size() == 4)
        {
          if (with_normals)
          {
            qfaces->add(points[0], normals[0], dvals[0],
                        points[1], normals[1], dvals[1],
                        points[2], normals[2], dvals[2],
                        points[3], normals[3], dvals[3]);
          }
          else
          {
            qfaces->add(points[0], dvals[0],
                        points[1], dvals[1],
                        points[2], dvals[2],
                        points[3], dvals[3]);
          }
        }
        else
        {
          for (i=2; i<nodes.size(); i++)
          {
            if (with_normals)
            {
              faces->add(points[0], normals[0], dvals[0],
                         points[i-1], normals[i-1], dvals[i-1],
                         points[i], normals[i], dvals[i]);
            }
            else
            {
              faces->add(points[0], dvals[0],
                         points[i-1], dvals[i-1],
                         points[i], dvals[i]);
            }
          }
        }
      }
    }
    else if (sfld->basis_order() == 0 && mesh->dimensionality() == 2 && !def_color)
    {
      if (vec_color)
      {
        Vector vval;
        sfld->get_value(vval,*fiter);
        sciVectorToColor(vcol[0]->diffuse, vval);
        if (nodes.size() == 4)
        {
          if (with_normals)
          {
            qfaces->add(points[0], normals[0], vcol[0],
                        points[1], normals[1], vcol[0],
                        points[2], normals[2], vcol[0],
                        points[3], normals[3], vcol[0]);
          }
          else
          {
            qfaces->add(points[0], vcol[0],
                        points[1], vcol[0],
                        points[2], vcol[0],
                        points[3], vcol[0]);
          }
        }
        else
        {
          for (i=2; i<nodes.size(); i++)
          {
            if (with_normals)
            {
              faces->add(points[0], normals[0], vcol[0],
                         points[i-1], normals[i-1], vcol[0],
                         points[i], normals[i], vcol[0]);
            }
            else
            {
              faces->add(points[0], vcol[0],
                         points[i-1], vcol[0],
                         points[i], vcol[0]);
            }
          }
        }
      }
      else
      {
        double dval;
        sfld->get_value(dval,*fiter);
        
        if (nodes.size() == 4)
        {
          if (with_normals)
          {
            qfaces->add(points[0], normals[0], dval,
                        points[1], normals[1], dval,
                        points[2], normals[2], dval,
                        points[3], normals[3], dval);
          }
          else
          {
            qfaces->add(points[0], dval,
                        points[1], dval,
                        points[2], dval,
                        points[3], dval);
          }
        }
        else
        {
          for (i=2; i<nodes.size(); i++)
          {
            if (with_normals)
            {
              faces->add(points[0], normals[0], dval,
                         points[i-1], normals[i-1], dval,
                         points[i], normals[i], dval);
            }
            else
            {
              faces->add(points[0], dval,
                         points[i-1], dval,
                         points[i], dval);
            }
          }
        }
      }
    }
    else if (sfld->basis_order() == 0 && mesh->dimensionality() == 3 && !def_color)
    {
      mesh->get_elems(cells,*fiter);

      if (vec_color)
      {
        Vector vval, vval2;
        sfld->get_value(vval, cells[0]);
        if (cells.size() > 1) sfld->get_value(vval2, cells[1]); else vval2 = vval;
        sciVectorToColor(vcol[0]->diffuse, vval);
        sciVectorToColor(vcol[1]->diffuse, vval2);

        if (nodes.size() == 4)
        {
          if (with_normals)
          {
            tqfaces->add(points[0], normals[0], vcol[0], vcol[1],
                        points[1], normals[1], vcol[0], vcol[1],
                        points[2], normals[2], vcol[0], vcol[1],
                        points[3], normals[3], vcol[0], vcol[1]);
          }
          else
          {
            tqfaces->add(points[0], vcol[0], vcol[1],
                        points[1], vcol[0], vcol[1],
                        points[2], vcol[0], vcol[1],
                        points[3], vcol[0], vcol[1]);
          }
        }
        else
        {
          for (i=2; i<nodes.size(); i++)
          {
            if (with_normals)
            {
              tfaces->add(points[0], normals[0], vcol[0], vcol[1],
                         points[i-1], normals[i-1], vcol[0], vcol[1],
                         points[i], normals[i], vcol[0], vcol[1]);
            }
            else
            {
              tfaces->add(points[0], vcol[0], vcol[1],
                         points[i-1], vcol[0], vcol[1],
                         points[i], vcol[0], vcol[1]);
            }
          }
        }
      }
      else
      {
        double dval, dval2;
        sfld->get_value(dval, cells[0]);
        if (cells.size() > 1) sfld->get_value(dval2, cells[1]); else dval2 = dval;
        
        if (nodes.size() == 4)
        {
          if (with_normals)
          {
            tqfaces->add(points[0], normals[0], dval, dval2,
                        points[1], normals[1], dval, dval2,
                        points[2], normals[2], dval, dval2,
                        points[3], normals[3], dval, dval2);
          }
          else
          {
            tqfaces->add(points[0], dval, dval2,
                        points[1], dval, dval2,
                        points[2], dval, dval2,
                        points[3], dval, dval2);
          }
        }
        else
        {
          for (i=2; i<nodes.size(); i++)
          {
            if (with_normals)
            {
              tfaces->add(points[0], normals[0], dval, dval2,
                         points[i-1], normals[i-1], dval, dval2,
                         points[i], normals[i], dval, dval2);
            }
            else
            {
              tfaces->add(points[0], dval, dval2,
                         points[i-1], dval, dval2,
                         points[i], dval, dval2);
            }
          }
        }
      }
    }
    else
    {
      if (nodes.size() == 4)
      {
        if (with_normals)
        {
          qfaces->add(points[0], normals[0],
                      points[1], normals[1],
                      points[2], normals[2],
                      points[3], normals[3]);
        }
        else
        {
          qfaces->add(points[0], points[1], points[2], points[3]);
        }
      }
      else
      {
        for (i=2; i<nodes.size(); i++)
        {
          if (with_normals)
          {
            faces->add(points[0], normals[0],
                       points[i-1], normals[i-1],
                       points[i], normals[i]);
          }
          else
          {
            faces->add(points[0], points[i-1], points[i]);
          }
        }
      }
    }
    ++fiter;     
  }

  return face_switch;
}


GeomHandle 
RenderFieldVirtual::render_text(FieldHandle field_handle,
				   bool use_color_map,
				   bool use_default_material,
				   bool backface_cull_p,
				   int fontsize,
				   int precision,
				   bool render_locations,
				   bool render_data,
				   bool render_nodes,
				   bool render_edges,
				   bool render_faces,
				   bool render_cells)
{
  GeomGroup *texts = scinew GeomGroup;
  GeomHandle text_switch = scinew GeomSwitch(texts);

  if (render_data)
  {
    texts->add(render_text_data(field_handle, use_color_map,
				use_default_material,
				backface_cull_p,
				fontsize, precision));
  }
  if (render_nodes)
  {
    texts->add(render_text_nodes(field_handle, use_color_map,
				 use_default_material,
				 backface_cull_p,
				 fontsize, precision, render_locations));
  }
  if (render_edges)
  {
    texts->add(render_text_edges(field_handle, use_color_map,
				 use_default_material,
				 fontsize, precision, render_locations));
  }
  if (render_faces)
  {
    texts->add(render_text_faces(field_handle, use_color_map,
				 use_default_material,
				 fontsize, precision, render_locations));
  }
  if (render_cells)
  {
    texts->add(render_text_cells(field_handle, use_color_map,
				 use_default_material,
				 fontsize, precision, render_locations));
  }
  return text_switch;
}


GeomHandle RenderFieldVirtual::render_text_data(FieldHandle field_handle,
					bool use_color_map,
					bool use_default_material,
					bool backface_cull_p,
					int fontsize,
					int precision)
{
  if (backface_cull_p && field_handle->basis_order() == 1)
  {
    return render_text_data_nodes(field_handle, use_color_map,
				  use_default_material,
				  backface_cull_p, fontsize,
				  precision);
  }

  Field* fld = field_handle.get_rep();
  Mesh* mesh = fld->mesh().get_rep();

  GeomTexts *texts = scinew GeomTexts();
  GeomHandle text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  texts->set_font_index(fontsize);

  std::ostringstream buffer;
  buffer.precision(precision);

  FieldInformation fi(fld);
  
  bool vec_color = false;
  if (!use_color_map)
  {
    if (!use_default_material && fi.is_vector())
    {
      vec_color = true;
    }
    else
    {
      use_default_material = true;
    }
  }
 
  int field_order = fld->basis_order();
  Point p;
 

  unsigned int datasize = fld->data_size();
  
  if (fi.is_scalar())
  {
    for (Mesh::index_type i = 0; i < datasize; i++)
    {
      if (field_order == 0) mesh->get_center(p,Mesh::VElem::index_type(i));
      else mesh->get_center(p,Mesh::VNode::index_type(i));
    
      double val;
      fld->get_value(val, i);

      buffer.str("");
      value_to_string(buffer, val);

      if (use_default_material)
      {
        texts->add(buffer.str(), p);
      }
      else if (vec_color)
      {
        Vector vval(0, 0, 0);
        to_vector(val, vval);
        texts->add(buffer.str(), p, Color(vval.x(), vval.y(), vval.z()));
      }
      else
      {
        double dval = 0.0;
        to_double(val, dval);
        texts->add(buffer.str(), p, dval);
      }
    }
  }
  else if (fi.is_vector())
  {
    for (Mesh::index_type i = 0; i < datasize; i++)
    {
      if (field_order == 0) mesh->get_center(p,Mesh::VElem::index_type(i));
      else mesh->get_center(p,Mesh::VNode::index_type(i));
    
      Vector val;
      fld->get_value(val, i);

      buffer.str("");
      value_to_string(buffer, val);

      if (use_default_material)
      {
        texts->add(buffer.str(), p);
      }
      else if (vec_color)
      {
        Vector vval(0, 0, 0);
        to_vector(val, vval);
        texts->add(buffer.str(), p, Color(vval.x(), vval.y(), vval.z()));
      }
      else
      {
        double dval = 0.0;
        to_double(val, dval);
        texts->add(buffer.str(), p, dval);
      }
    }  
  
  }
  else if (fi.is_tensor())
  {
    for (Mesh::index_type i = 0; i < datasize; i++)
    {
      if (field_order == 0) mesh->get_center(p,Mesh::VElem::index_type(i));
      else mesh->get_center(p,Mesh::VNode::index_type(i));
    
      Tensor val;
      fld->get_value(val, i);

      buffer.str("");
      value_to_string(buffer, val);

      if (use_default_material)
      {
        texts->add(buffer.str(), p);
      }
      else if (vec_color)
      {
        Vector vval(0, 0, 0);
        to_vector(val, vval);
        texts->add(buffer.str(), p, Color(vval.x(), vval.y(), vval.z()));
      }
      else
      {
        double dval = 0.0;
        to_double(val, dval);
        texts->add(buffer.str(), p, dval);
      }
    }  
  }
  
  return text_switch;
}


GeomHandle 
RenderFieldVirtual::render_text_data_nodes(FieldHandle field_handle,
					      bool use_color_map,
					      bool use_default_material,
					      bool backface_cull_p,
					      int fontsize,
					      int precision)
{
  Field* fld = field_handle.get_rep();
  Mesh* mesh = fld->mesh().get_rep();
  
  FieldInformation fi(fld);

  GeomTexts *texts = 0;
  GeomTextsCulled *ctexts = 0;
  GeomHandle text_switch = 0;
  const bool culling_p = backface_cull_p && mesh->has_normals();
  if (culling_p)
  {
    mesh->synchronize(Mesh::NORMALS_E);
    ctexts = scinew GeomTextsCulled();
    text_switch = scinew GeomSwitch(ctexts);
    ctexts->set_font_index(fontsize);
  }
  else
  {
    texts = scinew GeomTexts();
    text_switch = scinew GeomSwitch(scinew GeomDL(texts));
    texts->set_font_index(fontsize);
  }

  std::ostringstream buffer;
  buffer.precision(precision);

  bool vec_color = false;
  if (fld->basis_order() != 1)
  {
    use_default_material = true;
  }
  else if (!use_color_map)
  {
    if (!use_default_material && fi.is_vector())
    {
      vec_color = true;
    }
    else
    {
      use_default_material = true;
    }
  }

  bool is_scalar = fi.is_scalar();
  bool is_vector = fi.is_vector();
  bool is_tensor = fi.is_tensor();

  Mesh::VNode::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  Vector n;
  while (iter != end) 
  {
    mesh->get_center(p, *iter);

    if (is_scalar)
    {
      double val;
      fld->get_value(val, *iter);
      buffer.str("");
      value_to_string(buffer, val);      
    }
    else if (is_vector)
    {
      Vector val;
      fld->get_value(val, *iter);
      buffer.str("");
      value_to_string(buffer, val);          
    }
    else if (is_tensor)
    {
      Tensor val;
      fld->get_value(val, *iter);
      buffer.str("");
      value_to_string(buffer, val);      
    }

    if (use_default_material)
    {
      if (culling_p)
      {
        mesh->get_normal(n, *iter);
        ctexts->add(buffer.str(), p, n);
      }
      else
      {
        texts->add(buffer.str(), p);
      }
    }
    else if (vec_color)
    {
      Vector vval;
      fld->get_value(vval, *iter);
      if (culling_p)
      {
        mesh->get_normal(n, *iter);
        ctexts->add(buffer.str(), p, n, Color(vval.x(), vval.y(), vval.z()));
      }
      else
      {
        texts->add(buffer.str(), p, Color(vval.x(), vval.y(), vval.z()));
      }
    }
    else
    {
      double dval;
      fld->get_value(dval, *iter);
      if (culling_p)
      {
        mesh->get_normal(n, *iter);
        ctexts->add(buffer.str(), p, n, dval);
      }
      else
      {
        texts->add(buffer.str(), p, dval);
      }
    }
    ++iter;
  }

  return text_switch;
}


GeomHandle 
RenderFieldVirtual::render_text_nodes(FieldHandle field_handle,
					 bool use_color_map,
					 bool use_default_material,
					 bool backface_cull_p,
					 int fontsize,
					 int precision,
					 bool render_locations)
{
  Field* fld = field_handle.get_rep();
  Mesh* mesh = fld->mesh().get_rep();
  
  GeomTexts *texts = 0;
  GeomTextsCulled *ctexts = 0;
  GeomHandle text_switch = 0;

  const bool culling_p = backface_cull_p && mesh->has_normals();
  if (culling_p)
  {
    mesh->synchronize(Mesh::NORMALS_E);
    ctexts = scinew GeomTextsCulled();
    text_switch = scinew GeomSwitch(ctexts);
    ctexts->set_font_index(fontsize);
  }
  else
  {
    texts = scinew GeomTexts();
    text_switch = scinew GeomSwitch(scinew GeomDL(texts));
    texts->set_font_index(fontsize);
  }

  FieldInformation fi(fld);
  bool vec_color = false;
  if (!(fld->basis_order() == 1 || mesh->dimensionality() == 0))
  {
    use_default_material = true;
  }
  else if (!use_color_map)
  {
    if (!use_default_material && fi.is_vector())
    {
      vec_color = true;
    }
    else
    {
      use_default_material = true;
    }
  }

  ostringstream buffer;
  buffer.precision(precision);

  Mesh::VNode::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  Vector n;
  while (iter != end)
  {
    mesh->get_center(p, *iter);

    buffer.str("");
    if (render_locations)
    {
      buffer << p;
    }
    else
    {
      (*iter).str_render(buffer);
    }

    if (use_default_material)
    {
      if (culling_p)
      {
        mesh->get_normal(n, *iter);
        ctexts->add(buffer.str(), p, n);
      }
      else
      {
        texts->add(buffer.str(), p);
      }
    }
    else if (vec_color)
    {
      Vector vval;
      fld->get_value(vval,*iter);
      if (culling_p)
      {
        mesh->get_normal(n, *iter);
        ctexts->add(buffer.str(), p, n, Color(vval.x(), vval.y(), vval.z()));
      }
      else
      {
        texts->add(buffer.str(), p, Color(vval.x(), vval.y(), vval.z()));
      }
    }
    else
    {
      double dval;
      fld->get_value(dval, *iter);
      if (culling_p)
      {
        mesh->get_normal(n, *iter);
        ctexts->add(buffer.str(), p, n, dval);
      }
      else
      {
        texts->add(buffer.str(), p, dval);
      }
    }

    ++iter;
  }
  return text_switch;
}

GeomHandle 
RenderFieldVirtual::render_text_edges(FieldHandle field_handle,
					 bool use_color_map,
					 bool use_default_material,
					 int fontsize,
					 int precision,
					 bool render_locations)
{
  Field *fld = field_handle.get_rep();
  Mesh* mesh = fld->mesh().get_rep();
  mesh->synchronize(Mesh::EDGES_E);

  GeomTexts *texts = scinew GeomTexts;
  GeomHandle text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  texts->set_font_index(fontsize);

  ostringstream buffer;
  buffer.precision(precision);

  FieldInformation fi(fld);

  bool vec_color = false;
  if (! (fld->basis_order() == 0 && mesh->dimensionality() == 1))
  {
    use_default_material = true;
  }
  else if (!use_color_map)
  {
    if (!use_default_material && fi.is_vector())
    {
      vec_color = true;
    }
    else
    {
      use_default_material = true;
    }
  }

  Mesh::VEdge::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  while (iter != end)
  {
    mesh->get_center(p, *iter);

    buffer.str("");
    if (render_locations)
    {
      buffer << p;
    }
    else
    {
      buffer << (int)(*iter);
    }

    if (use_default_material)
    {
      texts->add(buffer.str(), p);
    }
    else if (vec_color)
    {
      Vector vval;
      fld->get_value(vval, *iter);
      texts->add(buffer.str(), p, Color(vval.x(), vval.y(), vval.z()));
    }
    else
    {
      double dval;
      fld->get_value(dval, *iter);
      texts->add(buffer.str(), p, dval);
    } 
    ++iter;
  }
  return text_switch;
}


GeomHandle 
RenderFieldVirtual::render_text_faces(FieldHandle field_handle,
					 bool use_color_map,
					 bool use_default_material,
					 int fontsize,
					 int precision,
					 bool render_locations)
{
  Field *fld = field_handle.get_rep();
  Mesh* mesh = fld->mesh().get_rep();
  mesh->synchronize(Mesh::FACES_E);

  GeomTexts *texts = scinew GeomTexts;
  GeomHandle text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  texts->set_font_index(fontsize);

  ostringstream buffer;
  buffer.precision(precision);

  FieldInformation fi(fld);

  bool vec_color = false;
  if (! (fld->basis_order() == 0 && mesh->dimensionality() == 2))
  {
    use_default_material = true;
  }
  else if (!use_color_map)
  {
    if (!use_default_material && fi.is_vector())
    {
      vec_color = true;
    }
    else
    {
      use_default_material = true;
    }
  }

  Mesh::VFace::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  while (iter != end)
  {
    mesh->get_center(p, *iter);

    buffer.str("");
    if (render_locations)
    {
      buffer << p;
    }
    else
    {
      (*iter).str_render(buffer);
    }

    if (use_default_material)
    {
      texts->add(buffer.str(), p);
    }
    else if (vec_color)
    {
      Vector vval;
      fld->get_value(vval, *iter);
      texts->add(buffer.str(), p, Color(vval.x(), vval.y(), vval.z()));
    }
    else
    {
      double dval;
      fld->get_value(dval, *iter);
      texts->add(buffer.str(), p, dval);
    }
    ++iter;
  }
  return text_switch;
}


GeomHandle 
RenderFieldVirtual::render_text_cells(FieldHandle field_handle,
					 bool use_color_map,
					 bool use_default_material,
					 int fontsize,
					 int precision,
					 bool render_locations)
{
  Field *fld = field_handle.get_rep();

  Mesh* mesh = fld->mesh().get_rep();
  mesh->synchronize(Mesh::CELLS_E);

  GeomTexts *texts = scinew GeomTexts;
  GeomHandle text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  texts->set_font_index(fontsize);

  ostringstream buffer;
  buffer.precision(precision);
  FieldInformation fi(fld);

  bool vec_color = false;
  if (fld->basis_order() != 0)
  {
    use_default_material = true;
  }
  else if (!use_color_map)
  {
    if (!use_default_material && fi.is_vector())
    {
      vec_color = true;
    }
    else
    {
      use_default_material = true;
    }
  }

  Mesh::VCell::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  while (iter != end)
  {
    mesh->get_center(p, *iter);

    buffer.str("");
    if (render_locations)
    {
      buffer << p;
    }
    else
    {
      (*iter).str_render(buffer);
    }

    if (use_default_material)
    {
      texts->add(buffer.str(), p);
    }
    else if (vec_color)
    {
      Vector vval;
      fld->get_value(vval, *iter);
      texts->add(buffer.str(), p, Color(vval.x(), vval.y(), vval.z()));
    }
    else
    {
      double dval;
      fld->get_value(dval, *iter);
      texts->add(buffer.str(), p, dval);
    }

    ++iter;
  }
  return text_switch;
}

} // end namespace SCIRun
