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

//    File   : RenderField.h
//    Author : Martin Cole
//    Date   : Fri May 11 15:49:10 2001

#if !defined(Visualization_RenderField_h)
#define Visualization_RenderField_h

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Math/MiscMath.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomArrows.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomCone.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomQuads.h>
#include <Core/Geom/GeomBox.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomDL.h>
#include <Core/Geom/GeomColorMap.h>
#include <Core/Geom/GeomPoint.h>
#include <Core/Geom/GeomTexRectangle.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Containers/StringUtil.h>
#include <sci_hash_map.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Datatypes/ImageMesh.h>
#include <sstream>
#include <iostream>

using std::cerr;

#if defined(_WIN32) && !defined(uint)
  // for some reason, this isn't defined...
#define uint unsigned
#endif


namespace SCIRun {
class GeomEllipsoid;
class GeomArrows;

typedef ImageMesh<QuadBilinearLgn<Point> > IMesh;
inline void
sciVectorToColor(Color &c, const Vector &v)
{
  c = Color(fabs(v.x()), fabs(v.y()), fabs(v.z()));
}

inline bool IsPowerOf2(uint n)
{
  return (n & (n-1)) == 0;
}

inline unsigned int NextPowerOf2( unsigned int n)
{
  // if n is power of 2, return 
  if (IsPowerOf2(n)) return n;
  unsigned int v;
  for(int i=31; i>=0; i--) {
    v = n & (1 << i);
    if (v) {
      v = (1 << (i+1));
      break;
    }
  }
  return v;
}

//! RenderFieldBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! RenderFieldBase from the DynamicAlgoBase they will have a pointer to.
class RenderFieldBase : public DynamicAlgoBase
{
public:
  virtual void render(FieldHandle f,
		      bool nodes, bool edges, bool faces,
		      ColorMapHandle color_handle, MaterialHandle def_mat,
		      const string &ndt, const string &edt, 
		      double ns, double es, double vs, bool normalize, 
		      int sphere_resolution, int cylinder_resolution,
		      bool use_normals,
		      bool node_transparency,
		      bool edge_transparency,
		      bool face_transparency,
		      bool node_force_def_color,
		      bool edge_force_def_color,
		      bool face_force_def_color,
		      unsigned approx_div,
		      bool face_usetexture) = 0;

  virtual GeomHandle render_text(FieldHandle fld,
				 bool use_color_map,
				 bool use_default_material,
				 bool backface_cull_p,
				 int  fontsize,
				 int  precision,
				 bool render_locations,
				 bool render_data,
				 bool render_nodes,
				 bool render_edges,
				 bool render_faces,
				 bool render_cells) = 0;

  RenderFieldBase();
  virtual ~RenderFieldBase();

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *ltd);
  
  GeomHandle               node_switch_;
  GeomHandle               edge_switch_;
  GeomHandle               face_switch_;

protected:
  void add_axis(const Point &p, double scale, GeomLines *lines);
  void add_axis(const Point &p, double scale, GeomLines *lines, double val);
  void add_axis(const Point &p, double scale, GeomLines *lines,
		const MaterialHandle &vcol);
};


template <class Fld, class Loc>
class RenderField : public RenderFieldBase
{
public:
  //! virtual interface. 
  virtual void render(FieldHandle fh,  
		      bool nodes, bool edges, bool faces,
		      ColorMapHandle color_handle,
		      MaterialHandle def_mat,
		      const string &ndt, const string &edt, 
		      double ns, double es, double vs, bool normalize, 
		      int sphere_resolution, int cylinder_resolution,
		      bool use_normals,
		      bool node_transparency,
		      bool edge_transparency,
		      bool face_transparency,
		      bool node_force_def_color,
		      bool edge_force_def_color,
		      bool face_force_def_color,
		      unsigned approx_div,
		      bool face_usetexture);


  virtual GeomHandle render_text(FieldHandle fld,
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
				 bool render_cells);

protected:
  GeomHandle render_nodes(Fld *fld, 
			  const string &node_display_mode,
			  ColorMapHandle color_handle,
			  MaterialHandle def_mat,
			  bool force_def_color,
			  double node_scale,
			  int node_resolution,
			  bool use_transparency);
  GeomHandle render_edges(Fld *fld,
			  const string &edge_display_mode,
			  ColorMapHandle color_handle,
			  MaterialHandle def_mat,
			  bool force_def_color,
			  double edge_scale,
			  int cylinder_resolution,
			  bool transparent_p, unsigned div);
  GeomHandle render_faces(Fld *fld, 
			  ColorMapHandle color_handle,
			  MaterialHandle def_mat,
			  bool force_def_color,
			  bool use_normals,
			  bool use_transparency, unsigned div,
			  bool use_texture_for_face = false);
  virtual GeomHandle render_texture_face(Fld *fld, 
                                         ColorMapHandle color_handle,
                                         MaterialHandle def_mat,
                                         bool force_def_color,
                                         bool use_normals,
                                         bool use_transparency);

  GeomHandle render_text_data(FieldHandle fld,
			      bool use_color_map,
			      bool use_default_material,
			      bool backface_cull_p,
			      int fontsize,
			      int precision);
  GeomHandle render_text_data_nodes(FieldHandle fld,
				    bool use_color_map,
				    bool use_default_material,
				    bool backface_cull_p,
				    int fontsize,
				    int precision);
  GeomHandle render_text_nodes(FieldHandle fld,
			       bool use_color_map,
			       bool use_default_material,
			       bool backface_cull_p,
			       int fontsize,
			       int precision,
			       bool render_locations);
  GeomHandle render_text_edges(FieldHandle fld,
			       bool use_color_map,
			       bool use_default_material,
			       int fontsize,
			       int precision,
			       bool render_locations);
  GeomHandle render_text_faces(FieldHandle fld,
			       bool use_color_map,
			       bool use_default_material,
			       int fontsize,
			       int precision,
			       bool render_locations);
  GeomHandle render_text_cells(FieldHandle fld,
			       bool use_color_map,
			       bool use_default_material,
			       int fontsize,
			       int precision,
			       bool render_locations);
};


template <class T>
bool
to_vector(const T& tmp, Vector &val)
{
  return false;
}


template <>
bool
to_vector(const Vector&, Vector &);


template <class T>
bool
to_double(const T& tmp, double &val)
{
  val = (double)tmp;
  return true;
}

template <>
bool
to_double(const Vector&, double &);

template <>
bool
to_double(const Tensor&, double &);


template <>
bool
to_double(const string&, double &);

template <class T>
bool
to_float(const T& tmp, float &val)
{
  val = (float)tmp;
  return true;
}

template <>
bool
to_float(const Vector&, float &);

template <>
bool
to_float(const Tensor&, float &);


template <>
bool
to_float(const string&, float &);


template <class Dat>
bool 
add_data(const Point &, const Dat &, GeomArrows *, 
	 MaterialHandle &, const string &, double, bool, bool)
{
  return false;
}

template <>
bool 
add_data(const Point &, const Vector &, GeomArrows *, 
	 MaterialHandle &, const string &, double, bool, bool);

template <>
bool 
add_data(const Point &, const Tensor &, GeomArrows *, 
	 MaterialHandle &, const string &, double, bool, bool);



template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::render(FieldHandle fh,  bool nodes, 
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
  Fld *fld = dynamic_cast<Fld*>(fh.get_rep());
  ASSERT(fld != 0);

  if (nodes)
  {
    node_switch_ = render_nodes(fld, ndt, color_handle, def_mat, nfdc,
				ns, sphere_res, n_transp);
  }
  
  if (edges)
  {
    edge_switch_ = render_edges(fld, edt, color_handle, def_mat, efdc,
				es, cyl_res, e_transp, div);
  }

  if (faces)
  {
    face_switch_ = render_faces(fld, color_handle, def_mat, ffdc,
				use_normals, f_transp, div, fut);
  }
}



template <class Fld, class Loc>
GeomHandle
RenderField<Fld, Loc>::render_nodes(Fld *sfld, 
				    const string &node_display_mode,
				    ColorMapHandle color_handle,
				    MaterialHandle def_mat,
				    bool force_def_color,
				    double node_scale,
				    int node_resolution,
				    bool use_transparency)
{
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
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
  if (def_color && sfld->query_vector_interface().get_rep()
      && !force_def_color)
  {
    def_color = false;
    vec_color = true;
    vcol = scinew Material();
    vcol->transparency = 1.0;
  }

  // First pass: over the nodes
  mesh->synchronize(Mesh::NODES_E);
  typename Fld::mesh_type::Node::iterator niter;  mesh->begin(niter);  
  typename Fld::mesh_type::Node::iterator niter_end;  mesh->end(niter_end);  
  while (niter != niter_end) {
    Point p;
    mesh->get_point(p, *niter);

    // val is double because the color index field must be scalar.
    Vector vec(0,0,0);
    double val;

    if ((sfld->basis_order() > 0) || (sfld->basis_order() == 0 && 
				      mesh->dimensionality() == 1)) {
      typename Fld::value_type tmp;
      sfld->value(tmp, *niter);
	
      to_vector(tmp, vec);
      to_double(tmp, val);
      if (vec_color) { sciVectorToColor(vcol->diffuse, vec); }

    } else {
      def_color = true;
    }

    switch (mode)
    {
    case 0: // Points
      if (def_color)
      {
	points->add(p);
      }
      else if (vec_color)
      {
	points->add(p, vcol);
      }
      else
      {
	points->add(p, val);
      }
      break;

    case 1: // Spheres
      if (def_color)
      {
	spheres->add(p);
      }
      else if (vec_color)
      {
	spheres->add(p, vcol);
      }
      else
      {
	spheres->add(p, val);
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
	  spheres->add(p);
	}
	else if (vec_color)
	{
	  spheres->add(p, vcol);
	}
	else
	{
	  spheres->add(p, val);
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

template <class fc_t>
struct eqfc
{
  bool operator()(const fc_t &s1, const fc_t &s2) const
  {
    for(unsigned i = 0; i < s1.size(); ++i) {
      if (s1[i] != s2[i]) return false;
    }
    return true;
  }
};

template <class fc_t>
struct hash_nds
{
  size_t operator()(const fc_t &f) const {
    //const unsigned spc = sizeof(size_t);
    //const unsigned div = f.size();
    //const unsigned spc_div = spc / div;
    size_t val = 0;
//     for (unsigned i = 0; i < div; ++i) {
//       val = val | (f[i] << ((div - i - 1) * spc_div));
//       cerr << "hash[" << i << "]: "<< val;
//     }

    val = (f[0] << 24) | (f[1] << 16) | (f[2] << 8) | f[3];
    //    cerr << "hard hash: " << val;
    //    cerr << endl;
    return val;
  }
};

template <class Val_t>
void add_edge_geom(GeomLines *lines, GeomCylinders *cylinders,
		   const Point &p0, const Point &p1, 
		   const Val_t &val0, const Val_t &val1,
		   bool def_color, bool vec_color, bool cyl,
		   MaterialHandle vcol0, MaterialHandle vcol1)
{
  if (def_color)
  {
    if (cyl)
    {
      cylinders->add(p0, p1);
    }
    else
    {
      lines->add(p0, p1);
    }
  }
  else if (vec_color)
  {
    Vector v0(0, 0, 0), v1(0, 0, 0);
    to_vector(val0, v0);
    to_vector(val1, v1);
    sciVectorToColor(vcol0->diffuse, v0);
    sciVectorToColor(vcol1->diffuse, v1);
    if (cyl)
    {
      cylinders->add(p0, vcol0, p1, vcol1);
    }
    else
    {
      lines->add(p0, vcol0, p1, vcol1);
    }
  }
  else
  {
    double dval0, dval1;
    to_double(val0, dval0);
    to_double(val1, dval1);
    if (cyl)
    {
      cylinders->add(p0, dval0, p1, dval1);
    }
    else
    {
      lines->add(p0, dval0, p1, dval1);
    }
  }
}

template <class Fld, class Loc>
GeomHandle
RenderField<Fld, Loc>::render_edges(Fld *sfld,
				    const string &edge_display_mode,
				    ColorMapHandle color_handle,
				    MaterialHandle def_mat,
				    bool force_def_color,
				    double edge_scale,
				    int cylinder_resolution,
				    bool transparent_p, unsigned div) 
{
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();

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
    lines->setLineWidth(edge_scale);
  }

  // Use a default color?
  bool def_color = !(color_handle.get_rep()) || force_def_color;
  bool vec_color = false;
  MaterialHandle vcol0(0), vcol1(0);
  if (def_color && sfld->query_vector_interface().get_rep()
      && !force_def_color)
  {
    def_color = false;
    vec_color = true;
    vcol0 = scinew Material();
    vcol0->transparency = 1.0;
    vcol1 = scinew Material();
    vcol1->transparency = 1.0;
  }

  // Second pass: over the edges
  mesh->synchronize(Mesh::EDGES_E | Mesh::FACES_E | Mesh::CELLS_E);
  typename Fld::mesh_type::Elem::iterator eiter; mesh->begin(eiter);  
  typename Fld::mesh_type::Elem::iterator eiter_end; mesh->end(eiter_end);  
  while (eiter != eiter_end) {  
    typename Fld::mesh_type::Edge::array_type edges;
    mesh->get_edges(edges, *eiter);
    // render all the edges FIX_ME has the edges, render each only once.
    typename Fld::mesh_type::Edge::array_type::iterator edge_iter;
    edge_iter = edges.begin();
    while (edge_iter != edges.end()) {
    
      vector<vector<double> > coords;
      mesh->pwl_approx_edge(coords, *eiter, *edge_iter++, div);
      vector<vector<double> >::iterator coord_iter = coords.begin();
      do {
	vector<double> &c0 = *coord_iter++;
	if (coord_iter == coords.end()) break;
	vector<double> &c1 = *coord_iter;
	Point p0, p1;      
	typename Fld::value_type val0, val1;
	
	// get the geometry at the approx.
	mesh->interpolate(p0, c0, *eiter);
	mesh->interpolate(p1, c1, *eiter);

	// get the field variables values at the approx (if they exist)
	if (sfld->basis_order() >= 0) {
	  sfld->interpolate(val0, c0, *eiter);
	  sfld->interpolate(val1, c1, *eiter);
	}
	// add the geom_obj for this part....
	  
	add_edge_geom(lines, cylinders, p0, p1, val0, val1,
		      def_color, vec_color, cyl, vcol0, vcol1);
	
      } while (coords.size() > 1 && coord_iter != coords.end()); 
    }
    
    ++eiter;
  }
  
  return display_list;
}


template <class Val_t>
void add_face_geom(GeomFastTriangles *faces, GeomFastQuads *qfaces,
		   const vector<Point> &points,
		   const vector<Vector> &normals,
		   const vector<Val_t>  &vals,
		   bool def_color,
		   bool vec_color,
		   bool with_normals,
		   vector<MaterialHandle> &vcol,
		   vector<Vector> &vvals,
		   vector<double> &dvals)
{
  if (def_color) {
    if (points.size() == 4)
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
      for (unsigned i = 2; i < points.size(); i++)
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
  } else  if (vec_color) {
    for (unsigned i = 0; i < points.size(); i++)
    {
      to_vector(vals[i], vvals[i]);
      sciVectorToColor(vcol[i]->diffuse, vvals[i]);
    }
    if (points.size() == 4)
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
      for (unsigned i = 2; i < points.size(); i++)
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
    for (unsigned i = 0; i < points.size(); i++)
    {
      to_double(vals[i], dvals[i]);
    }
    if (points.size() == 4)
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
      for (unsigned i = 2; i < points.size(); i++)
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

template <class Fld, class Loc>      //      cerr << " - render - ";


GeomHandle
RenderField<Fld, Loc>::render_texture_face(Fld *sfld,
                                           ColorMapHandle color_handle,
                                           MaterialHandle def_mat,
                                           bool force_def_color,
                                           bool use_normals,
                                           bool use_transparency)
{
  return 0;
}

template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_faces(Fld *sfld,
				    ColorMapHandle color_handle,
				    MaterialHandle def_mat,
				    bool force_def_color,
				    bool use_normals,
				    bool use_transparency, unsigned div,
				    bool use_texture_for_face)
{
  unsigned int i;

  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();

  // if we have an ImageMesh, we can render it as a single polygon
  // with a textured face, providing the flag is true.
  if(dynamic_cast<IMesh *> (mesh.get_rep())){
    if( use_texture_for_face ){
      return render_texture_face(sfld, color_handle, def_mat,
				 force_def_color, use_normals,
				 use_transparency);
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
  vector<typename Fld::value_type> vals(20);
  vector<double> dvals(20);
  if (def_color && sfld->query_vector_interface().get_rep()
      && !force_def_color)
  {
    def_color = false;
    vec_color = true;
    for (i = 0; i < 20; i++)
    {
      vcol[i] = scinew Material();
      vcol[i]->transparency = 1.0;
    }
  }


  typedef typename Fld::mesh_type::Node::array_type fc_t;
  typedef hash_set<fc_t, hash_nds<fc_t>, eqfc<fc_t> > face_ht_t;
  face_ht_t rendered_faces; 
  
  mesh->synchronize(Mesh::FACES_E | Mesh::CELLS_E);
  typename Fld::mesh_type::Elem::iterator eiter; mesh->begin(eiter);  
  typename Fld::mesh_type::Elem::iterator eiter_end; mesh->end(eiter_end);  
  int count = 0;
  while (eiter != eiter_end) {  
    typename Fld::mesh_type::Face::array_type face_indecies;
    mesh->get_faces(face_indecies, *eiter);
    // render all the edges FIX_ME has the edges, render each only once.
    typename Fld::mesh_type::Face::array_type::iterator face_iter;
    face_iter = face_indecies.begin();
    while (face_iter != face_indecies.end()) {
      typename Fld::mesh_type::Node::array_type nodes;
      typename Fld::mesh_type::Face::index_type fidx = *face_iter++;

      mesh->get_nodes(nodes, fidx);
      sort(nodes.begin(), nodes.end());
      
      typename face_ht_t::const_iterator it = rendered_faces.find(nodes);

      if (it != rendered_faces.end()) {
	continue;
      } else {
	rendered_faces.insert(nodes);
      }
      count++;

      //coords organized as scanlines of quad/tri strips.
      vector<vector<vector<double> > > coords;
      mesh->pwl_approx_face(coords, *eiter, fidx, div);
      const int face_sz = mesh->get_basis().get_approx_face_elements();

      vector<vector<vector<double> > >::iterator coord_iter = coords.begin();
      while (coord_iter != coords.end()) {
	vector<vector<double> > &sl = *coord_iter++;
	vector<vector<double> >::iterator sliter = sl.begin();

	for (unsigned int i = 0; i < sl.size() - 2; i++) {
	  //while(sliter != sl.end()) {
	  if (face_sz == 3) {  // TRI STRIPS
	    vector<vector<double> >::iterator it0,it1;
	    
	    vector<double> &c0 = !i%2 ? sl[i] : sl[i+1];
	    vector<double> &c1 = !i%2 ? sl[i+1] : sl[i];
	    vector<double> &c2 = sl[i+2];

	    const int face_sz = mesh->get_basis().get_approx_face_elements();
	    vector<Point> pnts(face_sz);
	    vector<Vector> norms(face_sz);
	    vector<typename Fld::value_type> vals(face_sz);
	  
	    mesh->interpolate(pnts[0], c0, *eiter);
	    mesh->interpolate(pnts[1], c1, *eiter);
	    mesh->interpolate(pnts[2], c2, *eiter);
	  
	    //FIX_ME need to interp normals in meshes that have normals
	  
	    // get the field variables values at the approx (if they exist)
	    if (sfld->basis_order() >= 0) {
	      sfld->interpolate(vals[0], c0, *eiter);
	      sfld->interpolate(vals[1], c1, *eiter);
	      sfld->interpolate(vals[2], c2, *eiter);
	    
	    } else {
	      def_color = true;
	    }
	    // add the geom_obj for this part....
	  
	    add_face_geom(faces, qfaces, pnts, norms, vals,
			  def_color, vec_color, with_normals, 
			  vcol, vvals, dvals);

	  } else { // QUADS

	    vector<double> &c0 = *sliter++;
	    if (sliter == sl.end()) break;
	    vector<double> &c1 = *sliter++;
	    if (sliter == sl.end()) break;
	    vector<double> &c2 = *sliter;
	    vector<double> &c3 = *(sliter + 1);

	    vector<Point> pnts(face_sz);
	    vector<Vector> norms(face_sz);
	    vector<typename Fld::value_type> vals(face_sz);
	  
	    // get the geometry at the approx.
	    mesh->interpolate(pnts[0], c2, *eiter);
	    mesh->interpolate(pnts[1], c3, *eiter);
	    mesh->interpolate(pnts[2], c1, *eiter);
	    mesh->interpolate(pnts[3], c0, *eiter);

	    //FIX_ME need to interp normals in meshes that have normals

	    // get the field variables values at the approx (if they exist)
	    if (sfld->basis_order() >= 0) {
	      sfld->interpolate(vals[0], c2, *eiter);
	      sfld->interpolate(vals[1], c3, *eiter);
	      sfld->interpolate(vals[2], c1, *eiter);
	      sfld->interpolate(vals[3], c0, *eiter);
	    } else {
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
    ++eiter;
  }
  return face_switch;
}


template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_text(FieldHandle field_handle,
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


template <class T>
void
value_to_string(std::ostringstream &buffer, const T &value)
{
  buffer << value;
}

template <>
void value_to_string(std::ostringstream &buffer, const char &value);

template <>
void value_to_string(std::ostringstream &buffer, const unsigned char &value);


template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_text_data(FieldHandle field_handle,
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

  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  GeomTexts *texts = scinew GeomTexts();
  GeomHandle text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  texts->set_font_index(fontsize);

  std::ostringstream buffer;
  buffer.precision(precision);

  bool vec_color = false;
  if (!use_color_map)
  {
    if (!use_default_material &&
	field_handle->query_vector_interface().get_rep())
    {
      vec_color = true;
    }
    else
    {
      use_default_material = true;
    }
  }
 
  typename Loc::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  while (iter != end)
  {
    typename Fld::value_type val;
    if (fld->value(val, *iter))
    {
      mesh->get_center(p, *iter);

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
    ++iter;
  }

  return text_switch;
}


template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_text_data_nodes(FieldHandle field_handle,
					      bool use_color_map,
					      bool use_default_material,
					      bool backface_cull_p,
					      int fontsize,
					      int precision)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

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
    if (!use_default_material &&
	field_handle->query_vector_interface().get_rep())
    {
      vec_color = true;
    }
    else
    {
      use_default_material = true;
    }
  }

  typename Fld::mesh_type::Node::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  Vector n;
  while (iter != end) {
    typename Fld::value_type val;
    if (fld->value(val, *iter)) {
      mesh->get_center(p, *iter);
      
      buffer.str("");
      value_to_string(buffer, val);

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
	Vector vval(0, 0, 0);
	to_vector(val, vval);
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
	double dval = 0.0;
	to_double(val, dval);
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
    }
    ++iter;
  }

  return text_switch;
}



template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_text_nodes(FieldHandle field_handle,
					 bool use_color_map,
					 bool use_default_material,
					 bool backface_cull_p,
					 int fontsize,
					 int precision,
					 bool render_locations)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();
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

  bool vec_color = false;
  if (!(fld->basis_order() == 1 || mesh->dimensionality() == 0))
  {
    use_default_material = true;
  }
  else if (!use_color_map)
  {
    if (!use_default_material &&
	field_handle->query_vector_interface().get_rep())
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

  typename Fld::mesh_type::Node::iterator iter, end;
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
      buffer << (*iter);
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
      typename Fld::value_type val;
      fld->value(val, *iter);
      Vector vval(0, 0, 0);
      to_vector(val, vval);
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
      typename Fld::value_type val;
      fld->value(val, *iter);
      double dval = 0.0;
      to_double(val, dval);
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


template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_text_edges(FieldHandle field_handle,
					 bool use_color_map,
					 bool use_default_material,
					 int fontsize,
					 int precision,
					 bool render_locations)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();
  mesh->synchronize(Mesh::EDGES_E);

  GeomTexts *texts = scinew GeomTexts;
  GeomHandle text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  texts->set_font_index(fontsize);

  ostringstream buffer;
  buffer.precision(precision);

  bool vec_color = false;
  if (! (fld->basis_order() == 0 && mesh->dimensionality() == 1))
  {
    use_default_material = true;
  }
  else if (!use_color_map)
  {
    if (!use_default_material &&
	field_handle->query_vector_interface().get_rep())
    {
      vec_color = true;
    }
    else
    {
      use_default_material = true;
    }
  }

  typename Fld::mesh_type::Edge::iterator iter, end;
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
      typename Fld::value_type val;
      fld->value(val, *iter);
      Vector vval(0, 0, 0);
      to_vector(val, vval);
      texts->add(buffer.str(), p, Color(vval.x(), vval.y(), vval.z()));
    }
    else
    {
      typename Fld::value_type val;
      fld->value(val, *iter);
      double dval = 0.0;
      to_double(val, dval);
      texts->add(buffer.str(), p, dval);
    }
 
    ++iter;
  }
  return text_switch;
}

template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_text_faces(FieldHandle field_handle,
					 bool use_color_map,
					 bool use_default_material,
					 int fontsize,
					 int precision,
					 bool render_locations)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();
  mesh->synchronize(Mesh::FACES_E);

  GeomTexts *texts = scinew GeomTexts;
  GeomHandle text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  texts->set_font_index(fontsize);

  ostringstream buffer;
  buffer.precision(precision);

  bool vec_color = false;
  if (! (fld->basis_order() == 0 && mesh->dimensionality() == 2))
  {
    use_default_material = true;
  }
  else if (!use_color_map)
  {
    if (!use_default_material &&
	field_handle->query_vector_interface().get_rep())
    {
      vec_color = true;
    }
    else
    {
      use_default_material = true;
    }
  }

  typename Fld::mesh_type::Face::iterator iter, end;
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
      buffer << (*iter);
    }

    if (use_default_material)
    {
      texts->add(buffer.str(), p);
    }
    else if (vec_color)
    {
      typename Fld::value_type val;
      fld->value(val, *iter);
      Vector vval(0, 0, 0);
      to_vector(val, vval);
      texts->add(buffer.str(), p, Color(vval.x(), vval.y(), vval.z()));
    }
    else
    {
      typename Fld::value_type val;
      fld->value(val, *iter);
      double dval = 0.0;
      to_double(val, dval);
      texts->add(buffer.str(), p, dval);
    }

    ++iter;
  }
  return text_switch;
}


template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_text_cells(FieldHandle field_handle,
					 bool use_color_map,
					 bool use_default_material,
					 int fontsize,
					 int precision,
					 bool render_locations)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();
  mesh->synchronize(Mesh::CELLS_E);

  GeomTexts *texts = scinew GeomTexts;
  GeomHandle text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  texts->set_font_index(fontsize);

  ostringstream buffer;
  buffer.precision(precision);

  bool vec_color = false;
  if (fld->basis_order() != 0)
  {
    use_default_material = true;
  }
  else if (!use_color_map)
  {
    if (!use_default_material &&
	field_handle->query_vector_interface().get_rep())
    {
      vec_color = true;
    }
    else
    {
      use_default_material = true;
    }
  }

  typename Fld::mesh_type::Cell::iterator iter, end;
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
      buffer << (*iter);
    }

    if (use_default_material)
    {
      texts->add(buffer.str(), p);
    }
    else if (vec_color)
    {
      typename Fld::value_type val;
      fld->value(val, *iter);
      Vector vval(0, 0, 0);
      to_vector(val, vval);
      texts->add(buffer.str(), p, Color(vval.x(), vval.y(), vval.z()));
    }
    else
    {
      typename Fld::value_type val;
      fld->value(val, *iter);
      double dval = 0.0;
      to_double(val, dval);
      texts->add(buffer.str(), p, dval);
    }

    ++iter;
  }
  return text_switch;
}


template <class Fld, class Loc>
class RenderFieldImage : public RenderField<Fld, Loc>
{
protected:
  virtual GeomHandle render_texture_face(Fld *fld, 
                                         ColorMapHandle color_handle,
                                         MaterialHandle def_mat,
                                         bool force_def_color,
                                         bool use_normals,
                                         bool use_transparency);
};


template <class Fld, class Loc>
GeomHandle
RenderFieldImage<Fld, Loc>::render_texture_face(Fld *sfld,
                                                ColorMapHandle color_handle,
                                                MaterialHandle def_mat,
                                                bool force_def_color,
                                                bool use_normals,
                                                bool use_transparency)
{
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
  GeomHandle texture_face;
  float tex_coords[8];
  float pos_coords[12];
  const int colorbytes = 4;

  GeomTexRectangle *tr = scinew GeomTexRectangle();
  texture_face = tr;

  IMesh *im = dynamic_cast<IMesh *> (mesh.get_rep());

  // Set up the texture parameters, power of 2 dimensions.
  int width = NextPowerOf2(im->get_ni());
  int height = NextPowerOf2(im->get_nj());

  // Use for the texture coordinates 
  double tmin_x, tmax_x, tmin_y, tmax_y;

  // Create texture array 
  unsigned char * texture = new unsigned char[colorbytes*width*height];

  //***************************************************
  // we need to find the corners of the square in space
  // use the node indices to grab the corner points
  typename Fld::mesh_type::Node::index_type ll(im, 0, 0);
  typename Fld::mesh_type::Node::index_type lr(im, im->get_ni() - 1, 0);
  typename Fld::mesh_type::Node::index_type ul(im, 0, im->get_nj() - 1);
  typename Fld::mesh_type::Node::index_type ur(im, im->get_ni() - 1, 
					       im->get_nj() - 1);
  
  Point p1, p2, p3, p4;
  im->get_center(p1, ll);
  pos_coords[0] = p1.x();
  pos_coords[1] = p1.y();
  pos_coords[2] = p1.z();

  im->get_center(p2, lr);
  pos_coords[3] = p2.x();
  pos_coords[4] = p2.y();
  pos_coords[5] = p2.z();

  im->get_center(p3, ur);
  pos_coords[6] = p3.x();
  pos_coords[7] = p3.y();
  pos_coords[8] = p3.z();

  im->get_center(p4, ul);
  pos_coords[9] = p4.x();
  pos_coords[10] = p4.y();
  pos_coords[11] = p4.z();

  vector<typename Fld::value_type> vals(20);
  vector<double> dvals(20);

  if ( sfld->basis_order() == 1)
  {
    tr->interpolate(true);

    tmin_x = 0.5/(double)width;
    tmax_x = (im->get_ni()- 0.5)/(double)width;
    tmin_y = 0.5/(double)height;
    tmax_y = (im->get_nj()-0.5)/(double)height;

    tex_coords[0] = tmin_x; tex_coords[1] = tmin_y;
    tex_coords[2] = tmax_x; tex_coords[3] = tmin_y;
    tex_coords[4] = tmax_x; tex_coords[5] = tmax_y;
    tex_coords[6] = tmin_x; tex_coords[7] = tmax_y;

    typename Fld::mesh_type::Node::iterator niter; mesh->begin(niter);  
    typename Fld::mesh_type::Node::iterator niter_end; mesh->end(niter_end);  
    typename Fld::mesh_type::Node::array_type nodes;
    while(niter != niter_end )
    {
      // Convert data values to double.
      typename Fld::value_type val;
      double dval;      
      sfld->value(val, *niter);
      to_double(val, dval);

      // Compute index into texture array.
      const int idx = (niter.i_ * colorbytes) + (niter.j_ * width *colorbytes);
      
      // Compute the ColorMap index and retreive the color.
      const double cmin = color_handle->getMin();
      const double cmax = color_handle->getMax();
      const double index = Clamp((dval - cmin)/(cmax - cmin), 0.0, 1.0);
      const Color &c = color_handle->getColor(index);

      // Fill the texture.
      texture[idx] = (unsigned char)(Clamp(c.r(), 0.0, 1.0)*255);
      texture[idx+1] =  (unsigned char)(Clamp(c.g(), 0.0, 1.0)*255);
      texture[idx+2] = (unsigned char)(Clamp(c.b(), 0.0, 1.0)*255);
      texture[idx+3] = (unsigned char)(Clamp(color_handle->getAlpha(index),
                                             0.0, 1.0)*255);
      ++niter;
    }
  }
  else if( sfld->basis_order() == 0)
  {
     tr->interpolate( false );
     tmin_x = 0.0;
     tmax_x = (im->get_ni()-1)/(double)width;
     tmin_y = 0.0;
     tmax_y = (im->get_nj()-1)/(double)height;
     tex_coords[0] = tmin_x; tex_coords[1] = tmin_y;
     tex_coords[2] = tmax_x; tex_coords[3] = tmin_y;
     tex_coords[4] = tmax_x; tex_coords[5] = tmax_y;
     tex_coords[6] = tmin_x; tex_coords[7] = tmax_y;
      
     typename Fld::mesh_type::Face::iterator fiter; mesh->begin(fiter);  
     typename Fld::mesh_type::Face::iterator fiter_end; mesh->end(fiter_end);  

     while (fiter != fiter_end)
     {
       typename Fld::value_type val;
       double dval;
       sfld->value(val, *fiter);
       to_double(val, dval);

       // Compute index into texture array.
       const int idx = (fiter.i_ * colorbytes) + (fiter.j_ * width *colorbytes);
       // Compute the ColorMap index and retreive the color.
       const double cmin = color_handle->getMin();
       const double cmax = color_handle->getMax();
       const double index = Clamp((dval - cmin)/(cmax - cmin), 0.0, 1.0);
       const Color &c = color_handle->getColor(index);

       // Fill the texture.
       texture[idx] = (unsigned char)(Clamp(c.r(), 0.0, 1.0)*255);
       texture[idx+1] =  (unsigned char)(Clamp(c.g(), 0.0, 1.0)*255);
       texture[idx+2] = (unsigned char)(Clamp(c.b(), 0.0, 1.0)*255);
       texture[idx+3] = (unsigned char)(Clamp(color_handle->getAlpha(index),
                                              0.0, 1.0) * 255);
       ++fiter;
     }
   }

  // Set normal for lighting.
  Vector normal = Cross( p2 - p1, p4 - p1 );
  normal.normalize();
  float n[3];
  n[0] = normal.x(); n[1] = normal.y(); n[2] = normal.z();
  tr->set_normal( n );

  if( use_transparency )
  {
    tr->set_transparency( true );
  }

  tr->set_coords(tex_coords, pos_coords);
  tr->set_texture( texture, colorbytes, width, height );

  delete [] texture;
  return texture_face;
}



//! RenderVectorFieldBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! RenderVectorFieldBase from the DynamicAlgoBase they will have a pointer to.
class RenderVectorFieldBase : public DynamicAlgoBase
{
public:

  virtual GeomHandle render_data(FieldHandle vfld_handle,
				 FieldHandle cfld_handle,
				 ColorMapHandle cmap,
				 MaterialHandle default_material,
				 bool force_def_color,
				 const string &data_display_mode,
				 double scale,
				 double linewidth,
				 bool normalize,
				 bool bidirectional,
				 int resolution) = 0;



  RenderVectorFieldBase();
  virtual ~RenderVectorFieldBase();

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *vftd,
					    const TypeDescription *cftd,
					    const TypeDescription *ltd);
};


template <class VFld, class CFld, class Loc>
class RenderVectorField : public RenderVectorFieldBase
{
public:
  virtual GeomHandle render_data(FieldHandle vfld_handle,
				 FieldHandle cfld_handle,
				 ColorMapHandle cmap,
				 MaterialHandle default_material,
				 bool force_def_color,
				 const string &data_display_mode,
				 double scale,
				 double linewidth,
				 bool normalize,
				 bool bidirectional,
				 int resolution);
};


template <class VFld, class CFld, class Loc>
GeomHandle 
RenderVectorField<VFld, CFld, Loc>::render_data(FieldHandle vfld_handle,
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
  VFld *vfld = dynamic_cast<VFld*>(vfld_handle.get_rep());
  CFld *cfld = dynamic_cast<CFld*>(cfld_handle.get_rep());

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
  if (def_color && cfld->query_vector_interface().get_rep()
      && !force_def_color)
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

  typename VFld::mesh_handle_type mesh = vfld->get_typed_mesh();

  typename Loc::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  while (iter != end)
  {
    typename VFld::value_type tmp;
    if (vfld->value(tmp, *iter))
    {
      Point p;
      mesh->get_center(p, *iter);

      if (disks_p)
      {
	if (tmp.length2() > 1.0e-10)
	{
	  if (normalize) { tmp.safe_normalize(); }
	  tmp *= (scale / 6.0);
	  
	  if (def_color)
	  {
	    if (normalize)
	    {
	      disks->add(p+tmp, p-tmp);
	    }
	    else
	    {
	      disks->add_radius(p+tmp, p+tmp, 6*tmp.length());
	    }
	  }
	  else if (vec_color)
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    Vector vtmp;
	    to_vector(ctmp, vtmp);
	    sciVectorToColor(vcol->diffuse, vtmp);
	    if (normalize)
	    {
	      disks->add(p+tmp, vcol, p-tmp, vcol);
	    }
	    else
	    {
	      disks->add_radius(p+tmp, vcol, p-tmp, vcol, 6*tmp.length());
	    }
	  }
	  else
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    double ctmpd;
	    to_double(ctmp, ctmpd);
	    if (normalize)
	    {
	      disks->add(p+tmp, ctmpd, p-tmp, ctmpd);
	    }
	    else
	    {
	      disks->add_radius(p+tmp, ctmpd, p+tmp, ctmpd, 6*tmp.length());
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
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    Vector vtmp;
	    to_vector(ctmp, vtmp);
	    sciVectorToColor(vcol->diffuse, vtmp);
	    spheres->add(p, vcol);
	  }
	  else
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    double ctmpd;
	    to_double(ctmp, ctmpd);
	    spheres->add(p, ctmpd);
	  }
	}
      }
      else if (cones_p)
      {
	if (tmp.length2() > 1.0e-10)
	{
	  if (normalize) { tmp.safe_normalize(); }
	  tmp *= scale;

	  if (def_color)
	  {
	    if (normalize)
	    {
	      cones->add(p, p+tmp);
	      if (bidirectional)
	      {
		cones->add(p, p-tmp);
	      }
	    }
	    else
	    {
	      const double len = tmp.length() / 6.0;
	      cones->add_radius(p, p+tmp, len);
	      if (bidirectional)
	      {
		cones->add_radius(p, p-tmp, len);
	      }
	    }
	  }
	  else if (vec_color)
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    Vector vtmp;
	    to_vector(ctmp, vtmp);
	    sciVectorToColor(vcol->diffuse, vtmp);
	    if (normalize)
	    {
	      cones->add(p, p+tmp, vcol);
	      if (bidirectional)
	      {
		cones->add(p, p-tmp, vcol);
	      }
	    }
	    else
	    {
	      const double len = tmp.length() / 6.0;
	      cones->add_radius(p, p+tmp, vcol, len);
	      if (bidirectional)
	      {
		cones->add_radius(p, p-tmp, vcol, len);
	      }
	    }
	  }
	  else
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    double ctmpd;
	    to_double(ctmp, ctmpd);
	    if (normalize)
	    {
	      cones->add(p, p+tmp, ctmpd);
	      if (bidirectional)
	      {
		cones->add(p, p-tmp, ctmpd);
	      }
	    }
	    else
	    {
	      const double len = tmp.length() / 6.0;
	      cones->add_radius(p, p+tmp, ctmpd, len);
	      if (bidirectional)
	      {
		cones->add_radius(p, p-tmp, ctmpd, len);
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
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    Vector vtmp;
	    to_vector(ctmp, vtmp);
	    sciVectorToColor(vcol->diffuse, vtmp);
	    spheres->add(p, vcol);
	  }
	  else
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    double ctmpd;
	    to_double(ctmp, ctmpd);
	    spheres->add(p, ctmpd);
	  }
	}
      }
      else if (arrows_p)
      {
	if (tmp.length2() > 1.0e-10)
	{
	  if (normalize) { tmp.safe_normalize(); }
	  tmp *= scale;
	  const Vector ltmp = tmp * 0.6;

	  if (def_color)
	  {
	    if (normalize)
	    {
	      cones->add(p+ltmp, p+tmp);
	      if (bidirectional)
	      {
		cones->add(p-ltmp, p-tmp);
	      }
	    }
	    else
	    {
	      const double len = tmp.length() * 0.15;
	      cones->add_radius(p+ltmp, p+tmp, len);
	      if (bidirectional)
	      {
		cones->add_radius(p-ltmp, p-tmp, len);
	      }
	    }
	  }
	  else if (vec_color)
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    Vector vtmp;
	    to_vector(ctmp, vtmp);
	    sciVectorToColor(vcol->diffuse, vtmp);
	    if (normalize)
	    {
	      cones->add(p+ltmp, p+tmp, vcol);
	      if (bidirectional)
	      {
		cones->add(p-ltmp, p-tmp, vcol);
	      }
	    }
	    else
	    {
	      const double len = tmp.length() * 0.15;
	      cones->add_radius(p+ltmp, p+tmp, vcol, len);
	      if (bidirectional)
	      {
		cones->add_radius(p-ltmp, p-tmp, vcol, len);
	      }
	    }
	  }
	  else
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    double ctmpd;
	    to_double(ctmp, ctmpd);
	    if (normalize)
	    {
	      cones->add(p+ltmp, p+tmp, ctmpd);
	      if (bidirectional)
	      {
		cones->add(p-ltmp, p-tmp, ctmpd);
	      }
	    }
	    else
	    {
	      const double len = tmp.length() * 0.15;
	      cones->add_radius(p+ltmp, p+tmp, ctmpd, len);
	      if (bidirectional)
	      {
		cones->add_radius(p-ltmp, p-tmp, ctmpd, len);
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
	      typename CFld::value_type ctmp;
	      cfld->value(ctmp, *iter);
	      Vector vtmp;
	      to_vector(ctmp, vtmp);
	      sciVectorToColor(vcol->diffuse, vtmp);
	      lines->add(p - ltmp, vcol, p + ltmp, vcol);
	    }
	    else
	    {
	      typename CFld::value_type ctmp;
	      cfld->value(ctmp, *iter);
	      double ctmpd;
	      to_double(ctmp, ctmpd);
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
	      typename CFld::value_type ctmp;
	      cfld->value(ctmp, *iter);
	      Vector vtmp;
	      to_vector(ctmp, vtmp);
	      sciVectorToColor(vcol->diffuse, vtmp);
	      lines->add(p, vcol, p + ltmp, vcol);
	    }
	    else
	    {
	      typename CFld::value_type ctmp;
	      cfld->value(ctmp, *iter);
	      double ctmpd;
	      to_double(ctmp, ctmpd);
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
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    Vector vtmp;
	    to_vector(ctmp, vtmp);
	    sciVectorToColor(vcol->diffuse, vtmp);
	    spheres->add(p, vcol);
	  }
	  else
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    double ctmpd;
	    to_double(ctmp, ctmpd);
	    spheres->add(p, ctmpd);
	  }
	}
      }
      else if (lines_p)
      {
	if (normalize) { tmp.safe_normalize(); }
	tmp *= scale;

	if (bidirectional)
	{
	  if (def_color)
	  {
	    lines->add(p - tmp, p + tmp);
	  }
	  else if (vec_color)
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    Vector vtmp;
	    to_vector(ctmp, vtmp);
	    sciVectorToColor(vcol->diffuse, vtmp);
	    lines->add(p - tmp, vcol, p + tmp, vcol);
	  }
	  else
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    double ctmpd;
	    to_double(ctmp, ctmpd);
	    lines->add(p - tmp, ctmpd, p + tmp, ctmpd);
	  }
	}
	else
	{
	  if (def_color)
	  {
	    lines->add(p, p + tmp);
	  }
	  else if (vec_color)
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    Vector vtmp;
	    to_vector(ctmp, vtmp);
	    sciVectorToColor(vcol->diffuse, vtmp);
	    lines->add(p, vcol, p + tmp, vcol);
	  }
	  else
	  {
	    typename CFld::value_type ctmp;
	    cfld->value(ctmp, *iter);
	    double ctmpd;
	    to_double(ctmp, ctmpd);
	    lines->add(p, ctmpd, p + tmp, ctmpd);
	  }
	}
      }
      else // Needles
      {
	if (normalize) { tmp.safe_normalize(); }
	tmp *= scale;
	
	if (def_color)
	{
	  lines->add(p, opaque, p + tmp, transparent);
	  if (bidirectional)
	  {
	    lines->add(p, opaque, p - tmp, transparent);
	  }
	}
	else if (vec_color)
	{
	  typename CFld::value_type ctmp;
	  cfld->value(ctmp, *iter);
	  Vector vtmp;
	  to_vector(ctmp, vtmp);
	  sciVectorToColor(vcol->diffuse, vtmp);
	  transparent->diffuse = vcol->diffuse;
	  
	  lines->add(p, vcol, p + tmp, transparent);
	  if (bidirectional)
	  {
	    lines->add(p, vcol, p - tmp, transparent);
	  }
	}
	else
	{
	  typename CFld::value_type ctmp;
	  cfld->value(ctmp, *iter);
	  double ctmpd;
	  to_double(ctmp, ctmpd);

	  lines->add(p, opaque, ctmpd, p + tmp, transparent, ctmpd);
	  if (bidirectional)
	  {
	    lines->add(p, opaque, ctmpd, p - tmp, transparent, ctmpd);
	  }
	}
      }
    }
    ++iter;
  }
  return data_switch;
}



//! RenderTensorFieldBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! RenderTensorFieldBase from the DynamicAlgoBase they will have a pointer to.
class RenderTensorFieldBase : public DynamicAlgoBase
{
public:

  virtual GeomHandle render_data(FieldHandle vfld_handle,
				 FieldHandle cfld_handle,
				 ColorMapHandle cmap,
				 MaterialHandle default_material,
				 bool force_def_color,
				 const string &data_display_mode,
				 double scale,
				 int resolution,
				 double emphasis) = 0;



  RenderTensorFieldBase();
  virtual ~RenderTensorFieldBase();

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *vftd,
					    const TypeDescription *cftd,
					    const TypeDescription *ltd);

protected:

  void add_item(GeomGroup *g, GeomHandle glyph, const Point &p, Tensor &t,
		double scale, bool colorize);

  void add_super_quadric(GeomGroup *g, MaterialHandle mat,
			 const Point &p, Tensor &t,
			 double scale, int resolution, bool colorize,
			 double emphasis);

  double map_emphasis(double zero_to_one);
};


template <class VFld, class CFld, class Loc>
class RenderTensorField : public RenderTensorFieldBase
{
public:
  virtual GeomHandle render_data(FieldHandle vfld_handle,
				 FieldHandle cfld_handle,
				 ColorMapHandle cmap,
				 MaterialHandle default_material,
				 bool force_def_color,
				 const string &data_display_mode,
				 double scale,
				 int resolution,
				 double zo_emphasis);
};


template <class VFld, class CFld, class Loc>
GeomHandle 
RenderTensorField<VFld, CFld, Loc>::render_data(FieldHandle vfld_handle,
						FieldHandle cfld_handle,
						ColorMapHandle cmap,
						MaterialHandle def_mat,
						bool force_def_color,
						const string &display_mode,
						double scale, 
						int resolution,
						double zo_emphasis)
{
  VFld *vfld = dynamic_cast<VFld*>(vfld_handle.get_rep());
  CFld *cfld = dynamic_cast<CFld*>(cfld_handle.get_rep()); 

  const bool box_p = (display_mode == "Boxes");
  const bool sphere_p = (display_mode == "Ellipsoids");
  const bool squad_p = (display_mode == "Superquadrics");
  const bool cbox_p = (display_mode == "Colored Boxes");

  const double emph = map_emphasis(zo_emphasis);

  GeomHandle glyph;
  if (box_p)
  {
    glyph = scinew GeomSimpleBox(Point(-1.0, -1.0, -1.0),
				 Point(1.0, 1.0, 1.0));
  }
  else if (sphere_p)
  {
    glyph = scinew GeomSphere(Point(0.0, 0.0, 0.0), 1.0,
			      resolution, resolution);
  }
  else if (squad_p)
  {
    glyph = 0;
  }
  else // cbox_p, default
  {
    glyph = scinew GeomCBox(Point(-1.0, -1.0, -1.0),
			    Point(1.0, 1.0, 1.0));
  }

  GeomGroup *objs = scinew GeomGroup(); 
  GeomHandle data_switch =
    scinew GeomSwitch(scinew GeomMaterial(scinew GeomDL(objs), def_mat));

  int colorstyle = 0;
  if (cmap.get_rep())
  {
    colorstyle = 1;
  }
  else if (cfld->query_vector_interface().get_rep())
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
  if (def_color && cfld->query_vector_interface().get_rep()
      && !force_def_color)
  {
    def_color = false;
    vec_color = true;
  }

  typename VFld::mesh_handle_type mesh = vfld->get_typed_mesh();

  typename Loc::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  while (iter != end)
  {
    typename VFld::value_type tmp;
    if (vfld->value(tmp, *iter))
    {
      Point p;
      mesh->get_center(p, *iter);

      if (colorstyle == 0)
      {
	if (squad_p)
	{
	  add_super_quadric(objs, 0, p, tmp, scale, resolution, true, emph);
	}
	else
	{
	  add_item(objs, glyph, p, tmp, scale, true);
	}
      }
      else if (colorstyle == 1)
      {
	typename CFld::value_type ctmp;
	cfld->value(ctmp, *iter);
	double ctmpd;
	to_double(ctmp, ctmpd);

	if (squad_p)
	{
	  add_super_quadric(objs, cmap->lookup(ctmpd),
			    p, tmp, scale, resolution, false, emph);
	}
	else
	{
	  add_item(objs, scinew GeomMaterial(glyph, cmap->lookup(ctmpd)),
		   p, tmp, scale, false);
	}
      }
      else if (colorstyle == 2)
      {
	typename CFld::value_type ctmp;
	cfld->value(ctmp, *iter);
	Vector ctmpv;
	to_vector(ctmp, ctmpv);

	MaterialHandle vcol = scinew Material();
	sciVectorToColor(vcol->diffuse, ctmpv);
	if (squad_p)
	{
	  add_super_quadric(objs, vcol, p, tmp, scale, resolution, false, emph);
	}
	else
	{
	  add_item(objs, scinew GeomMaterial(glyph, vcol),
		   p, tmp, scale, false);
	}
      }
      else
      {
	if (squad_p)
	{
	  add_super_quadric(objs, 0, p, tmp, scale, resolution, false, emph);
	}
	else
	{
	  add_item(objs, glyph, p, tmp, scale, false);
	}
      }
    }
    ++iter;
  }
  return data_switch;
}


//! RenderScalarFieldBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! RenderScalarFieldBase from the DynamicAlgoBase they will have a pointer to.
class RenderScalarFieldBase : public DynamicAlgoBase
{
public:

  virtual GeomHandle render_data(FieldHandle vfld_handle,
				 FieldHandle cfld_handle,
				 ColorMapHandle cmap,
				 MaterialHandle default_material,
				 bool force_def_color,
				 const string &data_display_mode,
				 double scale,
				 int resolution,
				 bool transparent_p) = 0;



  RenderScalarFieldBase();
  virtual ~RenderScalarFieldBase();

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *vftd,
					    const TypeDescription *cftd,
					    const TypeDescription *ltd);
};


template <class VFld, class CFld, class Loc>
class RenderScalarField : public RenderScalarFieldBase
{
public:
  virtual GeomHandle render_data(FieldHandle vfld_handle,
				 FieldHandle cfld_handle,
				 ColorMapHandle cmap,
				 MaterialHandle default_material,
				 bool force_def_color,
				 const string &data_display_mode,
				 double scale,
				 int resolution,
				 bool transparent_p);
};


template <class SFld, class CFld, class Loc>
GeomHandle 
RenderScalarField<SFld, CFld, Loc>::render_data(FieldHandle sfld_handle,
						FieldHandle cfld_handle,
						ColorMapHandle cmap,
						MaterialHandle def_mat,
						bool force_def_color,
						const string &display_mode,
						double scale, 
						int resolution,
						bool transparent_p)
{
  SFld *sfld = dynamic_cast<SFld*>(sfld_handle.get_rep());
  CFld *cfld = dynamic_cast<CFld*>(cfld_handle.get_rep()); 

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
  
  typename SFld::mesh_handle_type mesh = sfld->get_typed_mesh();

  // Use a default color?
  bool def_color = !(cmap.get_rep()) || force_def_color;
  bool vec_color = false;
  MaterialHandle vcol(0);
  if (def_color && cfld->query_vector_interface().get_rep()
      && !force_def_color)
  {
    def_color = false;
    vec_color = true;
    vcol = scinew Material();
    vcol->transparency = 1.0;
  }

  typename Loc::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  while (iter != end)
  {
    typename SFld::value_type tmp;
    if (sfld->value(tmp, *iter))
    {
      Point p;
      mesh->get_center(p, *iter);

      if (points_p)
      {
	if (def_color)
	{
	  points->add(p);
	}
	else if (vec_color)
	{
	  typename CFld::value_type ctmp;
	  cfld->value(ctmp, *iter);

	  Vector vtmp;
	  to_vector(ctmp, vtmp);
	  sciVectorToColor(vcol->diffuse, vtmp);
	  points->add(p, vcol);
	}
	else
	{
	  typename CFld::value_type ctmp;
	  cfld->value(ctmp, *iter);

	  double ctmpd;
	  to_double(ctmp, ctmpd);
	  points->add(p, ctmpd);
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
	  typename CFld::value_type ctmp;
	  cfld->value(ctmp, *iter);

	  Vector ctmpv;
	  to_vector(ctmp, ctmpv);
	  sciVectorToColor(vcol->diffuse, ctmpv);
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
	  typename CFld::value_type ctmp;
	  cfld->value(ctmp, *iter);

	  double ctmpd;
	  to_double(ctmp, ctmpd);

	  if (sized_p)
	  {
	    const double dtmp = fabs((double)tmp * scale);
	    if (!spheres->add_radius(p, dtmp, ctmpd))
	    {
	      points->add(p, ctmpd);
	    }
	  }
	  else
	  {
	    spheres->add(p, ctmpd);
	  }
	}
      }
    }
    ++iter;
  }
  return data_switch;
}


} // end namespace SCIRun

#endif // Visualization_RenderField_h
