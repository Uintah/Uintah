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
//    File   : RenderField.h
//    Author : Martin Cole
//    Date   : Fri May 11 15:49:10 2001

#if !defined(Visualization_RenderField_h)
#define Visualization_RenderField_h

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/Switch.h>
#include <Core/Geom/GeomArrows.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/Pt.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/ColorMap.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>

namespace SCIRun {

class GeomEllipsoid;
class GeomArrows;

//! RenderFieldBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! RenderFieldBase from the DynamicAlgoBase they will have a pointer to.
class RenderFieldBase : public DynamicAlgoBase
{
public:
  virtual void render(FieldHandle f, bool nodes, bool edges, 
		      bool faces, bool data, MaterialHandle def_mat, 
		      bool def_col, ColorMapHandle color_handle,
		      const string &ndt, const string &edt, 
		      double ns, double es, double vs, int res) = 0;

  virtual ~RenderFieldBase();

  static const string& get_h_file_path();

  static string dyn_file_name(const TypeDescription *td) {
    return template_class_name() + "." + 
      td->get_name(".", "."); // add no extension.
  }

  static const string base_class_name() {
    static string name("RenderFieldBase");
    return name;
  }

  static const string template_class_name() {
    static string name("RenderField");
    return name;
  }

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *td);

  GeomSwitch*              node_switch_;
  GeomSwitch*              edge_switch_;
  GeomSwitch*              face_switch_;
  GeomSwitch*              data_switch_;
  GeomArrows*              vec_node_;
  MaterialHandle           def_mat_handle_;
  ColorMapHandle           color_handle_;
  int                      res_;
};

template <class Fld>
class RenderField : public RenderFieldBase
{
public:
  void render_nodes(const Fld *fld, 
		    const string &node_display_type,
		    bool use_def_color,
		    double node_scale);
  void render_edges(const Fld *fld,
		    const string &edge_display_type,
		    bool use_def_color,
		    double edge_scale);
  void render_faces(const Fld *fld, bool use_def_color);

  void render_all(const Fld *fld,  
		  bool nodes, bool edges, bool faces, bool data, 
		  bool def_col, const string &ndt, const string &edt, 
		  double ns, double es, double vs);

  void render_data(const Fld *fld, 
		   const string &data_display_type,
		   double scale);

  //! virtual interface. 
  virtual void render(FieldHandle fh,  
		      bool nodes, bool edges, bool faces, bool data,
		      MaterialHandle def_mat,
		      bool def_col, ColorMapHandle color_handle,
		      const string &ndt, const string &edt, 
		      double ns, double es, double vs, int res);
    
private:
  inline void add_sphere(const Point &p, double scale, GeomGroup *g, 
			 MaterialHandle m0);
  inline void add_axis(const Point &p, double scale, GeomGroup *g, 
		       MaterialHandle m0);
  inline void add_point(const Point &p, GeomPts *g, 
			MaterialHandle m0);
  inline void add_edge(const Point &p1, const Point &p2, double scale, 
		       GeomGroup *g, MaterialHandle mh_avg,
		       bool cyl = true);
  inline void add_face(const Point &p1, const Point &p2, const Point &p3, 
		       MaterialHandle m0, MaterialHandle m1, MaterialHandle m2,
		       GeomTriangles *g);
  inline void add_face(const Point &p1, const Point &p2, const Point &p3, 
		       const Vector &v1, const Vector &v2, const Vector &v3, 
		       MaterialHandle m0, MaterialHandle m1, MaterialHandle m2,
		       GeomTriangles *g);
  
  inline  MaterialHandle choose_mat(bool def, double val) {  
    if (def) return def_mat_handle_;
    return color_handle_->lookup(val);
  }
};

template <class T>
bool
to_double(const T& tmp, double &val)
{
  return false;
}

template <>
bool
to_double(const double&, double &);

template <>
bool
to_double(const int&, double &);

template <>
bool
to_double(const short&, double &);

template <>
bool
to_double(const unsigned char&, double &);

template <class Dat>
bool 
add_data(const Point &, const Dat &, GeomArrows *, 
	 GeomSwitch *, MaterialHandle &, const string &, double)
{
  return false;
}

template <>
bool 
add_data(const Point &, const Vector &, GeomArrows *, 
	 GeomSwitch *, MaterialHandle &, const string &, double);

template <>
bool 
add_data(const Point &, const Tensor &, GeomArrows *, 
	 GeomSwitch *, MaterialHandle &, const string &, double);

template <class Fld>
void 
RenderField<Fld>::render(FieldHandle fh,  bool nodes, 
			 bool edges, bool faces, bool data,
			 MaterialHandle def_mat,
			 bool def_col, ColorMapHandle color_handle,
			 const string &ndt, const string &edt,
			 double ns, double es, double vs, int res)
{
  Fld *fld = dynamic_cast<Fld*>(fh.get_rep());
  ASSERT(fld != 0);
  def_mat_handle_ = def_mat;
  color_handle_ = color_handle;
  res_ = res;
  render_all(fld, nodes, edges, faces, data, def_col,  ndt, edt, ns, es, vs);
}


template <class Fld>
void 
RenderField<Fld>::render_data(const Fld *sfld, 
			       const string &display_type,
			       double scale) 
{
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();

  // pass: over the data
  switch (sfld->data_at()) {
  case Field::NODE:
    {
      typename Fld::mesh_type::Node::iterator niter = mesh->node_begin();  
      while (niter != mesh->node_end()) {
	typename Fld::value_type tmp;
	if (sfld->value(tmp, *niter)) {
	  Point p;
	  mesh->get_point(p, *niter);
	  add_data(p, tmp, vec_node_, data_switch_, def_mat_handle_, 
		   display_type, scale); 
	}
	++niter;
      }
    }
    break;
  case Field::EDGE:      
    {
      typename Fld::mesh_type::Edge::iterator eiter = mesh->edge_begin();  
      while (eiter != mesh->edge_end()) { 
	typename Fld::value_type tmp;
	if (sfld->value(tmp, *eiter)) {
	  Point p;
	  mesh->get_center(p, *eiter);
	  add_data(p, tmp, vec_node_, data_switch_, def_mat_handle_, 
		   display_type, scale);
	}
	++eiter;
      }
    }
    break;
  case Field::FACE:
    {
      typename Fld::mesh_type::Face::iterator fiter = mesh->face_begin();  
      while (fiter != mesh->face_end()) { 
	typename Fld::value_type tmp;
	if (sfld->value(tmp, *fiter)) {
	  Point p;
	  mesh->get_center(p, *fiter);
	  add_data(p, tmp, vec_node_, data_switch_, def_mat_handle_, 
		   display_type, scale);
	}
	++fiter;
      }
    }
    break;
  case Field::CELL:
    {
      typename Fld::mesh_type::Cell::iterator citer = mesh->cell_begin();  
      while (citer != mesh->cell_end()) { 
	typename Fld::value_type tmp;
	if (sfld->value(tmp, *citer)) {
	  Point p;
	  mesh->get_center(p, *citer);
	  add_data(p, tmp, vec_node_, data_switch_, def_mat_handle_, 
		   display_type, scale);
	}
	++citer;
      }
    }
    break;
  case Field::NONE:
    break;
  }
}

template <class Fld>
void 
RenderField<Fld>::render_nodes(const Fld *sfld, 
			       const string &node_display_type,
			       bool use_def_color,
			       double node_scale) 
{
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
  GeomGroup* nodes = scinew GeomGroup;
  node_switch_ = scinew GeomSwitch(nodes);
  GeomPts *pts = 0;

  if (node_display_type == "Points") {
    pts = scinew GeomPts(mesh->nodes_size());
  }
  // First pass: over the nodes
  typename Fld::mesh_type::Node::iterator niter = mesh->node_begin();  
  while (niter != mesh->node_end()) {
    // Use a default color?
    bool def_color = (use_def_color || (color_handle_.get_rep() == 0));
    
    Point p;
    mesh->get_point(p, *niter);

    // val is double because the color index field must be scalar.
    double val = 0.L;
 
    switch (sfld->data_at()) {
    case Field::NODE:
      {
	typename Fld::value_type tmp;
	if (! (sfld->value(tmp, *niter) && (to_double(tmp, val)))) { 
	  def_color = true; 
	}
      }
      break;
    case Field::EDGE:
    case Field::FACE:
    case Field::CELL:
    case Field::NONE:
      def_color = true;
      break;
    }

    if (node_display_type == "Spheres") {
      add_sphere(p, node_scale, nodes, 
		 choose_mat(def_color, val));
    } else if (node_display_type == "Axes") {
      add_axis(p, node_scale, nodes, choose_mat(def_color, val));
    } else {
      add_point(p, pts, choose_mat(def_color, val));
    }
    ++niter;
  }
  if (node_display_type == "Points") {
    nodes->add(pts);
  }
}


template <class Fld>
void 
RenderField<Fld>::render_edges(const Fld *sfld,
			       const string &edge_display_type,
			       bool use_def_color, 
			       double edge_scale) 
{
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
  GeomGroup* edges = scinew GeomGroup;
  edge_switch_ = scinew GeomSwitch(edges);
  // Second pass: over the edges
  typename Fld::mesh_type::Edge::iterator eiter = mesh->edge_begin();  
  while (eiter != mesh->edge_end()) {  
    // Use a default color?
    bool def_color = (use_def_color || (color_handle_.get_rep() == 0));

    typename Fld::mesh_type::Node::array_type nodes;
    mesh->get_nodes(nodes, *eiter); ++eiter;
    
      
    Point p1, p2;
    mesh->get_point(p1, nodes[0]);
    mesh->get_point(p2, nodes[1]);
    double val1 = 0.L;
    double val2 = 0.L;
    double val_avg = 0.L;
    switch (sfld->data_at()) {
    case Field::NODE:
      {
	typename Fld::value_type tmp1;
	typename Fld::value_type tmp2;
	if (! (sfld->value(tmp1, nodes[0]) && to_double(tmp1, val1) &&
	       sfld->value(tmp2, nodes[1]) && to_double(tmp2, val2))) { 
	  def_color = true; 
	} else {
	  val_avg = (val1+val2)/2.;
	}
      }
      break;
    case Field::EDGE:
      {
	typename Fld::value_type tmp;
	if (! (sfld->value(tmp, *eiter) && to_double(tmp, val_avg))) { 
	  def_color = true; 
	}
      }
      break;
    case Field::FACE:
    case Field::CELL:
    case Field::NONE:
      def_color = true;
      break;
    }
    bool cyl = false;
    if (edge_display_type == "Cylinders") { cyl = true; }
    add_edge(p1, p2, edge_scale, edges, 
	     choose_mat(def_color, val_avg), cyl);
  }
}

template <class Fld>
void 
RenderField<Fld>::render_faces(const Fld *sfld,
			       bool use_def_color)
{
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
  const bool with_normals = mesh->has_normals();

  GeomTriangles* faces = scinew GeomTriangles;
  face_switch_ = scinew GeomSwitch(faces);
  // Third pass: over the faces
  typename Fld::mesh_type::Face::iterator fiter = mesh->face_begin();  
  typename Fld::mesh_type::Node::array_type nodes;

  while (fiter != mesh->face_end()) {  
    // Use a default color?
    bool def_color = (use_def_color || (color_handle_.get_rep() == 0));

    mesh->get_nodes(nodes, *fiter); 
    ++fiter;     
 
    unsigned int i;
    vector<Point> points(nodes.size());
    vector<Vector> normals(nodes.size());
    vector<double> vals(nodes.size(), 0.0);
    for (i = 0; i < nodes.size(); i++)
    {
      mesh->get_point(points[i], nodes[i]);
    }

    if (with_normals) {
      for (i = 0; i < nodes.size(); i++)
      {
	mesh->get_normal(normals[i], nodes[i]);
      }
    }

    switch (sfld->data_at()) {
    case Field::NODE:
      {
	typename Fld::value_type tmp;
	for (i=0; i < nodes.size(); i++)
	{
	  if (! (sfld->value(tmp, nodes[i]) && to_double(tmp, vals[i])))
	  {
	    def_color = true;
	    break;
	  }
	}
      }
      break;

    case Field::FACE: 
      {
	typename Fld::value_type tmp;
	if (! sfld->value(tmp, *fiter))
	{
	  def_color = true;
	}
	for (i = 0; i < nodes.size(); i++)
	{
	  if (! to_double(tmp, vals[i]))
	  {
	    def_color = true;
	    break;
	  }
	}
      }
      break;

    case Field::EDGE:
    case Field::CELL:
    case Field::NONE:
      def_color = true;
      break;
    }

    for (i=2; i<nodes.size(); i++)
    {
      if (with_normals) {
	
      add_face(points[0], points[i-1], points[i],
	       normals[0], normals[i-1], normals[i], 
	       choose_mat(def_color, vals[0]), 
	       choose_mat(def_color, vals[i-1]), 
	       choose_mat(def_color, vals[i]), 
	       faces);
      } else {
	add_face(points[0], points[i-1], points[i], 
		 choose_mat(def_color, vals[0]), 
		 choose_mat(def_color, vals[i-1]), 
		 choose_mat(def_color, vals[i]), 
		 faces);
      }
    }
  }
}

template <class Fld>
void
RenderField<Fld>::render_all(const Fld *fld, bool nodes, 
			     bool edges, bool faces, bool data,
			     bool def_col,
			     const string &ndt, const string &edt,
			     double ns, double es, double vs)
{
  if (nodes) render_nodes(fld, ndt, def_col, ns);
  if (edges) render_edges(fld, edt, def_col, es);
  if (faces) render_faces(fld, def_col);
  
  if (data) {
    vec_node_ = scinew GeomArrows(vs);
    data_switch_ = scinew GeomSwitch(vec_node_);
    render_data(fld, ndt, vs);
  }
}

template <class Fld>
void 
RenderField<Fld>::add_face(const Point &p0, const Point &p1, const Point &p2, 
		    const Vector &n0, const Vector &n1, const Vector &n2,
		    MaterialHandle m0, MaterialHandle m1, MaterialHandle m2,
		    GeomTriangles *g) 
{
  g->add(p0, n0, m0, 
	 p1, n1, m1, 
	 p2, n2, m2);
}
template <class Fld>
void 
RenderField<Fld>::add_face(const Point &p0, const Point &p1, const Point &p2, 
		    MaterialHandle m0, MaterialHandle m1, MaterialHandle m2,
		    GeomTriangles *g) 
{
  g->add(p0, m0, 
	 p1, m1, 
	 p2, m2);
}

template <class Fld>
void 
RenderField<Fld>::add_edge(const Point &p0, const Point &p1,  
		    double scale, GeomGroup *g, MaterialHandle mh_avg,
		    bool cyl) 
{
  if (cyl) {
    GeomCylinder *c = new GeomCylinder(p0, p1, scale, 2*res_);
    g->add(scinew GeomMaterial(c, mh_avg));
  } else {
    GeomLine *l = new GeomLine(p0, p1);
    l->setLineWidth(scale);
    g->add(scinew GeomMaterial(l, mh_avg));
  }
}

template <class Fld>
void 
RenderField<Fld>::add_sphere(const Point &p0, double scale, 
		      GeomGroup *g, MaterialHandle mh) {
  GeomSphere *s = scinew GeomSphere(p0, scale, res_, res_);
  g->add(scinew GeomMaterial(s, mh));
}

template <class Fld>
void 
RenderField<Fld>::add_axis(const Point &p0, double scale, 
		    GeomGroup *g, MaterialHandle mh) 
{
  static const Vector x(1., 0., 0.);
  static const Vector y(0., 1., 0.);
  static const Vector z(0., 0., 1.);

  Point p1 = p0 + x * scale;
  Point p2 = p0 - x * scale;
  GeomLine *l = new GeomLine(p1, p2);
  l->setLineWidth(3.0);
  g->add(scinew GeomMaterial(l, mh));
  p1 = p0 + y * scale;
  p2 = p0 - y * scale;
  l = new GeomLine(p1, p2);
  l->setLineWidth(3.0);
  g->add(scinew GeomMaterial(l, mh));
  p1 = p0 + z * scale;
  p2 = p0 - z * scale;
  l = new GeomLine(p1, p2);
  l->setLineWidth(3.0);
  g->add(scinew GeomMaterial(l, mh));
}

template <class Fld>
void 
RenderField<Fld>::add_point(const Point &p, GeomPts *pts, MaterialHandle mh) {
  pts->add(p, mh->diffuse);
}

} // end namespace SCIRun

#endif // Visualization_RenderField_h
