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
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <sci_hash_map.h>
#include <Core/Datatypes/TetVolMesh.h>

namespace SCIRun {

class GeomEllipsoid;
class GeomArrows;

//! RenderFieldBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! RenderFieldBase from the DynamicAlgoBase they will have a pointer to.
class RenderFieldBase : public DynamicAlgoBase
{
public:
#ifdef HAVE_HASH_MAP
  typedef hash_map<int, MaterialHandle> ind_mat_t;
#else
  typedef map<int, MaterialHandle> ind_mat_t;
#endif
  virtual void set_mat_map(ind_mat_t *mm) = 0;
  virtual void render(FieldHandle f, bool nodes, bool edges, 
		      bool faces, bool data, MaterialHandle def_mat, 
		      bool def_col, ColorMapHandle color_handle,
		      const string &ndt, const string &edt, 
		      double ns, double es, double vs, bool normalize, 
		      int res, bool use_normals, bool use_transparency) = 0;

  RenderFieldBase();
  virtual ~RenderFieldBase();

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *ftd,
				       const TypeDescription *ltd);

  GeomSwitch*              node_switch_;
  GeomSwitch*              edge_switch_;
  GeomSwitch*              face_switch_;
  GeomSwitch*              data_switch_;
  GeomArrows*              vec_node_;
  MaterialHandle           def_mat_handle_;
  ColorMapHandle           color_handle_;
  int                      res_;
  ind_mat_t               *mats_;
};

template <class Fld, class Loc>
class RenderField : public RenderFieldBase
{
public:
  void set_mat_map(ind_mat_t *mm) { mats_ = mm; }
  void render_nodes(const Fld *fld, 
		    const string &node_display_type,
		    double node_scale);
  void render_edges(const Fld *fld,
		    const string &edge_display_type,
		    double edge_scale);
  void render_faces(const Fld *fld, 
		    bool use_normals,
		    bool use_transparency);

  void render_all(const Fld *fld,  
		  bool nodes, bool edges, bool faces, bool data, 
		  bool data_at, const string &ndt, const string &edt, 
		  double ns, double es, double vs, bool normalize,
		  bool use_normals, bool use_transparency);

  void render_data(const Fld *fld, 
		   const string &data_display_type,
		   double scale, bool normalize);

  void render_materials(const Fld *fld, const string &data_display_type);

  //! virtual interface. 
  virtual void render(FieldHandle fh,  
		      bool nodes, bool edges, bool faces, bool data,
		      MaterialHandle def_mat,
		      bool data_at, ColorMapHandle color_handle,
		      const string &ndt, const string &edt, 
		      double ns, double es, double vs, bool normalize, 
		      int res, bool use_normals, bool use_transparency);
    
private:
  inline void add_sphere(const Point &p, double scale, GeomGroup *g, 
			 MaterialHandle m0);
  inline void add_disk(const Point &p, const Vector& v, double scale, 
		       GeomGroup *g, MaterialHandle m0);
  inline void add_axis(const Point &p, double scale, GeomGroup *g, 
		       MaterialHandle m0);
  inline void add_point(const Point &p, GeomPts *g, 
			MaterialHandle m0);
  inline void add_edge(const Point &p1, const Point &p2, double scale, 
		       GeomGroup *g, MaterialHandle mh_avg,
		       bool cyl = true);
  
  inline  MaterialHandle choose_mat(bool def, int idx) {  
    if (def) return def_mat_handle_;
    ASSERT(mats_ != 0); 
    return (*mats_)[idx];
  }
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

template <>
bool
to_double(const Vector&, double &);

template <class Dat>
bool 
add_data(const Point &, const Dat &, GeomArrows *, 
	 GeomSwitch *, MaterialHandle &, const string &, double, bool)
{
  return false;
}

template <>
bool 
add_data(const Point &, const Vector &, GeomArrows *, 
	 GeomSwitch *, MaterialHandle &, const string &, double, bool);

template <>
bool 
add_data(const Point &, const Tensor &, GeomArrows *, 
	 GeomSwitch *, MaterialHandle &, const string &, double, bool);

template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::render(FieldHandle fh,  bool nodes, 
			      bool edges, bool faces, bool data,
			      MaterialHandle def_mat,
			      bool def_col, ColorMapHandle color_handle,
			      const string &ndt, const string &edt,
			      double ns, double es, double vs, bool normalize, 
			      int res, bool use_normals, bool use_transparency)
{
  Fld *fld = dynamic_cast<Fld*>(fh.get_rep());
  ASSERT(fld != 0);
  def_mat_handle_ = def_mat;
  color_handle_ = color_handle;
  res_ = res;
  render_all(fld, nodes, edges, faces, data, def_col,  ndt, edt, ns, es, vs, 
	     normalize, use_normals, use_transparency);
}


template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::render_data(const Fld *fld,
				   const string &display_type,
				   double scale, 
				   bool normalize)
{
  //cerr << "rendering data_at" << endl;
  double val = 0.0L;
  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  typename Loc::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  while (iter != end) {
    typename Fld::value_type tmp;
    if (fld->value(tmp, *iter)) {
      Point p;
      mesh->get_center(p, *iter);
      to_double(tmp, val);
      MaterialHandle m = choose_mat(false, *iter);
      add_data(p, tmp, vec_node_, data_switch_, 
	       m, display_type, scale, normalize); 
    }
    ++iter;
  }
}


// if a colormap exists, and we have data at a location, map the index to 
// a materialhandle to be used by all the render passes.
template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::render_materials(const Fld *sfld, 
					const string &node_display_type) 
{
  //cerr << "rendering materials" << endl;

  const string dat_mat("data_at_materials");
#ifdef HAVE_HASH_MAP
  typedef hash_map<int, MaterialHandle> ind_mat_t;
#else
  typedef map<int, MaterialHandle> ind_mat_t;
#endif
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();

  if (! mats_) {
    ASSERTFAIL("must call set_mat_map first");
  }

  bool def_color = false;
  // val is double because the color index field must be scalar.
  double val = 0.L;
  Vector vec(0,0,0);
  switch (sfld->data_at()) {
  case Field::NODE:
    {
      typename Fld::mesh_type::Node::iterator niter;  
      mesh->begin(niter);  
      typename Fld::mesh_type::Node::iterator niter_end;  
      mesh->end(niter_end);
      
      while (niter != niter_end) {
	typename Fld::value_type tmp;

	if (node_display_type == "Disks") {
	  if (sfld->value(tmp, *niter) && (to_vector(tmp, vec))) { 
	    val = vec.length();
	  } else {
	    def_color = true; 
	  }
	} else {
	  if (!(sfld->value(tmp, *niter) && (to_double(tmp, val)))) { 
	    def_color = true; 
	  }
	}
	
	MaterialHandle mat;
	if (color_handle_.get_rep() == 0) def_color = true;
	if (def_color) mat = def_mat_handle_;
	else mat = color_handle_->lookup(val);

	int nidx = *niter;
	ind_mat_t::iterator iter = mats_->find(nidx);
	if (iter != mats_->end()) {
	  // we have stored a color before.
          MaterialHandle &existing = (*mats_)[nidx];
	  //actually change the underlying object for all who point to it.
          *(existing.get_rep()) = *(mat.get_rep());
        } else {
	  mat.detach();
	  (*mats_)[nidx] = mat;
        }
	++niter;  
      }
    }
    break;

  case Field::EDGE:
    {
      mesh->synchronize(Mesh::EDGES_E);
      typename Fld::mesh_type::Edge::iterator eiter;  
      mesh->begin(eiter);  
      typename Fld::mesh_type::Edge::iterator eiter_end;  
      mesh->end(eiter_end);
      
      while (eiter != eiter_end) {
	typename Fld::value_type tmp;
	
	if (!(sfld->value(tmp, *eiter) && (to_double(tmp, val)))) { 
	  def_color = true; 
	}
	
	MaterialHandle mat;
	if (color_handle_.get_rep() == 0) def_color = true;
	if (def_color) mat = def_mat_handle_;
	else mat = color_handle_->lookup(val);

	int eidx = *eiter;
	ind_mat_t::iterator iter = mats_->find(eidx);
	if (iter != mats_->end()) {
	  // we have stored a color before.
	  MaterialHandle &existing = (*mats_)[eidx];
	  //actually change the underlying object for all who point to it.
	  *(existing.get_rep()) = *(mat.get_rep());
	} else {
	  mat.detach();
	  (*mats_)[eidx] = mat;
	}
	++eiter;  
      }
    }
    break;

  case Field::FACE:
    {
      typename Fld::mesh_type::Face::iterator fiter;  
      mesh->begin(fiter);  
      typename Fld::mesh_type::Face::iterator fiter_end;  
      mesh->end(fiter_end);

      while (fiter != fiter_end) {
	typename Fld::value_type tmp;
	
	if (!(sfld->value(tmp, *fiter) && (to_double(tmp, val)))) { 
	  def_color = true; 
	}
	
	MaterialHandle mat;
	if (color_handle_.get_rep() == 0) def_color = true;
	if (def_color) mat = def_mat_handle_;
	else mat = color_handle_->lookup(val);
	
	int fidx = *fiter;
	ind_mat_t::iterator iter = mats_->find(fidx);
	if (iter != mats_->end()) {
	  // we have stored a color before.
	  MaterialHandle &existing = (*mats_)[fidx];
	  //actually change the underlying object for all who point to it.
	  *(existing.get_rep()) = *(mat.get_rep());
	} else {
	  mat.detach();
	  (*mats_)[fidx] = mat;
	}
	++fiter;  
      }
    }
    break;

  case Field::CELL:
    {
      typename Fld::mesh_type::Cell::iterator citer;  
      mesh->begin(citer);  
      typename Fld::mesh_type::Cell::iterator citer_end;  
      mesh->end(citer_end);
      
      while (citer != citer_end) {
	typename Fld::value_type tmp;
	
	if (!(sfld->value(tmp, *citer) && (to_double(tmp, val)))) { 
	  def_color = true; 
	}
	
	MaterialHandle mat;
	if (color_handle_.get_rep() == 0) def_color = true;
	if (def_color) mat = def_mat_handle_;
	else mat = color_handle_->lookup(val);
	
	int cidx = *citer;
	ind_mat_t::iterator iter = mats_->find(cidx);
	if (iter != mats_->end()) {
	  // we have stored a color before.
	  MaterialHandle &existing = (*mats_)[cidx];
	  //actually change the underlying object for all who point to it.
	  *(existing.get_rep()) = *(mat.get_rep());
	} else {
	  mat.detach();
	  (*mats_)[cidx] = mat;
	}
	++citer;  
      }
    }
    break;

  case Field::NONE:
  default:
    cerr << "Unknown data location." << endl;
    break;
  }
}


template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::render_nodes(const Fld *sfld, 
				    const string &node_display_type,
				    double node_scale) 
{
  //cerr << "rendering nodes" << endl;
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
  GeomGroup* nodes = scinew GeomGroup;
  node_switch_ = scinew GeomSwitch(nodes);
  GeomPts *pts = 0;

  if (node_display_type == "Points") {
    typename Fld::mesh_type::Node::size_type nsize;
    mesh->size(nsize);
    pts = scinew GeomPts((unsigned int)(nsize));
  }
  // First pass: over the nodes
  mesh->synchronize(Mesh::NODES_E);
  typename Fld::mesh_type::Node::iterator niter;  mesh->begin(niter);  
  typename Fld::mesh_type::Node::iterator niter_end;  mesh->end(niter_end);  
  while (niter != niter_end) {
    // Use a default color?
    bool def_color = false;
    
    Point p;
    mesh->get_point(p, *niter);

    // val is double because the color index field must be scalar.
    Vector vec(0,0,0);
    switch (sfld->data_at()) {
    case Field::NODE:
      {
	typename Fld::value_type tmp;
	// color was selected in the render_materials pass.
	if (node_display_type == "Disks") {
	  if (sfld->value(tmp, *niter) && (to_vector(tmp, vec))) {
	  }
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
		 choose_mat(def_color, *niter));
    } else if (node_display_type == "Axes") {
      add_axis(p, node_scale, nodes, choose_mat(def_color, *niter));
    } else if (node_display_type == "Disks") {
      add_disk(p, vec, node_scale, nodes, choose_mat(def_color, *niter));
    } else {
      add_point(p, pts, choose_mat(def_color, *niter));
    }
    ++niter;
  }
  if (node_display_type == "Points") {
    nodes->add(pts);
  }
}


template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::render_edges(const Fld *sfld,
				    const string &edge_display_type,
				    double edge_scale) 
{
  //cerr << "rendering edges" << endl;
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
  GeomGroup* edges = scinew GeomGroup;
  edge_switch_ = scinew GeomSwitch(edges);
  // Second pass: over the edges
  mesh->synchronize(Mesh::EDGES_E);
  typename Fld::mesh_type::Edge::iterator eiter; mesh->begin(eiter);  
  typename Fld::mesh_type::Edge::iterator eiter_end; mesh->end(eiter_end);  
  while (eiter != eiter_end) {  
    typename Fld::mesh_type::Node::array_type nodes;
    mesh->get_nodes(nodes, *eiter);
      
    Point p1, p2;
    mesh->get_point(p1, nodes[0]);
    mesh->get_point(p2, nodes[1]);
    bool cyl = false;
    if (edge_display_type == "Cylinders") { cyl = true; }
    switch (sfld->data_at()) {
    case Field::NODE:
      {
	// does not average anymore, so that switching color maps is fast.
	MaterialHandle m1 = choose_mat(false, nodes[0]);
	add_edge(p1, p2, edge_scale, edges, m1, cyl);
      }
      break;
    case Field::EDGE:
      {
	add_edge(p1, p2, edge_scale, edges,
		 choose_mat(false, *eiter), cyl);
      }
      break;
    case Field::FACE:
    case Field::CELL:
    case Field::NONE:
      add_edge(p1, p2, edge_scale, edges, choose_mat(true, 0), cyl);
      break;
    }
    
    ++eiter;
  }
}

template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::render_faces(const Fld *sfld,
				    bool use_normals,
				    bool use_transparency)
{
  //cerr << "rendering faces" << endl;
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
  const bool with_normals = (use_normals && mesh->has_normals());

  GeomTriangles* faces;
  if (use_transparency)
  {
    faces = scinew GeomTranspTriangles;
  }
  else
  {
    faces = scinew GeomTriangles;
  }

  face_switch_ = scinew GeomSwitch(faces);
  // Third pass: over the faces
  mesh->synchronize(Mesh::FACES_E);
  typename Fld::mesh_type::Face::iterator fiter; mesh->begin(fiter);  
  typename Fld::mesh_type::Face::iterator fiter_end; mesh->end(fiter_end);  
  typename Fld::mesh_type::Node::array_type nodes;

  while (fiter != fiter_end) {
    mesh->get_nodes(nodes, *fiter); 
 
    unsigned int i;
    vector<Point> points(nodes.size());
    vector<Vector> normals(nodes.size());
    vector<MaterialHandle> mats(nodes.size());
    for (i = 0; i < nodes.size(); i++)
    {
      mesh->get_point(points[i], nodes[i]);
    }

    if (with_normals) {
      for (i = 0; i < nodes.size(); i++) {
	mesh->get_normal(normals[i], nodes[i]);
      }
    }

    switch (sfld->data_at()) {
    case Field::NODE:
      {
	for (i=0; i < nodes.size(); i++) {
	  mats[i] = choose_mat(false, nodes[i]);
	}
      }
      break;
      
    case Field::FACE: 
      {
	MaterialHandle m = choose_mat(false, *fiter);
	for (i = 0; i < nodes.size(); i++) {
	  mats[i] = m;
	}
      }
      break;
      
    case Field::EDGE:
    case Field::CELL:
    case Field::NONE:
      {
	MaterialHandle m = choose_mat(true, 0);
	for (i = 0; i < nodes.size(); i++) {
	  mats[i] = m;
	}
      }
      break;
    }
    
    for (i=2; i<nodes.size(); i++) {
      if (with_normals) {
	faces->add(points[0], normals[0], mats[0],
		   points[i-1], normals[i-1], mats[i-1],
		   points[i], normals[i], mats[i]);
      } else {
	faces->add(points[0], mats[0],
		   points[i-1], mats[i-1],
		   points[i], mats[i]);
      }
    }
    ++fiter;     
  }
}

template <class Fld, class Loc>
void
RenderField<Fld, Loc>::render_all(const Fld *fld, bool nodes, 
				  bool edges, bool faces, bool data,
				  bool data_at,
				  const string &ndt, const string &edt,
				  double ns, double es, double vs, bool normalize,
				  bool use_normals, bool use_transparency)
{
  if (data_at) render_materials(fld, ndt);
  if (nodes) render_nodes(fld, ndt, ns);
  if (edges) render_edges(fld, edt, es);
  if (faces) render_faces(fld, use_normals, use_transparency);
  
  if (data) {
    vec_node_ = scinew GeomArrows(0.15, 0.6);
    data_switch_ = scinew GeomSwitch(vec_node_);
    render_data(fld, ndt, vs, normalize);
  }
}


template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::add_edge(const Point &p0, const Point &p1,  
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

template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::add_sphere(const Point &p0, double scale, 
				  GeomGroup *g, MaterialHandle mh) {
  GeomSphere *s = scinew GeomSphere(p0, scale, res_, res_);
  g->add(scinew GeomMaterial(s, mh));
}

template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::add_disk(const Point &p, const Vector &vin, double scale, 
				GeomGroup *g, MaterialHandle mh) {

  Vector v = vin;
  if (v.length() > 0.00001) {
    v.safe_normalize();
    v*=scale/6;
    GeomCappedCylinder *d = scinew GeomCappedCylinder(p + v, p - v, scale, 
						      res_, 1, 1);
    g->add(scinew GeomMaterial(d, mh));
  } else {
    GeomSphere *s = scinew GeomSphere(p, scale, res_, res_);
    g->add(scinew GeomMaterial(s, mh));
  }
}

template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::add_axis(const Point &p0, double scale, 
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

template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::add_point(const Point &p, GeomPts *pts, MaterialHandle mh) {
  pts->add(p, mh);
}

} // end namespace SCIRun

#endif // Visualization_RenderField_h
