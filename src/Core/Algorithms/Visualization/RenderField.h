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
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomArrows.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomDL.h>
#include <Core/Geom/Pt.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Containers/StringUtil.h>
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

  void set_mat_map(ind_mat_t *mm) { mats_ = mm; }

  virtual void render(FieldHandle f, bool nodes, bool edges, 
		      bool faces, bool data, MaterialHandle def_mat, 
		      bool def_col, ColorMapHandle color_handle,
		      const string &ndt, const string &edt, 
		      double ns, double es, double vs, bool normalize, 
		      int sphere_resolution, int cylinder_resolution,
		      bool use_normals, bool use_transparency,
		      bool bidirectional, bool arrow_heads) = 0;

  virtual GeomSwitch *render_text(FieldHandle fld,
				  bool use_default_material,
				  MaterialHandle default_material,
				  bool backface_cull_p,
				  int  fontsize,
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
  GeomHandle               data_switch_;

protected:
  MaterialHandle           def_mat_handle_;
  ColorMapHandle           color_handle_;
  ind_mat_t               *mats_;
};


template <class Fld, class Loc>
class RenderField : public RenderFieldBase
{
public:
  //! virtual interface. 
  virtual void render(FieldHandle fh,  
		      bool nodes, bool edges, bool faces, bool data,
		      MaterialHandle def_mat,
		      bool data_at, ColorMapHandle color_handle,
		      const string &ndt, const string &edt, 
		      double ns, double es, double vs, bool normalize, 
		      int sphere_resolution, int cylinder_resolution,
		      bool use_normals, bool use_transparency,
		      bool bidirectional, bool arrow_heads);

  virtual GeomSwitch *render_text(FieldHandle fld,
				  bool use_default_material,
				  MaterialHandle default_material,
				  bool backface_cull_p,
				  int  fontsize,
				  bool render_data,
				  bool render_nodes,
				  bool render_edges,
				  bool render_faces,
				  bool render_cells);

private:
  GeomSwitch *render_nodes(const Fld *fld, 
			   const string &node_display_mode,
			   double node_scale,
			   int node_resolution);
  GeomSwitch *render_edges(const Fld *fld,
			   const string &edge_display_mode,
			   double edge_scale,
			   int cylinder_resolution);
  GeomSwitch *render_faces(const Fld *fld, 
			   bool use_normals,
			   bool use_transparency);

  GeomSwitch *render_text_data(FieldHandle fld,
			       bool use_default_material,
			       MaterialHandle default_material,
			       bool backface_cull_p,
			       int fontsize);
  GeomSwitch *render_text_data_nodes(FieldHandle fld,
				     bool use_default_material,
				     MaterialHandle default_material,
				     bool backface_cull_p,
				     int fontsize);
  GeomSwitch *render_text_nodes(FieldHandle fld,
  				bool use_default_material,
  				MaterialHandle default_material,
				bool backface_cull_p,
				int fontsize);
  GeomSwitch *render_text_edges(FieldHandle fld,
  				bool use_default_material,
  				MaterialHandle default_material,
				int fontsize);
  GeomSwitch *render_text_faces(FieldHandle fld,
  				bool use_default_material,
  				MaterialHandle default_material,
				int fontsize);
  GeomSwitch *render_text_cells(FieldHandle fld,
				bool use_default_material,
				MaterialHandle default_material,
				int fontsize);

  GeomSwitch *render_data(const Fld *fld, 
			  const string &data_display_mode,
			  double scale, bool normalize, bool bidirectional,
			  bool arrow_heads);

  void render_materials(const Fld *fld, const string &data_display_mode);

  inline void add_sphere(const Point &p, double scale, 
			 int resolution, GeomGroup *g, 
			 MaterialHandle m0);
  inline void add_disk(const Point &p, const Vector& v, double scale, 
		       int resolution, GeomGroup *g, MaterialHandle m0);
  inline void add_axis(const Point &p, double scale, GeomGroup *g, 
		       MaterialHandle m0);

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
RenderField<Fld, Loc>::add_sphere(const Point &p0, double scale,
				  int resolution,
				  GeomGroup *g, MaterialHandle mh)
{
  GeomSphere *s = scinew GeomSphere(p0, scale, resolution, resolution);
  g->add(scinew GeomMaterial(s, mh));
}



template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::add_disk(const Point &p, const Vector &vin,
				double scale, int resolution,
				GeomGroup *g, MaterialHandle mh)
{
  Vector v = vin;
  if (v.length2() * scale > 1.0e-10)
  {
    v.safe_normalize();
    v*=scale/6;
    GeomCappedCylinder *d = scinew GeomCappedCylinder(p + v, p - v, scale, 
						      resolution, 1, 1);
    g->add(scinew GeomMaterial(d, mh));
  }
  else
  {
    GeomSphere *s = scinew GeomSphere(p, scale, resolution, resolution);
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
RenderField<Fld, Loc>::render(FieldHandle fh,  bool nodes, 
			      bool edges, bool faces, bool data,
			      MaterialHandle def_mat,
			      bool def_col, ColorMapHandle color_handle,
			      const string &ndt, const string &edt,
			      double ns, double es, double vs, bool normalize, 
			      int sphere_res, int cyl_res,
			      bool use_normals, bool use_transp,
			      bool bidirectional, bool arrow_heads)
{
  Fld *fld = dynamic_cast<Fld*>(fh.get_rep());
  ASSERT(fld != 0);
  def_mat_handle_ = def_mat;
  color_handle_ = color_handle;

  if (def_col) { render_materials(fld, ndt); }
  if (nodes) { node_switch_ = render_nodes(fld, ndt, ns, sphere_res); }
  if (edges) { edge_switch_ = render_edges(fld, edt, es, cyl_res); }
  if (faces) { face_switch_ = render_faces(fld, use_normals, use_transp); }
  if (data)
  {
    data_switch_ = render_data(fld, ndt, vs, normalize, bidirectional,
			       arrow_heads);
  }
}



template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_data(const Fld *fld,
				   const string &display_mode,
				   double scale, 
				   bool normalize,
				   bool bidirectional,
				   bool arrow_heads)
{
  GeomArrows *vec_node;
  if (arrow_heads)
  {
    vec_node = scinew GeomArrows(0.15, 0.6);
  }
  else
  {
    vec_node = scinew GeomArrows(0, 0.6);
  }
  GeomSwitch *data_switch = scinew GeomSwitch(vec_node);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  typename Loc::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  while (iter != end) {
    typename Fld::value_type tmp;
    if (fld->value(tmp, *iter)) {
      Point p;
      mesh->get_center(p, *iter);
      //to_double(tmp, val);
      MaterialHandle m = choose_mat(false, *iter);
      add_data(p, tmp, vec_node,
	       m, display_mode, scale, normalize, bidirectional); 
    }
    ++iter;
  }
  return data_switch;
}



// if a colormap exists, and we have data at a location, map the index to 
// a materialhandle to be used by all the render passes.
template <class Fld, class Loc>
void 
RenderField<Fld, Loc>::render_materials(const Fld *sfld, 
					const string &node_display_mode) 
{
  //cerr << "rendering materials" << endl;

  const string dat_mat("data_at_materials");
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
      const bool disks_p = node_display_mode == "Disks";

      while (niter != niter_end) {
	typename Fld::value_type tmp;

	if (disks_p) {
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

  // Update these when the data changes. 
  if (node_switch_.get_rep()) { node_switch_->reset_bbox(); }
  if (edge_switch_.get_rep()) { edge_switch_->reset_bbox(); }
  if (face_switch_.get_rep()) { face_switch_->reset_bbox(); }
  if (data_switch_.get_rep()) { data_switch_->reset_bbox(); }
}



template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_nodes(const Fld *sfld, 
				    const string &node_display_mode,
				    double node_scale,
				    int node_resolution) 
{
  //cerr << "rendering nodes" << endl;
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
  GeomGroup* nodes = scinew GeomGroup;
  GeomDL *display_list = scinew GeomDL(nodes);
  GeomSwitch *node_switch = scinew GeomSwitch(display_list);
  GeomPts *pts = 0;

  // 0 Points 1 Spheres 2 Axes 3 Disks
  int mode = 0;
  if (node_display_mode == "Points")       { mode = 0; }
  else if (node_display_mode == "Spheres") { mode = 1; }
  else if (node_display_mode == "Axes")    { mode = 2; }
  else if (node_display_mode == "Disks")   { mode = 3; }

  if (mode == 0) { // Points
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

    switch (sfld->data_at())
    {
    case Field::NODE:
      {
	typename Fld::value_type tmp;
	// color was selected in the render_materials pass.
	if (mode == 3 && sfld->value(tmp, *niter) && (to_vector(tmp, vec))) {}
      }
      break;

    case Field::EDGE:
    case Field::FACE:
    case Field::CELL:
    case Field::NONE:
      def_color = true;
      break;
    }

    switch (mode)
    {
    case 0: // Points
      pts->add(p, choose_mat(def_color, *niter));
      break;

    case 1: // Spheres
      add_sphere(p, node_scale, node_resolution,
		 nodes, choose_mat(def_color, *niter));
      break;

    case 2: // Axes
      add_axis(p, node_scale, nodes, choose_mat(def_color, *niter));
      break;

    case 3: // Disks
    default:
      add_disk(p, vec, node_scale, node_resolution,
	       nodes, choose_mat(def_color, *niter));
      break;
    }
    ++niter;
  }
  if (mode == 0) { // Points
    nodes->add(pts);
  }

  return node_switch;
}



template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_edges(const Fld *sfld,
				    const string &edge_display_mode,
				    double edge_scale,
				    int cylinder_resolution) 
{
  //cerr << "rendering edgess" << endl;
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();

  const bool cyl = edge_display_mode == "Cylinders";

  GeomCLines* lines = NULL;
  GeomColoredCylinders* cylinders = NULL;
  GeomSwitch *edge_switch;
  if (cyl)
  {
    cylinders = scinew GeomColoredCylinders;
    cylinders->set_radius(edge_scale);
    cylinders->set_nu_nv(cylinder_resolution, 1);
    GeomDL *display_list = scinew GeomDL(cylinders);
    edge_switch = scinew GeomSwitch(display_list);
  }
  else
  {
    lines= scinew GeomCLines;
    lines->setLineWidth(edge_scale);
    GeomDL *display_list = scinew GeomDL(lines);
    edge_switch = scinew GeomSwitch(display_list);
  }

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
    switch (sfld->data_at()) {
    case Field::NODE:
      {
	MaterialHandle m1 = choose_mat(false, nodes[0]);
	MaterialHandle m2 = choose_mat(false, nodes[1]);
	if (cyl)
	{
	  cylinders->add(p1, m1, p2, m2);
	}
	else
	{
	  lines->add(p1, m1, p2, m2);
	}
      }
      break;
    case Field::EDGE:
      {
	MaterialHandle m1 = choose_mat(false, *eiter);
	if (cyl)
	{
	  cylinders->add(p1, m1, p2, m1);
	}
	else
	{
	  lines->add(p1, m1, p2, m1);
	}
      }
      break;
    case Field::FACE:
    case Field::CELL:
    case Field::NONE:
      {
	MaterialHandle m1 = choose_mat(true, 0);
	if (cyl)
	{
	  cylinders->add(p1, m1, p2, m1);
	}
	else
	{
	  lines->add(p1, m1, p2, m1);
	}
      }
      break;
    }
    
    ++eiter;
  }

  return edge_switch;
}



template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_faces(const Fld *sfld,
				    bool use_normals,
				    bool use_transparency)
{
  //cerr << "rendering faces" << endl;
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
  const bool with_normals = (use_normals && mesh->has_normals());

  GeomSwitch *face_switch;
  GeomTriangles* faces;
  if (use_transparency)
  {
    faces = scinew GeomTranspTriangles;
    face_switch = scinew GeomSwitch(faces);
  }
  else
  {
    faces = scinew GeomTriangles;
    GeomDL *display_list = scinew GeomDL(faces);
    face_switch = scinew GeomSwitch(display_list);
  }

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

  return face_switch;
}


template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_text(FieldHandle field_handle,
				   bool use_default_material,
				   MaterialHandle default_material,
				   bool backface_cull_p,
				   int fontsize,
				   bool render_data,
				   bool render_nodes,
				   bool render_edges,
				   bool render_faces,
				   bool render_cells)
{
  GeomGroup *texts = scinew GeomGroup;
  GeomSwitch *text_switch = scinew GeomSwitch(texts);

  if (render_data)
  {
    texts->add(render_text_data(field_handle, use_default_material,
				default_material, backface_cull_p,
				fontsize));
  }
  if (render_nodes)
  {
    texts->add(render_text_nodes(field_handle, use_default_material,
				 default_material, backface_cull_p,
				 fontsize));
  }
  if (render_edges)
  {
    texts->add(render_text_edges(field_handle, use_default_material,
				 default_material, fontsize));
  }
  if (render_faces)
  {
    texts->add(render_text_faces(field_handle, use_default_material,
				 default_material, fontsize));
  }
  if (render_cells)
  {
    texts->add(render_text_cells(field_handle, use_default_material,
				 default_material, fontsize));
  }
  return text_switch;
}


template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_text_data(FieldHandle field_handle,
					bool use_default_material,
					MaterialHandle default_material,
					bool backface_cull_p,
					int fontsize)
{
  if (backface_cull_p && field_handle->data_at() == Field::NODE)
  {
    return render_text_data_nodes(field_handle, use_default_material,
				  default_material, backface_cull_p, fontsize);
  }

  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  GeomTexts *texts = 0;
  GeomTextsCulled *ctexts = 0;
  GeomSwitch *text_switch = 0;
  const bool culling_p = false; //backface_cull_p && mesh->has_normals();
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
    text_switch = scinew GeomSwitch(texts);
    texts->set_font_index(fontsize);
  }

  char buffer[256];
  char format[256];
  snprintf(format, 256, "%%%d.%df", 1, 2);
  typename Loc::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  Vector n;
  while (iter != end) {
    typename Fld::value_type tmp;
    if (fld->value(tmp, *iter)) {
      mesh->get_center(p, *iter);
      double val;
      to_double(tmp, val);
      
      snprintf(buffer, 256, format, val);
      const std::string as_str(buffer);
      MaterialHandle m;
      if (use_default_material)
      {
	m = default_material;
      }
      else
      {
	m = choose_mat(use_default_material, *iter);
      }
      if (culling_p)
      {
	//mesh->get_normal(n, *iter);
	ctexts->add(as_str, p, n, m->diffuse);
      }
      else
      {
	texts->add(as_str, p, m->diffuse);
      }
    }
    ++iter;
  }

  return text_switch;
}


template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_text_data_nodes(FieldHandle field_handle,
					      bool use_default_material,
					      MaterialHandle default_material,
					      bool backface_cull_p,
					      int fontsize)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  GeomTexts *texts = 0;
  GeomTextsCulled *ctexts = 0;
  GeomSwitch *text_switch = 0;
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
    text_switch = scinew GeomSwitch(texts);
    texts->set_font_index(fontsize);
  }

  char buffer[256];
  char format[256];
  snprintf(format, 256, "%%%d.%df", 1, 2);
  typename Fld::mesh_type::Node::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  Vector n;
  while (iter != end) {
    typename Fld::value_type tmp;
    if (fld->value(tmp, *iter)) {
      mesh->get_center(p, *iter);
      double val;
      to_double(tmp, val);
      
      snprintf(buffer, 256, format, val);
      const std::string as_str(buffer);
      MaterialHandle m;
      if (use_default_material)
      {
	m = default_material;
      }
      else
      {
	m = choose_mat(use_default_material, *iter);
      }
      if (culling_p)
      {
	mesh->get_normal(n, *iter);
	ctexts->add(as_str, p, n, m->diffuse);
      }
      else
      {
	texts->add(as_str, p, m->diffuse);
      }
    }
    ++iter;
  }

  return text_switch;
}



template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_text_nodes(FieldHandle field_handle,
					 bool use_default_material,
					 MaterialHandle default_material,
					 bool backface_cull_p,
					 int fontsize)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();
  GeomTexts *texts = 0;
  GeomTextsCulled *ctexts = 0;
  GeomSwitch *text_switch = 0;

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
    text_switch = scinew GeomSwitch(texts);
    texts->set_font_index(fontsize);
  }

  char buffer[256];
  typename Fld::mesh_type::Node::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  Vector n;
  while (iter != end)
  {
    mesh->get_center(p, *iter);
    snprintf(buffer, 256, "%d", (int)(*iter));
    const std::string as_str(buffer);
    MaterialHandle m;
    if (use_default_material)
    {
      m = default_material;
    }
    else
    {
      m = choose_mat(use_default_material, *iter);
    }
    if (culling_p)
    {
      mesh->get_normal(n, *iter);
      ctexts->add(as_str, p, n, m->diffuse);
    }
    else
    {
      texts->add(as_str, p, m->diffuse);
    }

    ++iter;
  }
  return text_switch;
}


template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_text_edges(FieldHandle field_handle,
					 bool use_default_material,
					 MaterialHandle default_material,
					 int fontsize)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();
  mesh->synchronize(Mesh::EDGES_E);

  GeomTexts *texts = scinew GeomTexts;
  GeomSwitch *text_switch = scinew GeomSwitch(texts);
  texts->set_font_index(fontsize);
  char buffer[256];
  typename Fld::mesh_type::Edge::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  while (iter != end)
  {
    mesh->get_center(p, *iter);
    snprintf(buffer, 256, "%d", (int)(*iter));
    const std::string as_str(buffer);
    MaterialHandle m;
    if (use_default_material)
    {
      m = default_material;
    }
    else
    {
      m = choose_mat(use_default_material, *iter);
    }
    texts->add(as_str, p, m->diffuse);
 
    ++iter;
  }
  return text_switch;
}

template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_text_faces(FieldHandle field_handle,
					 bool use_default_material,
					 MaterialHandle default_material,
					 int fontsize)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();
  mesh->synchronize(Mesh::FACES_E);

  GeomTexts *texts = scinew GeomTexts;
  GeomSwitch *text_switch = scinew GeomSwitch(texts);
  texts->set_font_index(fontsize);
  char buffer[256];
  typename Fld::mesh_type::Face::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  while (iter != end)
  {
    mesh->get_center(p, *iter);
    snprintf(buffer, 256, "%d", (int)(*iter));
    const std::string as_str(buffer);
    MaterialHandle m;
    if (use_default_material)
    {
      m = default_material;
    }
    else
    {
      m = choose_mat(use_default_material, *iter);
    }
    texts->add(as_str, p, m->diffuse);

    ++iter;
  }
  return text_switch;
}

template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_text_cells(FieldHandle field_handle,
					 bool use_default_material,
					 MaterialHandle default_material,
					 int fontsize)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();
  mesh->synchronize(Mesh::CELLS_E);

  GeomTexts *texts = scinew GeomTexts;
  GeomSwitch *text_switch = scinew GeomSwitch(texts);
  texts->set_font_index(fontsize);
  char buffer[256];
  typename Fld::mesh_type::Cell::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  while (iter != end)
  {
    mesh->get_center(p, *iter);
    snprintf(buffer, 256, "%d", (int)(*iter));
    const std::string as_str(buffer);
    MaterialHandle m;
    if (use_default_material)
    {
      m = default_material;
    }
    else
    {
      m = choose_mat(use_default_material, *iter);
    }
    texts->add(as_str, p, m->diffuse);

    ++iter;
  }
  return text_switch;
}


//! RenderFieldBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! RenderFieldBase from the DynamicAlgoBase they will have a pointer to.
class RenderFieldDataBase : public DynamicAlgoBase
{
public:

  virtual GeomSwitch *render_data(FieldHandle vfld_handle,
				  FieldHandle cfld_handle,
				  ColorMapHandle cmap,
				  MaterialHandle default_material,
				  const string &data_display_mode,
				  double scale, bool normalize,
				  bool bidirectional,
				  bool arrow_heads,
				  int resolution) = 0;



  RenderFieldDataBase();
  virtual ~RenderFieldDataBase();

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *vftd,
					    const TypeDescription *cftd,
					    const TypeDescription *ltd);

protected:

  void add_disk(const Point &p, const Vector &vin,
		double scale, int resolution,
		GeomGroup *g, MaterialHandle mh,
		bool normalize);

  void add_cone(const Point &p, const Vector &vin,
		double scale, int resolution,
		GeomGroup *g, MaterialHandle mh,
		bool normalize);
};


template <class VFld, class CFld, class Loc>
class RenderFieldData : public RenderFieldDataBase
{
public:
  virtual GeomSwitch *render_data(FieldHandle vfld_handle,
				  FieldHandle cfld_handle,
				  ColorMapHandle cmap,
				  MaterialHandle default_material,
				  const string &data_display_mode,
				  double scale,
				  bool normalize,
				  bool bidirectional,
				  bool arrow_heads,
				  int resolution);
};


template <class VFld, class CFld, class Loc>
GeomSwitch *
RenderFieldData<VFld, CFld, Loc>::render_data(FieldHandle vfld_handle,
					      FieldHandle cfld_handle,
					      ColorMapHandle cmap,
					      MaterialHandle default_material,
					      const string &display_mode,
					      double scale, 
					      bool normalize,
					      bool bidirectional,
					      bool arrow_heads,
					      int resolution)
{
  VFld *vfld = dynamic_cast<VFld*>(vfld_handle.get_rep());
  CFld *cfld = dynamic_cast<CFld*>(cfld_handle.get_rep());

  GeomGroup *disks;
  GeomArrows *vec_node;
  const bool disks_p = (display_mode == "Disks");
  const bool cones_p = (display_mode == "Cones");
  GeomSwitch *data_switch;
  if (disks_p || cones_p)
  {
    disks = scinew GeomGroup();
    data_switch = scinew GeomSwitch(disks);
  }
  else
  {
    vec_node = scinew GeomArrows(arrow_heads?0.15:0.0, 0.6);
    data_switch = scinew GeomSwitch(vec_node);
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

      typename CFld::value_type ctmp;
      cfld->value(ctmp, *iter);

      double ctmpd;
      to_double(ctmp, ctmpd);

      if (disks_p)
      {
	add_disk(p, tmp, scale, resolution, disks,
		 (cmap.get_rep())?(cmap->lookup(ctmpd)):default_material,
		 normalize);
      }
      else if (cones_p)
      {
	add_cone(p, tmp, scale, resolution, disks,
		 (cmap.get_rep())?(cmap->lookup(ctmpd)):default_material,
		 normalize);
      }
      else
      {
	add_data(p, tmp, vec_node,
		 (cmap.get_rep())?(cmap->lookup(ctmpd)):default_material,
		 display_mode, scale, normalize, bidirectional);
      }
    }
    ++iter;
  }
  return data_switch;
}


} // end namespace SCIRun

#endif // Visualization_RenderField_h
