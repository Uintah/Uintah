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
#include <Core/Geom/GeomQuads.h>
#include <Core/Geom/GeomBox.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomDL.h>
#include <Core/Geom/GeomColorMap.h>
#include <Core/Geom/GeomPoint.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Containers/StringUtil.h>
#include <sci_hash_map.h>
#include <Core/Datatypes/TetVolMesh.h>

#include <sstream>

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
		      bool faces, MaterialHandle def_mat, 
		      bool def_col, ColorMapHandle color_handle,
		      const string &ndt, const string &edt, 
		      double ns, double es, double vs, bool normalize, 
		      int sphere_resolution, int cylinder_resolution,
		      bool use_normals,
		      bool node_transparency,
		      bool edge_transparency,
		      bool face_transparency,
		      bool bidirectional) = 0;

  virtual GeomSwitch *render_text(FieldHandle fld,
				  bool use_default_material,
				  MaterialHandle default_material,
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
  MaterialHandle           def_mat_handle_;
  ColorMapHandle           color_handle_;
  ind_mat_t               *mats_;

  void add_sphere(const Point &p, double scale, 
		  int resolution, GeomGroup *g, 
		  MaterialHandle m0);
  void add_disk(const Point &p, const Vector& v, double scale, 
		int resolution, GeomGroup *g, MaterialHandle m0);
  void add_axis(const Point &p, double scale, GeomCLines *lines, 
		MaterialHandle m0);

  inline  MaterialHandle choose_mat(bool def, int idx) {  
    if (def) return def_mat_handle_;
    ASSERT(mats_ != 0); 
    return (*mats_)[idx];
  }
};


template <class Fld, class Loc>
class RenderField : public RenderFieldBase
{
public:
  //! virtual interface. 
  virtual void render(FieldHandle fh,  
		      bool nodes, bool edges, bool faces,
		      MaterialHandle def_mat,
		      bool data_at, ColorMapHandle color_handle,
		      const string &ndt, const string &edt, 
		      double ns, double es, double vs, bool normalize, 
		      int sphere_resolution, int cylinder_resolution,
		      bool use_normals,
		      bool node_transparency,
		      bool edge_transparency,
		      bool face_transparency,
		      bool bidirectional);

  virtual GeomSwitch *render_text(FieldHandle fld,
				  bool use_default_material,
				  MaterialHandle default_material,
				  bool backface_cull_p,
				  int fontsize,
				  int precision,
				  bool render_locations,
				  bool render_data,
				  bool render_nodes,
				  bool render_edges,
				  bool render_faces,
				  bool render_cells);

private:
  GeomSwitch *render_nodes(const Fld *fld, 
			   const string &node_display_mode,
			   double node_scale,
			   int node_resolution,
			   bool use_transparency);
  GeomSwitch *render_edges(const Fld *fld,
			   const string &edge_display_mode,
			   double edge_scale,
			   int cylinder_resolution,
			   bool transparent_p);
  GeomSwitch *render_faces(const Fld *fld, 
			   bool use_normals,
			   bool use_transparency);

  GeomSwitch *render_text_data(FieldHandle fld,
			       bool use_default_material,
			       MaterialHandle default_material,
			       bool backface_cull_p,
			       int fontsize,
			       int precision);
  GeomSwitch *render_text_data_nodes(FieldHandle fld,
				     bool use_default_material,
				     MaterialHandle default_material,
				     bool backface_cull_p,
				     int fontsize,
				     int precision);
  GeomSwitch *render_text_nodes(FieldHandle fld,
  				bool use_default_material,
  				MaterialHandle default_material,
				bool backface_cull_p,
				int fontsize,
				int precision,
				bool render_locations);
  GeomSwitch *render_text_edges(FieldHandle fld,
  				bool use_default_material,
  				MaterialHandle default_material,
				int fontsize,
				int precision,
				bool render_locations);
  GeomSwitch *render_text_faces(FieldHandle fld,
  				bool use_default_material,
  				MaterialHandle default_material,
				int fontsize,
				int precision,
				bool render_locations);
  GeomSwitch *render_text_cells(FieldHandle fld,
				bool use_default_material,
				MaterialHandle default_material,
				int fontsize,
				int precision,
				bool render_locations);

  void render_materials(const Fld *fld, const string &data_display_mode);
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
			      MaterialHandle def_mat,
			      bool def_col, ColorMapHandle color_handle,
			      const string &ndt, const string &edt,
			      double ns, double es, double vs, bool normalize, 
			      int sphere_res, int cyl_res,
			      bool use_normals,
			      bool n_transp,
			      bool e_transp,
			      bool f_transp,
			      bool bidirectional)
{
  Fld *fld = dynamic_cast<Fld*>(fh.get_rep());
  ASSERT(fld != 0);
  def_mat_handle_ = def_mat;
  color_handle_ = color_handle;

  if (def_col) { render_materials(fld, ndt); }
  if (nodes)
  {
    node_switch_ = render_nodes(fld, ndt, ns, sphere_res, n_transp);
  }
  
  if (edges)
  {
    edge_switch_ = render_edges(fld, edt, es, cyl_res, e_transp);
  }

  if (faces)
  {
    face_switch_ = render_faces(fld, use_normals, f_transp);
  }
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
    // Is known, we do nothing here.
    break;
  }

  // Update these when the data changes. 
  if (node_switch_.get_rep()) { node_switch_->reset_bbox(); }
  if (edge_switch_.get_rep()) { edge_switch_->reset_bbox(); }
  if (face_switch_.get_rep()) { face_switch_->reset_bbox(); }
}



template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_nodes(const Fld *sfld, 
				    const string &node_display_mode,
				    double node_scale,
				    int node_resolution,
				    bool use_transparency)
{
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
  GeomGroup* nodes = scinew GeomGroup;
  GeomDL *display_list =
    scinew GeomDL(scinew GeomMaterial(nodes, def_mat_handle_));
  GeomSwitch *node_switch = scinew GeomSwitch(display_list);
  GeomPoints *points = 0;
  GeomCLines *lines = 0;

  // 0 Points 1 Spheres 2 Axes 3 Disks
  int mode = 0;
  if (node_display_mode == "Points")       { mode = 0; }
  else if (node_display_mode == "Spheres") { mode = 1; }
  else if (node_display_mode == "Axes")    { mode = 2; }
  else if (node_display_mode == "Disks")   { mode = 3; }

  if (mode == 0) { // Points
    if (use_transparency)
    {
      points = scinew GeomTranspPoints();
    }
    else
    {
      points = scinew GeomPoints();
    }
  }
  else if (mode == 2) // Axis
  {
    if (use_transparency)
    {
      lines = scinew GeomTranspLines();
    }
    else
    {
      lines = scinew GeomCLines();
    }
    lines->setLineWidth(3);
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
    double val;
    switch (sfld->data_at())
    {
    case Field::NODE:
      {
	typename Fld::value_type tmp;
	sfld->value(tmp, *niter);
	
	to_vector(tmp, vec);
	to_double(tmp, val);
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
      if (def_color)
      {
	points->add(p);
      }
      else
      {
	//points->add(p, choose_mat(def_color, *niter));
	points->add(p, val);
      }
      break;

    case 1: // Spheres
      if (def_color)
      {
	add_sphere(p, node_scale, node_resolution, nodes, 0);
      }
      else
      {
	add_sphere(p, node_scale, node_resolution, nodes,
		   choose_mat(def_color, *niter));
      }
      break;

    case 2: // Axes
      if (def_color)
      {
	add_axis(p, node_scale, lines, 0);
      }
      else
      {
	add_axis(p, node_scale, lines, choose_mat(def_color, *niter));
      }
      break;

    case 3: // Disks
    default:
      if (def_color)
      {
	add_disk(p, vec, node_scale, node_resolution, nodes, 0);
      }
      else
      {
	add_disk(p, vec, node_scale, node_resolution,
		 nodes, choose_mat(def_color, *niter));
      }
      break;
    }
    ++niter;
  }
  if (mode == 0) { // Points
    nodes->add(scinew GeomColorMap(scinew GeomMaterial(points, def_mat_handle_),
				   color_handle_));
  }
  else if (mode == 2) { // Axes
    nodes->add(scinew GeomMaterial(lines, def_mat_handle_));
  }

  return node_switch;
}



template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_edges(const Fld *sfld,
				    const string &edge_display_mode,
				    double edge_scale,
				    int cylinder_resolution,
				    bool transparent_p) 
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
    if (transparent_p)
    {
      lines = scinew GeomTranspLines;
    }
    else
    {
      lines = scinew GeomCLines;
    }
    lines->setLineWidth(edge_scale);
    GeomDL *display_list =
      scinew GeomDL(scinew GeomMaterial(lines, def_mat_handle_));
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
	  lines->add(p1, p2);
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
  GeomFastTriangles* faces;
  GeomFastQuads* qfaces;
  if (use_transparency)
  {
    faces = scinew GeomTranspTriangles;
    qfaces = scinew GeomTranspQuads;
    GeomGroup *tmp = scinew GeomGroup;
    tmp->add(faces);
    tmp->add(qfaces);
    face_switch = scinew GeomSwitch(tmp);
  }
  else
  {
    faces = scinew GeomFastTriangles;
    qfaces = scinew GeomFastQuads;
    GeomGroup *tmp = scinew GeomGroup;
    tmp->add(faces);
    tmp->add(qfaces);
    GeomDL *display_list = scinew GeomDL(tmp);
    face_switch = scinew GeomSwitch(display_list);
  }

  // Third pass: over the faces
  if (with_normals) mesh->synchronize(Mesh::NORMALS_E);
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

    if (nodes.size() == 4)
    {
      if (with_normals)
      {
	qfaces->add(points[0], normals[0], mats[0],
		    points[1], normals[1], mats[1],
		    points[2], normals[2], mats[2],
		    points[3], normals[3], mats[3]);
      }
      else
      {
	qfaces->add(points[0], mats[0],
		    points[1], mats[1],
		    points[2], mats[2],
		    points[3], mats[3]);
      }
    }
    else
    {
      for (i=2; i<nodes.size(); i++)
      {
	if (with_normals)
	{
	  faces->add(points[0], normals[0], mats[0],
		     points[i-1], normals[i-1], mats[i-1],
		     points[i], normals[i], mats[i]);
	}
	else
	{
	  faces->add(points[0], mats[0],
		     points[i-1], mats[i-1],
		     points[i], mats[i]);
	}
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
				   int precision,
				   bool render_locations,
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
				fontsize, precision));
  }
  if (render_nodes)
  {
    texts->add(render_text_nodes(field_handle, use_default_material,
				 default_material, backface_cull_p,
				 fontsize, precision, render_locations));
  }
  if (render_edges)
  {
    texts->add(render_text_edges(field_handle, use_default_material,
				 default_material, fontsize,
				 precision, render_locations));
  }
  if (render_faces)
  {
    texts->add(render_text_faces(field_handle, use_default_material,
				 default_material, fontsize,
				 precision, render_locations));
  }
  if (render_cells)
  {
    texts->add(render_text_cells(field_handle, use_default_material,
				 default_material, fontsize,
				 precision, render_locations));
  }
  return text_switch;
}


template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_text_data(FieldHandle field_handle,
					bool use_default_material,
					MaterialHandle default_material,
					bool backface_cull_p,
					int fontsize,
					int precision)
{
  if (backface_cull_p && field_handle->data_at() == Field::NODE)
  {
    return render_text_data_nodes(field_handle, use_default_material,
				  default_material, backface_cull_p, fontsize,
				  precision);
  }

  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  GeomTexts *texts = scinew GeomTexts();
  GeomSwitch *text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  texts->set_font_index(fontsize);

  std::ostringstream buffer;
  buffer.precision(precision);

  typename Loc::iterator iter, end;
  mesh->begin(iter);
  mesh->end(end);
  Point p;
  while (iter != end) {
    typename Fld::value_type val;
    if (fld->value(val, *iter)) {
      mesh->get_center(p, *iter);

      buffer.str("");
      buffer << val;

      MaterialHandle m;
      if (use_default_material)
      {
	m = default_material;
      }
      else
      {
	m = choose_mat(use_default_material, *iter);
      }
      texts->add(buffer.str(), p, m->diffuse);
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
					      int fontsize,
					      int precision)
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
    text_switch = scinew GeomSwitch(scinew GeomDL(texts));
    texts->set_font_index(fontsize);
  }

  std::ostringstream buffer;
  buffer.precision(precision);

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
      buffer << val;

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
	ctexts->add(buffer.str(), p, n, m->diffuse);
      }
      else
      {
	texts->add(buffer.str(), p, m->diffuse);
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
					 int fontsize,
					 int precision,
					 bool render_locations)
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
    text_switch = scinew GeomSwitch(scinew GeomDL(texts));
    texts->set_font_index(fontsize);
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
      buffer << (int)(*iter);
    }

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
      ctexts->add(buffer.str(), p, n, m->diffuse);
    }
    else
    {
      texts->add(buffer.str(), p, m->diffuse);
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
					 int fontsize,
					 int precision,
					 bool render_locations)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();
  mesh->synchronize(Mesh::EDGES_E);

  GeomTexts *texts = scinew GeomTexts;
  GeomSwitch *text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  texts->set_font_index(fontsize);

  ostringstream buffer;
  buffer.precision(precision);

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

    MaterialHandle m;
    if (use_default_material)
    {
      m = default_material;
    }
    else
    {
      m = choose_mat(use_default_material, *iter);
    }
    texts->add(buffer.str(), p, m->diffuse);
 
    ++iter;
  }
  return text_switch;
}

template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_text_faces(FieldHandle field_handle,
					 bool use_default_material,
					 MaterialHandle default_material,
					 int fontsize,
					 int precision,
					 bool render_locations)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();
  mesh->synchronize(Mesh::FACES_E);

  GeomTexts *texts = scinew GeomTexts;
  GeomSwitch *text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  texts->set_font_index(fontsize);

  ostringstream buffer;
  buffer.precision(precision);

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
      buffer << (int)(*iter);
    }

    MaterialHandle m;
    if (use_default_material)
    {
      m = default_material;
    }
    else
    {
      m = choose_mat(use_default_material, *iter);
    }
    texts->add(buffer.str(), p, m->diffuse);

    ++iter;
  }
  return text_switch;
}

template <class Fld, class Loc>
GeomSwitch *
RenderField<Fld, Loc>::render_text_cells(FieldHandle field_handle,
					 bool use_default_material,
					 MaterialHandle default_material,
					 int fontsize,
					 int precision,
					 bool render_locations)
{
  Fld *fld = dynamic_cast<Fld *>(field_handle.get_rep());
  ASSERT(fld);

  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();
  mesh->synchronize(Mesh::CELLS_E);

  GeomTexts *texts = scinew GeomTexts;
  GeomSwitch *text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  texts->set_font_index(fontsize);

  ostringstream buffer;
  buffer.precision(precision);

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
      buffer << (int)(*iter);
    }

    MaterialHandle m;
    if (use_default_material)
    {
      m = default_material;
    }
    else
    {
      m = choose_mat(use_default_material, *iter);
    }
    texts->add(buffer.str(), p, m->diffuse);

    ++iter;
  }
  return text_switch;
}


//! RenderVectorFieldBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! RenderVectorFieldBase from the DynamicAlgoBase they will have a pointer to.
class RenderVectorFieldBase : public DynamicAlgoBase
{
public:

  virtual GeomSwitch *render_data(FieldHandle vfld_handle,
				  FieldHandle cfld_handle,
				  ColorMapHandle cmap,
				  MaterialHandle default_material,
				  const string &data_display_mode,
				  double scale, bool normalize,
				  bool bidirectional,
				  int resolution) = 0;



  RenderVectorFieldBase();
  virtual ~RenderVectorFieldBase();

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *vftd,
					    const TypeDescription *cftd,
					    const TypeDescription *ltd);

protected:

  void add_disk(const Point &p, const Vector &vin,
		double scale, int resolution,
		GeomGroup *g, MaterialHandle mh,
		bool normalize, bool colorify);

  void add_cone(const Point &p, const Vector &vin,
		double scale, int resolution,
		GeomGroup *g, MaterialHandle mh,
		bool normalize, bool colorify);
};


template <class VFld, class CFld, class Loc>
class RenderVectorField : public RenderVectorFieldBase
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
				  int resolution);
};


template <class VFld, class CFld, class Loc>
GeomSwitch *
RenderVectorField<VFld, CFld, Loc>::render_data(FieldHandle vfld_handle,
						FieldHandle cfld_handle,
						ColorMapHandle cmap,
						MaterialHandle default_material,
						const string &display_mode,
						double scale, 
						bool normalize,
						bool bidirectional,
						int resolution)
{
  VFld *vfld = dynamic_cast<VFld*>(vfld_handle.get_rep());
  CFld *cfld = dynamic_cast<CFld*>(cfld_handle.get_rep());

  const bool colorify = false;

  GeomGroup *disks;
  GeomArrows *vec_node;
  GeomCLines *lines;
  const bool lines_p = (display_mode == "Lines");
  const bool needles_p = (display_mode == "Needles");
  const bool cones_p = (display_mode == "Cones");
  const bool arrows_p = (display_mode == "Arrows");
  const bool disks_p = (display_mode == "Disks");
  GeomSwitch *data_switch;
  if (disks_p || cones_p)
  {
    disks = scinew GeomGroup();
    data_switch =
      scinew GeomSwitch(scinew GeomDL(scinew GeomMaterial(disks,
							  default_material)));
  }
  else if (arrows_p)
  {
    vec_node = scinew GeomArrows(0.15, 0.6);
    data_switch = scinew GeomSwitch(scinew GeomDL(vec_node));
  }
  else if (lines_p || needles_p)
  {
    if (lines_p)
    {
      lines = scinew GeomCLines();
    }
    else
    {
      lines = scinew GeomTranspLines();
    }

    data_switch =
      scinew GeomSwitch(scinew GeomDL(scinew GeomMaterial(lines,
							  default_material)));
  }

  MaterialHandle tdefmat = default_material->clone();
  tdefmat->transparency = 0.0;

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
		 (cmap.get_rep())?(cmap->lookup(ctmpd)):0,
		 normalize, colorify);
      }
      else if (cones_p)
      {
	add_cone(p, tmp, scale, resolution, disks,
		 (cmap.get_rep())?(cmap->lookup(ctmpd)):0,
		 normalize, colorify);
      }
      else if (arrows_p)
      {
	add_data(p, tmp, vec_node,
		 (cmap.get_rep())?(cmap->lookup(ctmpd)):default_material,
		 display_mode, scale, normalize, bidirectional);
      }
      else if (lines_p)
      {
	if (normalize)
	{
	  tmp.safe_normalize();
	}
	tmp *= scale;
	if (bidirectional)
	{
	  if (cmap.get_rep())
	  {
	    lines->add(p - tmp, cmap->lookup(ctmpd),
		       p + tmp, cmap->lookup(ctmpd));
	  }
	  else
	  {
	    lines->add(p - tmp, p + tmp);
	  }
	}
	else
	{
	  if (cmap.get_rep())
	  {
	    lines->add(p, cmap->lookup(ctmpd),
		       p + tmp, cmap->lookup(ctmpd));
	  }
	  else
	  {
	    lines->add(p, p + tmp);
	  }
	}
      }
      else // Needles
      {
	if (normalize)
	{
	  tmp.safe_normalize();
	}
	tmp *= scale;
	if (cmap.get_rep())
	{
	  MaterialHandle color = cmap->lookup(ctmpd);
	  MaterialHandle tcolor = color->clone();
	  tcolor->transparency = 0.0;
	  lines->add(p, color, p + tmp, tcolor);
	  if (bidirectional)
	  {
	    lines->add(p, color, p - tmp, tcolor);
	  }
	}
	else
	{
	  lines->add(p, default_material, p + tmp, tdefmat);
	  if (bidirectional)
	  {
	    lines->add(p, default_material, p - tmp, tdefmat);
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

  virtual GeomSwitch *render_data(FieldHandle vfld_handle,
				  FieldHandle cfld_handle,
				  ColorMapHandle cmap,
				  MaterialHandle default_material,
				  const string &data_display_mode,
				  double scale,
				  int resolution) = 0;



  RenderTensorFieldBase();
  virtual ~RenderTensorFieldBase();

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *vftd,
					    const TypeDescription *cftd,
					    const TypeDescription *ltd);

protected:

  void add_item(GeomHandle glyph, const Point &p, Tensor &t,
		double scale, int resolution, GeomGroup *g, bool colorize);
};


template <class VFld, class CFld, class Loc>
class RenderTensorField : public RenderTensorFieldBase
{
public:
  virtual GeomSwitch *render_data(FieldHandle vfld_handle,
				  FieldHandle cfld_handle,
				  ColorMapHandle cmap,
				  MaterialHandle default_material,
				  const string &data_display_mode,
				  double scale,
				  int resolution);
};


template <class VFld, class CFld, class Loc>
GeomSwitch *
RenderTensorField<VFld, CFld, Loc>::render_data(FieldHandle vfld_handle,
						FieldHandle cfld_handle,
						ColorMapHandle cmap,
						MaterialHandle def_mat,
						const string &display_mode,
						double scale, 
						int resolution)
{
  VFld *vfld = dynamic_cast<VFld*>(vfld_handle.get_rep());
  CFld *cfld = dynamic_cast<CFld*>(cfld_handle.get_rep()); 

  const bool box_p = (display_mode == "Boxes");
  const bool sphere_p = (display_mode == "Ellipsoids");
  const bool cbox_p = (display_mode == "Colored Boxes");

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
  else // cbox_p, default
  {
    glyph = scinew GeomCBox(Point(-1.0, -1.0, -1.0),
			    Point(1.0, 1.0, 1.0));
  }

  GeomGroup *objs = scinew GeomGroup(); 
  GeomSwitch *data_switch =
    scinew GeomSwitch(scinew GeomMaterial(scinew GeomDL(objs), def_mat));

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

      if (cmap.get_rep() && !cbox_p)
      {
	add_item(scinew GeomMaterial(glyph, cmap->lookup(ctmpd)),
		 p, tmp, scale, resolution, objs, false);
      }
      else
      {
	add_item(glyph, p, tmp, scale, resolution, objs, !cbox_p);
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

  virtual GeomSwitch *render_data(FieldHandle vfld_handle,
				  FieldHandle cfld_handle,
				  ColorMapHandle cmap,
				  MaterialHandle default_material,
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

protected:

  void add_sphere(const Point &p, double scale, int resolution,
		  GeomGroup *g, GeomPoints *points,
		  MaterialHandle color = 0);
};


template <class VFld, class CFld, class Loc>
class RenderScalarField : public RenderScalarFieldBase
{
public:
  virtual GeomSwitch *render_data(FieldHandle vfld_handle,
				  FieldHandle cfld_handle,
				  ColorMapHandle cmap,
				  MaterialHandle default_material,
				  const string &data_display_mode,
				  double scale,
				  int resolution,
				  bool transparent_p);
};


template <class SFld, class CFld, class Loc>
GeomSwitch *
RenderScalarField<SFld, CFld, Loc>::render_data(FieldHandle sfld_handle,
						FieldHandle cfld_handle,
						ColorMapHandle cmap,
						MaterialHandle def_mat,
						const string &display_mode,
						double scale, 
						int resolution,
						bool transparent_p)
{
  SFld *sfld = dynamic_cast<SFld*>(sfld_handle.get_rep());
  CFld *cfld = dynamic_cast<CFld*>(cfld_handle.get_rep()); 

  const bool points_p = (display_mode == "Points");
  const bool sized_p = (display_mode == "Scaled Spheres");

  GeomSwitch *data_switch = 0;
  GeomGroup *objs = 0;
  GeomPoints *points = 0;

  if (points_p)
  {
    if (transparent_p)
    {
      points = scinew GeomTranspPoints();
    }
    else
    {
      points = scinew GeomPoints();
    }
    data_switch =
      scinew GeomSwitch(scinew GeomColorMap(scinew GeomMaterial(points, def_mat), cmap));
  }
  else
  {
    objs = scinew GeomGroup();
    data_switch = scinew GeomSwitch(scinew GeomMaterial(objs, def_mat));
    if (sized_p)
    {
      points = scinew GeomPoints();
      objs->add(points);
    }
  }
  
  typename SFld::mesh_handle_type mesh = sfld->get_typed_mesh();

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
	if (cmap.get_rep())
	{
	  typename CFld::value_type ctmp;
	  cfld->value(ctmp, *iter);

	  double ctmpd;
	  to_double(ctmp, ctmpd);
	  points->add(p, ctmpd);
	}
	else
	{
	  points->add(p);
	}
      }
      else
      {
	const double dtmp = sized_p?fabs((double)tmp):1.0;
	if (cmap.get_rep())
	{
	  typename CFld::value_type ctmp;
	  cfld->value(ctmp, *iter);

	  double ctmpd;
	  to_double(ctmp, ctmpd);

	  add_sphere(p, scale * dtmp, resolution, objs, points,
		     cmap->lookup(ctmpd));
	}
	else
	{
	  add_sphere(p, scale * dtmp, resolution, objs, points);
	}
      }
    }
    ++iter;
  }
  return data_switch;
}


} // end namespace SCIRun

#endif // Visualization_RenderField_h
