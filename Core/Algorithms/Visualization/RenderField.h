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
		      bool bidirectional) = 0;

  virtual GeomHandle render_text(FieldHandle fld,
				 ColorMapHandle color_handle,
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

  void add_sphere(const Point &p, double scale, 
		  int resolution, GeomGroup *g, 
		  MaterialHandle m0);
  void add_disk(const Point &p, const Vector& v, double scale, 
		int resolution, GeomGroup *g, MaterialHandle m0);
  void add_axis(const Point &p, double scale, GeomLines *lines);
  void add_axis(const Point &p, double scale, GeomLines *lines, double val);
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
		      bool bidirectional);

  virtual GeomHandle render_text(FieldHandle fld,
				 ColorMapHandle color_handle,
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
  GeomHandle render_nodes(const Fld *fld, 
			  const string &node_display_mode,
			  ColorMapHandle color_handle,
			  MaterialHandle def_mat,
			  double node_scale,
			  int node_resolution,
			  bool use_transparency);
  GeomHandle render_edges(const Fld *fld,
			  const string &edge_display_mode,
			  ColorMapHandle color_handle,
			  MaterialHandle def_mat,
			  double edge_scale,
			  int cylinder_resolution,
			  bool transparent_p);
  GeomHandle render_faces(const Fld *fld, 
			  ColorMapHandle color_handle,
			  MaterialHandle def_mat,
			  bool use_normals,
			  bool use_transparency);

  GeomHandle render_text_data(FieldHandle fld,
			      ColorMapHandle color_handle,
			      bool use_default_material,
			      MaterialHandle default_material,
			      bool backface_cull_p,
			      int fontsize,
			      int precision);
  GeomHandle render_text_data_nodes(FieldHandle fld,
				    ColorMapHandle color_handle,
				    bool use_default_material,
				    MaterialHandle default_material,
				    bool backface_cull_p,
				    int fontsize,
				    int precision);
  GeomHandle render_text_nodes(FieldHandle fld,
			       ColorMapHandle color_handle,
			       bool use_default_material,
			       MaterialHandle default_material,
			       bool backface_cull_p,
			       int fontsize,
			       int precision,
			       bool render_locations);
  GeomHandle render_text_edges(FieldHandle fld,
			       ColorMapHandle color_handle,
			       bool use_default_material,
			       MaterialHandle default_material,
			       int fontsize,
			       int precision,
			       bool render_locations);
  GeomHandle render_text_faces(FieldHandle fld,
			       ColorMapHandle color_handle,
			       bool use_default_material,
			       MaterialHandle default_material,
			       int fontsize,
			       int precision,
			       bool render_locations);
  GeomHandle render_text_cells(FieldHandle fld,
			       ColorMapHandle color_handle,
			       bool use_default_material,
			       MaterialHandle default_material,
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
			      bool n_transp,
			      bool e_transp,
			      bool f_transp,
			      bool bidirectional)
{
  Fld *fld = dynamic_cast<Fld*>(fh.get_rep());
  ASSERT(fld != 0);

  if (nodes)
  {
    node_switch_ = render_nodes(fld, ndt, color_handle, def_mat,
				ns, sphere_res, n_transp);
  }
  
  if (edges)
  {
    edge_switch_ = render_edges(fld, edt, color_handle, def_mat,
				es, cyl_res, e_transp);
  }

  if (faces)
  {
    face_switch_ = render_faces(fld, color_handle, def_mat,
				use_normals, f_transp);
  }
}



template <class Fld, class Loc>
GeomHandle
RenderField<Fld, Loc>::render_nodes(const Fld *sfld, 
				    const string &node_display_mode,
				    ColorMapHandle color_handle,
				    MaterialHandle def_mat,
				    double node_scale,
				    int node_resolution,
				    bool use_transparency)
{
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
  GeomGroup *nodes = 0;
  GeomPoints *points = 0;
  GeomLines *lines = 0;
  GeomHandle display_list(0);

  // 0 Points 1 Spheres 2 Axes 3 Disks
  int mode = 0;
  if (node_display_mode == "Points")       { mode = 0; }
  else if (node_display_mode == "Spheres") { mode = 1; }
  else if (node_display_mode == "Axes")    { mode = 2; }
  else if (node_display_mode == "Disks")   { mode = 3; }

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
  else if (mode == 1 || mode == 3)
  {
    nodes = scinew GeomGroup();
    display_list = scinew GeomDL(nodes);
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

  // First pass: over the nodes
  mesh->synchronize(Mesh::NODES_E);
  typename Fld::mesh_type::Node::iterator niter;  mesh->begin(niter);  
  typename Fld::mesh_type::Node::iterator niter_end;  mesh->end(niter_end);  
  while (niter != niter_end) {
    // Use a default color?
    bool def_color = !(color_handle.get_rep());
    
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
		   color_handle->lookup(val));
      }
      break;

    case 2: // Axes
      if (def_color)
      {
	add_axis(p, node_scale, lines);
      }
      else
      {
	add_axis(p, node_scale, lines, val);
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
		 nodes, color_handle->lookup(val));
      }
      break;
    }
    ++niter;
  }

  return display_list;
}



template <class Fld, class Loc>
GeomHandle
RenderField<Fld, Loc>::render_edges(const Fld *sfld,
				    const string &edge_display_mode,
				    ColorMapHandle color_handle,
				    MaterialHandle def_mat,
				    double edge_scale,
				    int cylinder_resolution,
				    bool transparent_p) 
{
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();

  const bool cyl = edge_display_mode == "Cylinders";

  GeomLines* lines = NULL;
  GeomColoredCylinders* cylinders = NULL;
  GeomHandle display_list;
  if (cyl)
  {
    cylinders = scinew GeomColoredCylinders;
    cylinders->set_radius(edge_scale);
    cylinders->set_nu_nv(cylinder_resolution, 1);
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
	typename Fld::value_type val0, val1;
	sfld->value(val0, nodes[0]);
	sfld->value(val1, nodes[1]);
	double dval0, dval1;
	to_double(val0, dval0);
	to_double(val1, dval1);
	if (cyl)
	{
	  cylinders->add(p1, dval0, p2, dval1);
	}
	else
	{
	  lines->add(p1, dval0, p2, dval1);
	}
      }
      break;
    case Field::EDGE:
      {
	typename Fld::value_type val;
	sfld->value(val, *eiter);
	double dval;
	to_double(val, dval);
	if (cyl)
	{
	  cylinders->add(p1, dval, p2, dval);
	}
	else
	{
	  lines->add(p1, dval, p2, dval);
	}
      }
      break;
    case Field::FACE:
    case Field::CELL:
    case Field::NONE:
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
      break;
    }
    
    ++eiter;
  }

  return display_list;
}



template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_faces(const Fld *sfld,
				    ColorMapHandle color_handle,
				    MaterialHandle def_mat,
				    bool use_normals,
				    bool use_transparency)
{
  //cerr << "rendering faces" << endl;
  typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
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
	vector<typename Fld::value_type> vals(nodes.size());
	vector<double> dvals(nodes.size());
	for (i = 0; i < nodes.size(); i++)
	{
	  sfld->value(vals[i], nodes[i]);
	  to_double(vals[i], dvals[i]);
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
      break;
      
    case Field::FACE: 
      {
	typename Fld::value_type val;
	double dval;
	sfld->value(val, *fiter);
	to_double(val, dval);

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
      break;
      
    case Field::EDGE:
    case Field::CELL:
    case Field::NONE:
    default:
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


template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_text(FieldHandle field_handle,
				   ColorMapHandle color_handle,
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
  GeomHandle text_switch = scinew GeomSwitch(texts);

  if (render_data)
  {
    texts->add(render_text_data(field_handle, color_handle,
				use_default_material,
				default_material, backface_cull_p,
				fontsize, precision));
  }
  if (render_nodes)
  {
    texts->add(render_text_nodes(field_handle, color_handle,
				 use_default_material,
				 default_material, backface_cull_p,
				 fontsize, precision, render_locations));
  }
  if (render_edges)
  {
    texts->add(render_text_edges(field_handle, color_handle,
				 use_default_material,
				 default_material, fontsize,
				 precision, render_locations));
  }
  if (render_faces)
  {
    texts->add(render_text_faces(field_handle, color_handle,
				 use_default_material,
				 default_material, fontsize,
				 precision, render_locations));
  }
  if (render_cells)
  {
    texts->add(render_text_cells(field_handle, color_handle,
				 use_default_material,
				 default_material, fontsize,
				 precision, render_locations));
  }
  return text_switch;
}


template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_text_data(FieldHandle field_handle,
					ColorMapHandle color_handle,
					bool use_default_material,
					MaterialHandle default_material,
					bool backface_cull_p,
					int fontsize,
					int precision)
{
  if (backface_cull_p && field_handle->data_at() == Field::NODE)
  {
    return render_text_data_nodes(field_handle, color_handle,
				  use_default_material,
				  default_material, backface_cull_p, fontsize,
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
	double dval;
	to_double(val, dval);
	m = color_handle->lookup(dval);
      }
      texts->add(buffer.str(), p, m->diffuse);
    }
    ++iter;
  }

  return text_switch;
}


template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_text_data_nodes(FieldHandle field_handle,
					      ColorMapHandle color_handle,
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
	double dval;
	to_double(val, dval);
	m = color_handle->lookup(dval);
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
GeomHandle 
RenderField<Fld, Loc>::render_text_nodes(FieldHandle field_handle,
					 ColorMapHandle color_handle,
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
    if (use_default_material || fld->data_at() != Field::NODE)
    {
      m = default_material;
    }
    else
    {
      typename Fld::value_type val;
      fld->value(val, *iter);
      double dval;
      to_double(val, dval);
      m = color_handle->lookup(dval);
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
GeomHandle 
RenderField<Fld, Loc>::render_text_edges(FieldHandle field_handle,
					 ColorMapHandle color_handle,
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
  GeomHandle text_switch = scinew GeomSwitch(scinew GeomDL(texts));
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
    if (use_default_material || fld->data_at() != Field::EDGE)
    {
      m = default_material;
    }
    else
    {
      typename Fld::value_type val;
      fld->value(val, *iter);
      double dval;
      to_double(val, dval);
      m = color_handle->lookup(dval);
    }
    texts->add(buffer.str(), p, m->diffuse);
 
    ++iter;
  }
  return text_switch;
}

template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_text_faces(FieldHandle field_handle,
					 ColorMapHandle color_handle,
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
  GeomHandle text_switch = scinew GeomSwitch(scinew GeomDL(texts));
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
    if (use_default_material || fld->data_at() != Field::FACE)
    {
      m = default_material;
    }
    else
    {
      typename Fld::value_type val;
      fld->value(val, *iter);
      double dval;
      to_double(val, dval);
      m = color_handle->lookup(dval);
    }
    texts->add(buffer.str(), p, m->diffuse);

    ++iter;
  }
  return text_switch;
}


template <class Fld, class Loc>
GeomHandle 
RenderField<Fld, Loc>::render_text_cells(FieldHandle field_handle,
					 ColorMapHandle color_handle,
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
  GeomHandle text_switch = scinew GeomSwitch(scinew GeomDL(texts));
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
    if (use_default_material || fld->data_at() != Field::CELL)
    {
      m = default_material;
    }
    else
    {
      typename Fld::value_type val;
      fld->value(val, *iter);
      double dval;
      to_double(val, dval);
      m = color_handle->lookup(dval);
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

  virtual GeomHandle render_data(FieldHandle vfld_handle,
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
  virtual GeomHandle render_data(FieldHandle vfld_handle,
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
GeomHandle 
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
  GeomLines *lines;
  const bool lines_p = (display_mode == "Lines");
  const bool needles_p = (display_mode == "Needles");
  const bool cones_p = (display_mode == "Cones");
  const bool arrows_p = (display_mode == "Arrows");
  const bool disks_p = (display_mode == "Disks");
  GeomHandle data_switch;
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
      lines = scinew GeomLines();
    }
    else
    {
      lines = scinew GeomTranspLines();
    }

    data_switch =
      scinew GeomSwitch(scinew GeomColorMap(scinew GeomDL(scinew GeomMaterial(lines, default_material)), cmap));
  }

  MaterialHandle opaque = scinew Material(Color(1.0, 1.0, 1.0));
  opaque->transparency = 1.0;
  MaterialHandle transparent = scinew Material(Color(1.0, 1.0, 1.0));
  transparent->transparency = 0.0;

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
	    lines->add(p - tmp, ctmpd, p + tmp, ctmpd);
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
	    lines->add(p, ctmpd, p + tmp, ctmpd);
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

	lines->add(p, opaque, ctmpd, p + tmp, transparent, ctmpd);
	if (bidirectional)
	{
	  lines->add(p, opaque, ctmpd, p - tmp, transparent, ctmpd);
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
  virtual GeomHandle render_data(FieldHandle vfld_handle,
				 FieldHandle cfld_handle,
				 ColorMapHandle cmap,
				 MaterialHandle default_material,
				 const string &data_display_mode,
				 double scale,
				 int resolution);
};


template <class VFld, class CFld, class Loc>
GeomHandle 
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
  GeomHandle data_switch =
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

  virtual GeomHandle render_data(FieldHandle vfld_handle,
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
  virtual GeomHandle render_data(FieldHandle vfld_handle,
				 FieldHandle cfld_handle,
				 ColorMapHandle cmap,
				 MaterialHandle default_material,
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
						const string &display_mode,
						double scale, 
						int resolution,
						bool transparent_p)
{
  SFld *sfld = dynamic_cast<SFld*>(sfld_handle.get_rep());
  CFld *cfld = dynamic_cast<CFld*>(cfld_handle.get_rep()); 

  const bool points_p = (display_mode == "Points");
  const bool sized_p = (display_mode == "Scaled Spheres");

  GeomHandle data_switch = 0;
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
