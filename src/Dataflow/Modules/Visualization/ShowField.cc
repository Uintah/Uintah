/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  ShowField.cc
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *   Aug 31, 2000
 *
 *  Copyright (C) 2000 SCI Group
 */
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Datatypes/Dispatch1.h>
#include <Core/Datatypes/DispatchMesh1.h>

#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/Switch.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/Pt.h>

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Util/DebugStream.h>

#include <typeinfo>
#include <map.h>
#include <iostream>
using std::cerr;
using std::cin;
using std::endl;
#include <sstream>
using std::ostringstream;

namespace SCIRun {

class ShowField : public Module 
{
  
  //! Private Data
  DebugStream              dbg_;  

  //! input ports
  FieldIPort*              fld_;
  ColorMapIPort*           color_;
  ColorMapHandle           color_handle_;
  int                      fld_gen_;
  int                      colm_gen_;
  //! output port
  GeometryOPort           *ogeom_;  

  //! Scene graph ID's
  int                      node_id_;
  int                      edge_id_;
  int                      face_id_;

  //! top level nodes for switching on and off..
  //! nodes.
  GeomSwitch*              node_switch_;
  GuiInt                  nodes_on_;
  bool                     nodes_dirty_;
  //! edges.
  GeomSwitch*              edge_switch_;
  GuiInt                  edges_on_;
  bool                     edges_dirty_;
  //! faces.
  GeomSwitch*              face_switch_;
  GuiInt                  faces_on_;
  bool                     faces_dirty_;

  //! default color and material
  bool                     use_def_color_;
  GuiDouble                def_color_r_;
  GuiDouble                def_color_g_;
  GuiDouble                def_color_b_;
  MaterialHandle           def_mat_handle_;

  //! holds options for how to visualize nodes.
  GuiString                node_display_type_;
  GuiString                edge_display_type_;
  GuiDouble                node_scale_;
  GuiDouble                edge_scale_;
  GuiInt                   showProgress_;

  //! Refinement resolution for cylinders and spheres
  GuiInt                   resolution_;
  int                      res_;

  //! Private Methods
  template <class T> bool to_double(const T&, double &) const;
  template <class Msh> void finish_mesh(Msh* m);
  template <class Fld> void render_nodes(const Fld *sfld);
  template <class Fld> void render_edges(const Fld *sfld);
  template <class Fld> void render_faces(const Fld *sfld);
  template <class F1>  void render(const F1 *fld);

  inline  MaterialHandle choose_mat(bool def, double val);
public:
  ShowField(const string& id);
  virtual ~ShowField();
  virtual void execute();

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

  virtual void tcl_command(TCLArgs& args, void* userdata);

};

template <class Msh>
void 
ShowField::finish_mesh(Msh*) {}

template <>
void 
ShowField::finish_mesh(TetVolMesh* tvm) { tvm->finish_mesh(); }

ShowField::ShowField(const string& id) : 
  Module("ShowField", id, Filter, "Visualization", "SCIRun"), 
  dbg_("ShowField", true),
  fld_(0),
  color_(0),
  color_handle_(0),
  fld_gen_(-1),
  colm_gen_(-1),
  ogeom_(0),
  node_id_(0),
  edge_id_(0),
  face_id_(0),
  node_switch_(0),
  nodes_on_("nodes-on", id, this),
  nodes_dirty_(true),
  edge_switch_(0),
  edges_on_("edges-on", id, this),
  edges_dirty_(true),
  face_switch_(0),
  faces_on_("faces-on", id, this),
  faces_dirty_(true),
  use_def_color_(true),
  def_color_r_("def-color-r", id, this),
  def_color_g_("def-color-g", id, this),
  def_color_b_("def-color-b", id, this),
  def_mat_handle_(scinew Material(Color(0.5, 0.5, 0.5))),
  node_display_type_("node_display_type", id, this),
  edge_display_type_("edge_display_type", id, this),
  node_scale_("node_scale", id, this),
  edge_scale_("edge_scale", id, this),
  showProgress_("show_progress", id, this),
  resolution_("resolution", id, this),
  res_(0)
 {
  // Create the input ports
  fld_ = scinew FieldIPort(this, "Field", FieldIPort::Atomic);
  add_iport(fld_);
  color_ = scinew ColorMapIPort(this, "ColorMap", FieldIPort::Atomic);
  add_iport(color_);
    
  // Create the output port
  ogeom_ = scinew GeometryOPort(this, "Scene Graph", GeometryIPort::Atomic);
  add_oport(ogeom_);
 }

ShowField::~ShowField() {}

void 
ShowField::execute()
{
  // tell module downstream to delete everything we have sent it before.
  // This is typically viewer, it owns the scene graph memory we create here.
  FieldHandle fld_handle;
  fld_->get(fld_handle);
  if(!fld_handle.get_rep()){
    cerr << "No Data in port 1 field" << endl;
    return;
  } else if (fld_gen_ != fld_handle->generation) {
    fld_gen_ = fld_handle->generation;  
    nodes_dirty_ = true; edges_dirty_ = true; faces_dirty_ = true;
    MeshBaseHandle mh = fld_handle->mesh();
    dispatch_mesh1(mh, finish_mesh);
  }

  color_->get(color_handle_);
  if(!color_handle_.get_rep()){
    cerr << "No ColorMap in port 3 ColorMap" << endl;
    if (colm_gen_ != -1) {
      nodes_dirty_ = true; edges_dirty_ = true; faces_dirty_ = true;
    }
    colm_gen_ = -1;
  } else if (colm_gen_ != color_handle_->generation) {
    colm_gen_ = color_handle_->generation;  
    nodes_dirty_ = true; edges_dirty_ = true; faces_dirty_ = true;
  }

  if (resolution_.get() != res_) {
    nodes_dirty_ = true;
    edges_dirty_ = true;
  }
  res_ = resolution_.get();

  // check to see if we have something to do.
  if ((!nodes_dirty_) && (!edges_dirty_) && (!faces_dirty_))  { return; }

  use_def_color_ = ! fld_handle->is_scalar();

  dispatch1(fld_handle, render);
  if (disp_error) return; // dispatch already printed an error message. 

  // cleanup...
  if (nodes_dirty_) {
    if (node_id_) ogeom_->delObj(node_id_);
    node_id_ = 0;
    nodes_dirty_ = false;
    if (nodes_on_.get()) node_id_ = ogeom_->addObj(node_switch_, "Nodes");
  }
  if (edges_dirty_) {
    if (edge_id_) ogeom_->delObj(edge_id_); 
    edge_id_ = 0;
    edges_dirty_ = false;
    if (edges_on_.get()) edge_id_ = ogeom_->addObj(edge_switch_, "Edges");

  }
  if (faces_dirty_) {
    if (face_id_) ogeom_->delObj(face_id_); 
    face_id_ = 0;
    faces_dirty_ = false;
    if (faces_on_.get()) face_id_ = ogeom_->addObj(face_switch_, "Faces");
  }  
  ogeom_->flushViews();
}

template <class T>
bool
ShowField::to_double(const T& tmp, double &val) const
{
  val = tmp;
  return true;
}

template <>
bool
ShowField::to_double(const Vector&, double &) const
{
  return false;
}

template <>
bool
ShowField::to_double(const Tensor&, double &) const
{
  return false;
}

inline
MaterialHandle 
ShowField::choose_mat(bool def, double val) {
  if (def) return def_mat_handle_;
  return color_handle_->lookup(val);
}

template <class Fld>
void 
ShowField::render_nodes(const Fld *sfld) 
{
  if (nodes_dirty_) {
    typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
    GeomGroup* nodes = scinew GeomGroup;
    node_switch_ = scinew GeomSwitch(nodes);
    GeomPts *pts = 0;
    node_display_type_.reset();
    if (node_display_type_.get() == "Points") {
      pts = scinew GeomPts(mesh->nodes_size());
    }
    // First pass: over the nodes
    typename Fld::mesh_type::node_iterator niter = mesh->node_begin();  
    while (niter != mesh->node_end()) {
      // Use a default color?
      bool def_color = (use_def_color_ || (color_handle_.get_rep() == 0));
    
      Point p;
      mesh->get_point(p, *niter);

      // val is double because the color index field must be scalar.
      double val = 0.L;
 
      switch (sfld->data_at()) {
      case Field::NODE:
	{
	  typename Fld::value_type tmp = 0;
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

      node_scale_.reset();
      if (node_display_type_.get() == "Spheres") {
	add_sphere(p, node_scale_.get(), nodes, 
		   choose_mat(def_color, val));
      } else if (node_display_type_.get() == "Axes") {
	add_axis(p, node_scale_.get(), nodes, choose_mat(def_color, val));
      } else {
	add_point(p, pts, choose_mat(def_color, val));
      }
      ++niter;
    }
    if (node_display_type_.get() == "Points") {
      nodes->add(pts);
    }
  }
}


template <class Fld>
void 
ShowField::render_edges(const Fld *sfld) 
{
  if (edges_dirty_) { 
    typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
    GeomGroup* edges = scinew GeomGroup;
    edge_switch_ = scinew GeomSwitch(edges);
    // Second pass: over the edges
    typename Fld::mesh_type::edge_iterator eiter = mesh->edge_begin();  
    while (eiter != mesh->edge_end()) {  
      // Use a default color?
      bool def_color = (use_def_color_ || (color_handle_.get_rep() == 0));

      typename Fld::mesh_type::node_array nodes;
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
	  typename Fld::value_type tmp1 = 0;
	  typename Fld::value_type tmp2 = 0;
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
	  typename Fld::value_type tmp = 0;
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
      if (edge_display_type_.get() == "Cylinders") { cyl = true; }
      add_edge(p1, p2, edge_scale_.get(), edges, 
	       choose_mat(def_color, val_avg), cyl);
    }
  }
}


template <class Fld>
void 
ShowField::render_faces(const Fld *sfld) 
{
  if (faces_dirty_) {
    typename Fld::mesh_handle_type mesh = sfld->get_typed_mesh();
    GeomTriangles* faces = scinew GeomTriangles;
    face_switch_ = scinew GeomSwitch(faces);
    // Third pass: over the faces
    typename Fld::mesh_type::face_iterator fiter = mesh->face_begin();  
    while (fiter != mesh->face_end()) {  
      // Use a default color?
      bool def_color = (use_def_color_ || (color_handle_.get_rep() == 0));

      typename Fld::mesh_type::node_array nodes;
      mesh->get_nodes(nodes, *fiter); ++fiter;
      
      Point p1, p2, p3;
      mesh->get_point(p1, nodes[0]);
      mesh->get_point(p2, nodes[1]);
      mesh->get_point(p3, nodes[2]);
      double val1 = 0.L;
      double val2 = 0.L;
      double val3 = 0.L;

      switch (sfld->data_at()) {
      case Field::NODE:
	{
	  typename Fld::value_type tmp1 = 0;
	  typename Fld::value_type tmp2 = 0;
	  typename Fld::value_type tmp3 = 0;
	  if (! (sfld->value(tmp1, nodes[0]) && to_double(tmp1, val1) &&
		 sfld->value(tmp2, nodes[1]) && to_double(tmp2, val2) &&
		 sfld->value(tmp3, nodes[2]) && to_double(tmp3, val3))) { 
	    def_color = true; 
	  }
	}
	break;
      case Field::FACE: 
	{
	  typename Fld::value_type tmp = 0;
	  if (! (sfld->value(tmp, *fiter) && to_double(tmp, val1) && 
		 to_double(tmp, val2) && to_double(tmp, val3))) {
	    def_color = true; 
	  }
	}
	break;

      case Field::EDGE:
      case Field::CELL:
      case Field::NONE:
	def_color = true;
	break;
      }
      add_face(p1, p2, p3, 
	       choose_mat(def_color, val1), 
	       choose_mat(def_color, val2), 
	       choose_mat(def_color, val3), 
	       faces);
    }
  }
}

template <class F1>
void
ShowField::render(const F1 *fld)
{
  render_nodes(fld);
  render_edges(fld);
  render_faces(fld);
}

void 
ShowField::add_face(const Point &p0, const Point &p1, const Point &p2, 
		    MaterialHandle m0, MaterialHandle m1, MaterialHandle m2,
		    GeomTriangles *g) 
{
  g->add(p0, m0, 
	 p1, m1, 
	 p2, m2);
}

void 
ShowField::add_edge(const Point &p0, const Point &p1,  
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

void 
ShowField::add_sphere(const Point &p0, double scale, 
		      GeomGroup *g, MaterialHandle mh) {
  GeomSphere *s = scinew GeomSphere(p0, scale, res_, res_);
  g->add(scinew GeomMaterial(s, mh));
}

void 
ShowField::add_axis(const Point &p0, double scale, 
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

void 
ShowField::add_point(const Point &p, GeomPts *pts, MaterialHandle mh) {
  pts->add(p, mh->diffuse);
}

void 
ShowField::tcl_command(TCLArgs& args, void* userdata) {
  if(args.count() < 2){
    args.error("ShowField needs a minor command");
    return;
  }

  if (args[1] == "node_scale") {
    if (node_display_type_.get() == "Points") { return; }
    nodes_dirty_ = true;
  } else if (args[1] == "edge_scale") {
    edges_dirty_ = true;
  } else if (args[1] == "default_color_change") {
    def_color_r_.reset();
    def_color_g_.reset();
    def_color_b_.reset();
    Material m(Color(def_color_r_.get(), def_color_g_.get(), 
		     def_color_b_.get()));
    *def_mat_handle_.get_rep() = m;
    ogeom_->flushViews();
  } else if (args[1] == "node_display_type") {
    nodes_dirty_ = true;
    want_to_execute();
  } else if (args[1] == "edge_display_type") {
    edges_dirty_ = true;
    want_to_execute();
  } else if (args[1] == "toggle_display_nodes"){
    // Toggle the GeomSwitches.
    nodes_on_.reset();
    if (node_switch_) node_switch_->set_state(nodes_on_.get());

    if ((nodes_on_.get()) && (node_id_ == 0)) {
      nodes_dirty_ = true;
      want_to_execute();
    } else {
      ogeom_->flushViews();
    }
  } else if (args[1] == "toggle_display_edges"){
    // Toggle the GeomSwitch.
    edges_on_.reset();
    if (edge_switch_) edge_switch_->set_state(edges_on_.get());
    
    if ((edges_on_.get()) && (edge_id_ == 0)) {
      edges_dirty_ = true;
      want_to_execute();
    } else {
      ogeom_->flushViews();
    }
  } else if (args[1] == "toggle_display_faces"){
    // Toggle the GeomSwitch.
    faces_on_.reset();
    if (face_switch_) face_switch_->set_state(faces_on_.get());

    if ((faces_on_.get()) && (face_id_ == 0)) {
      faces_dirty_ = true;
      want_to_execute();
    } else {
      ogeom_->flushViews();
    }
  } else {
    Module::tcl_command(args, userdata);
  }
}

extern "C" Module* make_ShowField(const string& id) {
  return new ShowField(id);
}

} // End namespace SCIRun


