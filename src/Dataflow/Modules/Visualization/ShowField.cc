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
#include <Core/Datatypes/FieldAlgo.h>

#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/Switch.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/Pt.h>

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Util/DebugStream.h>

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
  FieldIPort*              geom_;
  FieldIPort*              data_;
  ColorMapIPort*           color_;
  int                      geom_gen_;
  int                      data_gen_;
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
  bool                     nodes_on_;
  bool                     nodes_dirty_;
  //! edges.
  GeomSwitch*              edge_switch_;
  bool                     edges_on_;
  bool                     edges_dirty_;
  //! faces.
  GeomSwitch*              face_switch_;
  bool                     faces_on_;
  bool                     faces_dirty_;

  //! default grey material
  Material                 def_mat_;
  MaterialHandle           def_mat_handle_;
  //! holds options for how to visualize nodes.
  GuiString                node_display_type_;
  GuiDouble                node_scale_;
  GuiInt                   showProgress_;

  //! Private Methods
  template <class Field>
  bool render(Field *f, Field *f1, ColorMapHandle cm);

  inline
  MaterialHandle choose_mat(bool def, ColorMapHandle cm, double val);
public:
  ShowField(const clString& id);
  virtual ~ShowField();
  virtual void execute();

  inline void add_sphere(const Point &p, double scale, GeomGroup *g, 
			 MaterialHandle m0);
  inline void add_axis(const Point &p, double scale, GeomGroup *g, 
		       MaterialHandle m0);
  inline void add_point(const Point &p, GeomPts *g, 
			MaterialHandle m0);
  inline void add_edge(const Point &p1, const Point &p2, double scale, 
		       GeomGroup *g, MaterialHandle m0);
  inline void add_face(const Point &p1, const Point &p2, const Point &p3, 
		       MaterialHandle m0, MaterialHandle m1, MaterialHandle m2,
		       GeomGroup *g);

  virtual void tcl_command(TCLArgs& args, void* userdata);

};

ShowField::ShowField(const clString& id) : 
  Module("ShowField", id, Filter), 
  dbg_("ShowField", true),
  geom_(0),
  data_(0),
  color_(0),
  geom_gen_(0),
  data_gen_(0),
  colm_gen_(0),
  ogeom_(0),
  node_id_(0),
  edge_id_(0),
  face_id_(0),
  node_switch_(0),
  nodes_on_(true),
  nodes_dirty_(true),
  edge_switch_(0),
  edges_on_(true),
  edges_dirty_(true),
  face_switch_(0),
  faces_on_(true),
  faces_dirty_(true),
  def_mat_(Color(.5, .5, .5)),
  def_mat_handle_(&def_mat_),
  node_display_type_("node_display_type", id, this),
  node_scale_("scale", id, this),
  showProgress_("show_progress", id, this)
 {
  // Create the input ports
  geom_ = scinew FieldIPort(this, "Field-Geometry", FieldIPort::Atomic);
  add_iport(geom_);
  data_ = scinew FieldIPort(this, "Field-ColorIndex", FieldIPort::Atomic);
  add_iport(data_);
  color_ = scinew ColorMapIPort(this, "ColorMap", FieldIPort::Atomic);
  add_iport(color_);
    
  // Create the output port
  ogeom_ = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom_);
 }

ShowField::~ShowField() {}

void 
ShowField::execute()
{

  // tell module downstream to delete everything we have sent it before.
  // This is typically viewer, it owns the scene graph memory we create here.
  FieldHandle geom_handle;
  geom_->get(geom_handle);
  if(!geom_handle.get_rep()){
    cerr << "No Geometry in port 1 field" << endl;
    return;
  } else if (geom_gen_ != geom_handle->generation) {
    geom_gen_ = geom_handle->generation;  
    nodes_dirty_ = true; edges_dirty_ = true; faces_dirty_ = true;
  }

  FieldHandle data_handle;
  data_->get(data_handle);
  if(!data_handle.get_rep()){
    cerr << "No Data in port 2 field" << endl;
    if (data_gen_) {
      nodes_dirty_ = true; edges_dirty_ = true; faces_dirty_ = true;
    }
    data_gen_ = -1;
  } else if (data_gen_ != data_handle->generation) {
    data_gen_ = data_handle->generation;
    nodes_dirty_ = true; edges_dirty_ = true; faces_dirty_ = true;
  }

  ColorMapHandle color_handle;
  color_->get(color_handle);
  if(!color_handle.get_rep()){
    cerr << "No ColorMap in port 3 ColorMap" << endl;
    if (colm_gen_) {
      nodes_dirty_ = true; edges_dirty_ = true; faces_dirty_ = true;
    }
    colm_gen_ = -1;
  } else if (colm_gen_ != color_handle->generation) {
    colm_gen_ = color_handle->generation;
    nodes_dirty_ = true; edges_dirty_ = true; faces_dirty_ = true;
  }

  // check to see if we have something to do.
  if ((!nodes_dirty_) && (!edges_dirty_) && (!faces_dirty_))  { return; }

  bool error = false;
  string msg;
  string name = geom_handle->get_type_name(0);
  if (name == "TetVol") {
    if (geom_handle->get_type_name(1) == "double") {
      TetVol<double> *tv = 0;
      TetVol<double> *tv1 = 0;
      tv = dynamic_cast<TetVol<double>*>(geom_handle.get_rep());
      if (data_gen_) {
	tv1 = dynamic_cast<TetVol<double>*>(data_handle.get_rep());
      }
      if (tv) { 
	// need faces and edges.
	tv->finish_mesh();
	render(tv, tv1, color_handle); 
      }
      else { error = true; msg = "Not a valid TetVol."; }
    } else {
      error = true; msg ="TetVol of unknown type.";
    }
  } else if (name == "LatticeVol") {
    if (geom_handle->get_type_name(1) == "double") {
      LatticeVol<double> *lv = 0;
      LatticeVol<double> *tv1 = 0;
      lv = dynamic_cast<LatticeVol<double>*>(geom_handle.get_rep());
      if (data_gen_) {
	tv1 = dynamic_cast<LatticeVol<double>*>(data_handle.get_rep());
      }
      if (lv)
	render(lv, tv1, color_handle);
      else { error = true; msg = "Not a valid LatticeVol."; }
    } else {
      error = true; msg ="LatticeVol of unknown type.";
    }
  } else if (name == "ContourField") {
    if (geom_handle->get_type_name(1) == "double") {
      ContourField<double> *cf = 0;
      ContourField<double> *tv1 = 0;
      cf = dynamic_cast<ContourField<double>*>(geom_handle.get_rep());
      if (data_gen_) {
	tv1 = dynamic_cast<ContourField<double>*>(data_handle.get_rep());
      }
      if (cf)
	render(cf, tv1, color_handle);
      else { error = true; msg = "Not a valid ContourField."; }
    } else {
      error = true; msg ="ContourField of unknown type.";
    }
  } else if (name == "TriSurf") {
    if (geom_handle->get_type_name(1) == "double") {
      TriSurf<double> *ts = 0;
      TriSurf<double> *ts1 = 0;
      ts = dynamic_cast<TriSurf<double>*>(geom_handle.get_rep());
      if (data_gen_) {
	ts1 = dynamic_cast<TriSurf<double>*>(data_handle.get_rep());
      }
      if (ts)
	render(ts, ts1, color_handle);
      else { error = true; msg = "Not a valid TriSurf."; }
    } else {
      error = true; msg ="ContourField of unknown type.";
    }
  } else if (error) {
    cerr << "ShowField Error: " << msg << endl;
    return;
  }
  
  // cleanup...
  if (nodes_dirty_) {
    if (node_id_) ogeom_->delObj(node_id_);
    node_id_ = 0;
    nodes_dirty_ = false;
    if (nodes_on_) node_id_ = ogeom_->addObj(node_switch_, "Nodes");
  }
  if (edges_dirty_) {
    if (edge_id_) ogeom_->delObj(edge_id_); 
    edge_id_ = 0;
    edges_dirty_ = false;
    if (edges_on_) edge_id_ = ogeom_->addObj(edge_switch_, "Edges");

  }
  if (faces_dirty_) {
    if (face_id_) ogeom_->delObj(face_id_); 
    face_id_ = 0;
    faces_dirty_ = false;
    if (faces_on_) face_id_ = ogeom_->addObj(face_switch_, "Faces");
  }  

  
  ogeom_->flushViews();
}

inline
MaterialHandle 
ShowField::choose_mat(bool def, ColorMapHandle cm, double val) {
  if (def) return def_mat_handle_;
  return cm->lookup(val);
}

template <class Field>
bool
ShowField::render(Field *geom, Field *c_index, ColorMapHandle cm) 
{
  typename Field::mesh_handle_type mesh = geom->get_typed_mesh();

  if (nodes_dirty_) {
    GeomGroup* nodes = scinew GeomGroup;
    node_switch_ = scinew GeomSwitch(nodes);
    GeomPts *pts = 0;
    node_display_type_.reset();
    if (node_display_type_.get() == "Points") {
      pts = scinew GeomPts(mesh->nodes_size());
    }
    // First pass: over the nodes
    typename Field::mesh_type::node_iterator niter = mesh->node_begin();  
    while (niter != mesh->node_end()) {
      // Use a default color?
      bool def_color = (c_index == 0);
    
      Point p;
      mesh->get_point(p, *niter);
      // val is double because the color index field must be scalar.
      double val = 0.L;
      if ((c_index) && (! c_index->value(val, *niter))) { def_color = true; }

      node_scale_.reset();
      if (node_display_type_.get() == "Spheres") {
	add_sphere(p, node_scale_.get(), nodes, 
		   choose_mat(def_color, cm, val));
      } else if (node_display_type_.get() == "Axes") {
	add_axis(p, node_scale_.get(), nodes, choose_mat(def_color, cm, val));
      } else {
	add_point(p, pts, choose_mat(def_color, cm, val));
      }
      ++niter;
    }
    if (node_display_type_.get() == "Points") {
      nodes->add(pts);
    }
  }

  if (edges_dirty_) { 
    GeomGroup* edges = scinew GeomGroup;
    edge_switch_ = scinew GeomSwitch(edges);
    // Second pass: over the edges
    typename Field::mesh_type::edge_iterator eiter = mesh->edge_begin();  
    while (eiter != mesh->edge_end()) {  
      // Use a default color?
      bool def_color = (c_index == 0);

      typename Field::mesh_type::node_array nodes;
      mesh->get_nodes(nodes, *eiter); ++eiter;
    
      Point p1, p2;
      mesh->get_point(p1, nodes[0]);
      mesh->get_point(p2, nodes[1]);
      double val1 = 0.L;
      if ((c_index) && (! c_index->value(val1, nodes[0]))) { 
	def_color = true; 
      }
      double val2 = 0.L;
      if ((c_index) && (! c_index->value(val2, nodes[1]))) { 
	def_color = true; 
      }
      val1 = (val1 + val2) * .5;
      add_edge(p1, p2, 1.0, edges, choose_mat(def_color, cm, val1));
    }
  }

  if (faces_dirty_) {
    GeomGroup* faces = scinew GeomGroup;
    face_switch_ = scinew GeomSwitch(faces);
    // Third pass: over the faces
    typename Field::mesh_type::face_iterator fiter = mesh->face_begin();  
    while (fiter != mesh->face_end()) {  
      // Use a default color?
      bool def_color = (c_index == 0);

      typename Field::mesh_type::node_array nodes;
      mesh->get_nodes(nodes, *fiter); ++fiter;
      
      Point p1, p2, p3;
      mesh->get_point(p1, nodes[0]);
      mesh->get_point(p2, nodes[1]);
      mesh->get_point(p3, nodes[2]);
      double val1 = 0.L;
      if ((c_index) && (! c_index->value(val1, nodes[0]))) { 
	def_color = true; 
      }
      double val2 = 0.L;
      if ((c_index) && (! c_index->value(val2, nodes[1]))) { 
	def_color = true; 
      }
      double val3 = 0.L;
      if ((c_index) && (! c_index->value(val3, nodes[2]))) { 
	def_color = true; 
      }

      add_face(p1, p2, p3, 
	       choose_mat(def_color, cm, val1), 
	       choose_mat(def_color, cm, val2), 
	       choose_mat(def_color, cm, val3), 
	       faces);
    }
  }
  return true;
}

void 
ShowField::add_face(const Point &p0, const Point &p1, const Point &p2, 
		    MaterialHandle m0, MaterialHandle m1, MaterialHandle m2,
		    GeomGroup *g) 
{
  GeomTri *t = new GeomTri(p0, p1, p2, m0, m1, m2);
  g->add(t);
}

void 
ShowField::add_edge(const Point &p0, const Point &p1,  
		    double scale, GeomGroup *g, MaterialHandle mh) {
  GeomLine *l = new GeomLine(p0, p1);
  l->setLineWidth(scale);
  g->add(scinew GeomMaterial(l, mh));
}

void 
ShowField::add_sphere(const Point &p0, double scale, 
		      GeomGroup *g, MaterialHandle mh) {
  GeomSphere *s = scinew GeomSphere(p0, scale, 8, 4);
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

  if (args[1] == "scale") {
    if (node_display_type_.get() == "Points") { return; }
    nodes_dirty_ = true;
  } else if (args[1] == "node_display_type") {
    nodes_dirty_ = true;
    want_to_execute();
  } else if (args[1] == "toggle_display_nodes"){
    // Toggle the GeomSwitches.
    nodes_on_ = ! nodes_on_;
    if (node_switch_) node_switch_->set_state(nodes_on_);

    if ((nodes_on_) && (node_id_ == 0)) {
      nodes_dirty_ = true;
      want_to_execute();
    } else {
      ogeom_->flushViews();
    }
  } else if (args[1] == "toggle_display_edges"){
    // Toggle the GeomSwitch.
    edges_on_ = ! edges_on_;
    if (edge_switch_) edge_switch_->set_state(edges_on_);
    
    if ((edges_on_) && (edge_id_ == 0)) {
      edges_dirty_ = true;
      want_to_execute();
    } else {
      ogeom_->flushViews();
    }
  } else if (args[1] == "toggle_display_faces"){
    // Toggle the GeomSwitch.
    faces_on_ = ! faces_on_;
    if (face_switch_) face_switch_->set_state(faces_on_);

    if ((faces_on_) && (face_id_ == 0)) {
      faces_dirty_ = true;
      want_to_execute();
    } else {
      ogeom_->flushViews();
    }
  } else {
    Module::tcl_command(args, userdata);
  }
}

extern "C" Module* make_ShowField(const clString& id) {
  return new ShowField(id);
}

} // End namespace SCIRun


