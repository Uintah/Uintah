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
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Datatypes/Dispatch1.h>
#include <Core/Datatypes/DispatchMesh1.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/Switch.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/Pt.h>
#include <Core/GuiInterface/GuiVar.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Visualization/RenderField.h>

#include <typeinfo>
#include <map.h>
#include <iostream>

namespace SCIRun {

class ShowField : public Module 
{
  
  //! Private Data

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
  GuiInt                   nodes_on_;
  bool                     nodes_dirty_;
  //! edges.
  GuiInt                   edges_on_;
  bool                     edges_dirty_;
  //! faces.
  GuiInt                   faces_on_;
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
  DynamicAlgoHandle        renderer_;

public:
  ShowField(const string& id);
  virtual ~ShowField();
  virtual void execute();



  virtual void tcl_command(TCLArgs& args, void* userdata);

};

ShowField::ShowField(const string& id) : 
  Module("ShowField", id, Filter, "Visualization", "SCIRun"), 
  fld_(0),
  color_(0),
  color_handle_(0),
  fld_gen_(-69),
  colm_gen_(-1),
  ogeom_(0),
  node_id_(0),
  edge_id_(0),
  face_id_(0),
  nodes_on_("nodes-on", id, this),
  nodes_dirty_(true),
  edges_on_("edges-on", id, this),
  edges_dirty_(true),
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
  res_(0),
  renderer_(0)
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
    warning("No Data in port 1 field.");
    return;
  } else if (fld_gen_ != fld_handle->generation) {
    const TypeDescription *td = fld_handle->get_type_description();

    error(td->get_h_file_path().c_str());

    // Get the Algorithm.
    CompileInfo *ci = RenderFieldBase::get_compile_info(td);
    if (! DynamicLoader::scirun_loader().get(*ci, renderer_)) {
      fld_gen_ = -1;
      error("Could not compile algorithm for ShowField -");
      error(td->get_name().c_str());
      return;
    }
    
    if (renderer_.get_rep() == 0) {
      error("ShowField could not get algorithm!!");
      return;
    }
    fld_gen_ = fld_handle->generation;  
    nodes_dirty_ = true; edges_dirty_ = true; faces_dirty_ = true;
    MeshBaseHandle mh = fld_handle->mesh();
    mh->finish_mesh();  
  }
  
  color_->get(color_handle_);
  if(!color_handle_.get_rep()){
    warning("No ColorMap in port 3 ColorMap.");
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

  //dispatch1(fld_handle, render);
  //  if (disp_error) return; // dispatch already printed an error message. 

  node_display_type_.reset();
  string ndt = node_display_type_.get();
  node_scale_.reset();
  double ns = node_scale_.get();
  edge_display_type_.reset();
  string edt = edge_display_type_.get();
  edge_scale_.reset();
  double es = edge_scale_.get();

  RenderFieldBase* alg = dynamic_cast<RenderFieldBase*>(renderer_.get_rep());

  alg->render(fld_handle, nodes_dirty_, edges_dirty_, faces_dirty_, 
		    def_mat_handle_, use_def_color_, color_handle_,
		    ndt, edt, ns, es, res_);

  // cleanup...
  if (nodes_dirty_) {
    if (node_id_) ogeom_->delObj(node_id_);
    node_id_ = 0;
    nodes_dirty_ = false;
    if (alg && nodes_on_.get()) 
      node_id_ = ogeom_->addObj(alg->node_switch_, "Nodes");
  }
  if (edges_dirty_) {
    if (edge_id_) ogeom_->delObj(edge_id_); 
    edge_id_ = 0;
    edges_dirty_ = false;
    if (alg && edges_on_.get()) 
      edge_id_ = ogeom_->addObj(alg->edge_switch_, "Edges");

  }
  if (faces_dirty_) {
    if (face_id_) ogeom_->delObj(face_id_); 
    face_id_ = 0;
    faces_dirty_ = false;
    if (alg && faces_on_.get()) 
      face_id_ = ogeom_->addObj(alg->face_switch_, "Faces");
  }  
  ogeom_->flushViews();
}


void 
ShowField::tcl_command(TCLArgs& args, void* userdata) {
  if(args.count() < 2){
    args.error("ShowField needs a minor command");
    return;
  }
  RenderFieldBase* alg = dynamic_cast<RenderFieldBase*>(renderer_.get_rep());
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
    if (alg && alg->node_switch_) 
      alg->node_switch_->set_state(nodes_on_.get());

    if ((nodes_on_.get()) && (node_id_ == 0)) {
      nodes_dirty_ = true;
      want_to_execute();
    } else {
      ogeom_->flushViews();
    }
  } else if (args[1] == "toggle_display_edges"){
    // Toggle the GeomSwitch.
    edges_on_.reset();
    if (alg && alg->edge_switch_) 
      alg->edge_switch_->set_state(edges_on_.get());
    
    if ((edges_on_.get()) && (edge_id_ == 0)) {
      edges_dirty_ = true;
      want_to_execute();
    } else {
      ogeom_->flushViews();
    }
  } else if (args[1] == "toggle_display_faces"){
    // Toggle the GeomSwitch.
    faces_on_.reset();
    if (alg && alg->face_switch_) 
      alg->face_switch_->set_state(faces_on_.get());

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


