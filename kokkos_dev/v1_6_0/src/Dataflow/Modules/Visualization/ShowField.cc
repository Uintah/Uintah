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
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/Switch.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/Pt.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Algorithms/Visualization/RenderField.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>


#include <typeinfo>
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
  int                      mesh_gen_;
  int                      colm_gen_;
  //! output port
  GeometryOPort           *ogeom_;  

  //! Scene graph ID's
  int                      node_id_;
  int                      edge_id_;
  int                      face_id_;
  int                      data_id_;

  //! top level nodes for switching on and off..
  //! Options for rendering nodes.
  GuiInt                   nodes_on_;
  GuiInt                   nodes_as_disks_;
  bool                     nodes_dirty_;
  //! Options for rendering edges.
  GuiInt                   edges_on_;
  bool                     edges_dirty_;
  //! Options for rendering faces.
  GuiInt                   use_normals_;
  GuiInt                   use_transparency_;
  GuiInt                   faces_on_;
  bool                     faces_dirty_;
  //! Options for rendering non-scalar data.
  GuiInt                   vectors_on_;
  GuiInt                   normalize_vectors_;
  GuiInt                   has_vec_data_;
  bool                     data_dirty_;
  
  //! default color and material
  GuiDouble                def_color_r_;
  GuiDouble                def_color_g_;
  GuiDouble                def_color_b_;
  GuiDouble                def_color_a_;
  MaterialHandle           def_mat_handle_;
#ifdef HAVE_HASH_MAP
  typedef hash_map<int, MaterialHandle> ind_mat_t;
#else
  typedef map<int, MaterialHandle> ind_mat_t;
#endif
  ind_mat_t                idx_mats_;

  //! re-render all the material handles?
  bool                     data_at_dirty_;

  //! holds options for how to visualize nodes.
  GuiString                node_display_type_;
  GuiString                edge_display_type_;
  GuiString                active_tab_; //! for saving nets state
  GuiDouble                node_scale_;
  GuiDouble                edge_scale_;
  GuiDouble                vectors_scale_;
  GuiInt                   showProgress_;
  GuiString                interactive_mode_;

  //! Refinement resolution for cylinders and spheres
  GuiInt                   resolution_;
  int                      res_;
  LockingHandle<RenderFieldBase>  renderer_;

  enum toggle_type_e {
    NODE = 0,
    EDGE,
    FACE,
    DATA,
    DATA_AT
  };
  vector<bool>               render_state_;
  void maybe_execute(toggle_type_e dis_type);
  Vector                  *bounding_vector_;
public:
  ShowField(GuiContext* ctx);
  virtual ~ShowField();
  virtual void execute();
  void check_for_vector_data(FieldHandle fld_handle);
  bool fetch_typed_algorithm(FieldHandle fld_handle);
  bool determine_dirty(FieldHandle fld_handle);
  virtual void tcl_command(GuiArgs& args, void* userdata);
};

ShowField::ShowField(GuiContext* ctx) : 
  Module("ShowField", ctx, Filter, "Visualization", "SCIRun"), 
  fld_(0),
  color_(0),
  color_handle_(0),
  fld_gen_(-1),
  mesh_gen_(-1),
  colm_gen_(-1),
  ogeom_(0),
  node_id_(0),
  edge_id_(0),
  face_id_(0),
  data_id_(0),
  nodes_on_(ctx->subVar("nodes-on")),
  nodes_as_disks_(ctx->subVar("nodes-as-disks")),
  nodes_dirty_(true),
  edges_on_(ctx->subVar("edges-on")),
  edges_dirty_(true),
  use_normals_(ctx->subVar("use-normals")),
  use_transparency_(ctx->subVar("use-transparency")),
  faces_on_(ctx->subVar("faces-on")),
  faces_dirty_(true),
  vectors_on_(ctx->subVar("vectors-on")),
  normalize_vectors_(ctx->subVar("normalize-vectors")),
  has_vec_data_(ctx->subVar("has_vec_data")),
  data_dirty_(true),
  def_color_r_(ctx->subVar("def-color-r")),
  def_color_g_(ctx->subVar("def-color-g")),
  def_color_b_(ctx->subVar("def-color-b")),
  def_color_a_(ctx->subVar("def-color-a")),
  def_mat_handle_(scinew Material(Color(0.5, 0.5, 0.5))),
  data_at_dirty_(true),
  node_display_type_(ctx->subVar("node_display_type")),
  edge_display_type_(ctx->subVar("edge_display_type")),
  active_tab_(ctx->subVar("active_tab")),
  node_scale_(ctx->subVar("node_scale")),
  edge_scale_(ctx->subVar("edge_scale")),
  vectors_scale_(ctx->subVar("vectors_scale")),
  showProgress_(ctx->subVar("show_progress")),
  interactive_mode_(ctx->subVar("interactive_mode")),
  resolution_(ctx->subVar("resolution")),
  res_(0),
  renderer_(0),
  render_state_(5),
  bounding_vector_(0)
{
  def_mat_handle_->transparency = 0.5;
  nodes_on_.reset();
  render_state_[NODE] = nodes_on_.get();
  edges_on_.reset();
  render_state_[EDGE] = edges_on_.get();
  faces_on_.reset();
  render_state_[FACE] = faces_on_.get();
  vectors_on_.reset();
  render_state_[DATA] = vectors_on_.get();
}

ShowField::~ShowField() {}

void
ShowField::check_for_vector_data(FieldHandle fld_handle) {
  // Test for vector data possibility
  has_vec_data_.reset();
  nodes_as_disks_.reset();
  if (fld_handle->query_vector_interface() != 0) {
    if (! has_vec_data_.get()) { 
      has_vec_data_.set(1); 
    }
    if (fld_handle->data_at() == Field::NODE && nodes_as_disks_.get() == 0) {
      nodes_as_disks_.set(1); 
    }
  } else if (nodes_as_disks_.get() == 1) {
    nodes_as_disks_.set(0);
  }
}


bool
ShowField::fetch_typed_algorithm(FieldHandle fld_handle) 
{
  const TypeDescription *ftd = fld_handle->get_type_description();
  const TypeDescription *ltd = fld_handle->data_at_type_description();

  // Get the Algorithm.
  CompileInfo *ci = RenderFieldBase::get_compile_info(ftd, ltd);
  if (!module_dynamic_compile(*ci, renderer_))
  {
    fld_gen_ = -1;
    mesh_gen_ = -1;
    return false;
  }
  return true;
}


bool
ShowField::determine_dirty(FieldHandle fld_handle) 
{
  bool mesh_new = fld_handle->mesh()->generation != mesh_gen_;
  bool field_new = fld_handle->generation != fld_gen_;

  if (mesh_new) {
    // completely new, all dirty, or just new geometry, so data_at invalid too.
    check_for_vector_data(fld_handle);
    if (!fetch_typed_algorithm(fld_handle)) { return false; }
    fld_gen_  = fld_handle->generation;  
    mesh_gen_ = fld_handle->mesh()->generation; 
    nodes_dirty_ = true; 
    edges_dirty_ = true; 
    faces_dirty_ = true; 
    data_dirty_ = true;
    data_at_dirty_ = true;
    Material *m = scinew Material(Color(def_color_r_.get(), def_color_g_.get(),
					def_color_b_.get()));
    m->transparency = def_color_a_.get();
    def_mat_handle_ = m;
    if (node_id_) ogeom_->delObj(node_id_);
    node_id_ = 0;
    if (edge_id_) ogeom_->delObj(edge_id_);
    edge_id_ = 0;
    if (face_id_) ogeom_->delObj(face_id_);
    face_id_ = 0;
    if (data_id_) ogeom_->delObj(data_id_);
    data_id_ = 0;

    if (bounding_vector_) delete bounding_vector_;
    bounding_vector_ = scinew Vector();
    *bounding_vector_ = fld_handle->mesh()->get_bounding_box().diagonal();
    
  } else if (!mesh_new && field_new) {
    // same geometry, new data.
    check_for_vector_data(fld_handle);
    //if (!fetch_typed_algorithm(fld_handle)) { return false; }
    data_at_dirty_ = true; //we need to rerender colors..
  } //else both are the same as last time, nothing dirty.
  
  return true;
}

void 
ShowField::execute()
{
  // tell module downstream to delete everything we have sent it before.
  // This is typically viewer, it owns the scene graph memory we create here.
  fld_ = (FieldIPort *)get_iport("Field");
  color_ = (ColorMapIPort *)get_iport("ColorMap");
  ogeom_ = (GeometryOPort *)get_oport("Scene Graph");
  FieldHandle fld_handle;

  if (!fld_) {
    error("Unable to initialize iport 'Field'.");
    return;
  }
  if (!color_) {
    error("Unable to initialize iport 'ColorMap'.");
    return;
  }
  if (!ogeom_) {
    error("Unable to initialize oport 'Scene Graph'.");
    return;
  }

  fld_->get(fld_handle);
  if(!fld_handle.get_rep()){
    warning("No Data in port 1 field.");
    return;
  } else {
    // What has changed from last time?  A false return value means that we 
    // could not load the algorithm from the dynamic loader.
    if (! determine_dirty(fld_handle)) { return; }
  }
  
  // if no colormap was attached, the argument doesn't get changed,
  // so we need to set it manually
  // if the user had a colormap attached for a previous execution,
  // and then detached it, we need to set color_handle_ to be empty

  if(!color_->get(color_handle_)) color_handle_=0;

  if(!color_handle_.get_rep()){
    //warning("No ColorMap in port 2 ColorMap.");
    if (colm_gen_ != -1) {
      data_at_dirty_ = true;
    }
    colm_gen_ = -1;
  } else if (colm_gen_ != color_handle_->generation) {
    colm_gen_ = color_handle_->generation;  
    data_at_dirty_ = true;
  }

  if (resolution_.get() != res_) {
    nodes_dirty_ = true;
    edges_dirty_ = true;
  }
  res_ = resolution_.get();

  // check to see if we have something to do.
  if ((!nodes_dirty_) && (!edges_dirty_) && 
      (!faces_dirty_) && (!data_dirty_) && (!data_at_dirty_))  { 
    return; 
  }

  node_display_type_.reset();
  string ndt = node_display_type_.get();
  node_scale_.reset();
  double ns = node_scale_.get();
  edge_display_type_.reset();
  string edt = edge_display_type_.get();
  edge_scale_.reset();
  double es = edge_scale_.get();
  vectors_scale_.reset();
  double vs = vectors_scale_.get();

  nodes_on_.reset();
  edges_on_.reset();
  faces_on_.reset();
  vectors_on_.reset();
  bool do_nodes = nodes_on_.get() && nodes_dirty_;
  bool do_edges = edges_on_.get() && edges_dirty_;
  bool do_faces = faces_on_.get() && faces_dirty_;
  bool do_data  = vectors_on_.get() && data_dirty_;

  if (render_state_[NODE] != nodes_on_.get()) {
    if (node_id_) ogeom_->delObj(node_id_);
    node_id_ = 0;
    render_state_[NODE] = nodes_on_.get();
  } 
  if (render_state_[EDGE] != edges_on_.get()) {
    if (edge_id_) ogeom_->delObj(edge_id_);
    edge_id_ = 0;
    render_state_[EDGE] = edges_on_.get();
  } 
  if (render_state_[FACE] != faces_on_.get()) {
    if (face_id_) ogeom_->delObj(face_id_);
    face_id_ = 0;
    render_state_[FACE] = faces_on_.get();
  }
  if (render_state_[DATA] != vectors_on_.get()) {
    if (data_id_) ogeom_->delObj(data_id_);
    data_id_ = 0;
    render_state_[DATA] = vectors_on_.get();
  }  

  normalize_vectors_.reset();
  if (renderer_.get_rep())
  {
    if (use_normals_.get()) fld_handle->mesh()->synchronize(Mesh::NORMALS_E);

    renderer_->set_mat_map(&idx_mats_);
    renderer_->render(fld_handle, 
		      do_nodes, do_edges, do_faces, do_data,
		      def_mat_handle_, data_at_dirty_, color_handle_,
		      ndt, edt, ns, es, vs, normalize_vectors_.get(), res_,
		      use_normals_.get(), use_transparency_.get());
  }

  // cleanup...
  if (do_nodes) {
    nodes_dirty_ = false;
    if (renderer_.get_rep() && nodes_on_.get()) {
      if (node_id_) ogeom_->delObj(node_id_);
      node_id_ = ogeom_->addObj(renderer_->node_switch_, "Nodes");
    }
  }
  if (do_edges) {
    edges_dirty_ = false;
    if (renderer_.get_rep() && edges_on_.get()) {
      if (edge_id_) ogeom_->delObj(edge_id_);
      edge_id_ = ogeom_->addObj(renderer_->edge_switch_, "Edges");
    }
  }
  if (do_faces) {
    faces_dirty_ = false;
    if (renderer_.get_rep() && faces_on_.get()) 
    {
      const char *name = use_transparency_.get()?"TransParent Faces":"Faces";
      if (face_id_) ogeom_->delObj(face_id_);
      face_id_ = ogeom_->addObj(renderer_->face_switch_, name);
    }
  }  
  if (do_data) {
    data_dirty_ = false;
    if (renderer_.get_rep() && vectors_on_.get()) {
      if (data_id_) ogeom_->delObj(data_id_);
      data_id_ = ogeom_->addObj(renderer_->data_switch_, "Vector Data");
    }
  }
  if (data_at_dirty_) { data_at_dirty_ = false; }

  ogeom_->flushViews();
}

void 
ShowField::maybe_execute(toggle_type_e dis_type) 
{
  bool do_execute = false;
  if (interactive_mode_.get() == "Interactive") {
    switch(dis_type) {
    case NODE :
      do_execute = nodes_on_.get();
	break;
    case EDGE :
      do_execute = edges_on_.get();
      break;
    case FACE :
      do_execute = faces_on_.get();
	break;
    case DATA :
      do_execute = vectors_on_.get();
	break;
    case DATA_AT :
      do_execute = true;
	break;
    }   
  }
  if (do_execute) {
    want_to_execute();
  }
}

void 
ShowField::tcl_command(GuiArgs& args, void* userdata) {
  if(args.count() < 2){
    args.error("ShowField needs a minor command");
    return;
  }
  bool now = false;
  interactive_mode_.reset();
  if (interactive_mode_.get() == "Interactive") now = true;
  if (args[1] == "node_scale") {
    if (node_display_type_.get() == "Points") { return; }
    nodes_dirty_ = true;
    maybe_execute(NODE);
  } else if (args[1] == "edge_scale") {
    edges_dirty_ = true;
    maybe_execute(EDGE);
  } else if (args[1] == "data_scale") {
    data_dirty_ = true;
    maybe_execute(DATA);
  } else if (args[1] == "default_color_change") {
    def_color_r_.reset();
    def_color_g_.reset();
    def_color_b_.reset();
    def_color_a_.reset();
    Material *m = scinew Material(Color(def_color_r_.get(), def_color_g_.get(),
					def_color_b_.get()));
    m->transparency = def_color_a_.get();
    def_mat_handle_ = m;
    data_at_dirty_ = true;
    maybe_execute(DATA_AT);
  } else if (args[1] == "node_display_type") {
    nodes_dirty_ = true;
    if (now && node_id_) {
      ogeom_->delObj(node_id_);
      node_id_ = 0;
    }
    maybe_execute(NODE);
  } else if (args[1] == "edge_display_type") {
    edges_dirty_ = true;
    if (now && edge_id_) {
      ogeom_->delObj(edge_id_);
      edge_id_ = 0;
    }
    maybe_execute(EDGE);
  } else if (args[1] == "toggle_display_nodes"){
    // Toggle the GeomSwitches.
    nodes_on_.reset();
    if (renderer_.get_rep() && renderer_->node_switch_ && now) 
      renderer_->node_switch_->set_state(nodes_on_.get());

    if ((nodes_on_.get()) && (node_id_ == 0)) {
      nodes_dirty_ = true;
      maybe_execute(NODE);
    } else {
      if (ogeom_) ogeom_->flushViews();
    }
  } else if (args[1] == "toggle_display_edges"){
    // Toggle the GeomSwitch.
    edges_on_.reset();
    if (renderer_.get_rep() && renderer_->edge_switch_ && now) 
      renderer_->edge_switch_->set_state(edges_on_.get());
    
    if ((edges_on_.get()) && (edge_id_ == 0)) {
      edges_dirty_ = true;
      maybe_execute(EDGE);
    } else {
      if (ogeom_) ogeom_->flushViews();
    }
  } else if (args[1] == "rerender_faces"){
    faces_dirty_ = true;
    if (now && face_id_) {
      ogeom_->delObj(face_id_);
      face_id_ = 0;
    }
    maybe_execute(FACE);
  } else if (args[1] == "toggle_display_faces"){
    // Toggle the GeomSwitch.
    faces_on_.reset();
    if (renderer_.get_rep() && renderer_->face_switch_ && now) 
      renderer_->face_switch_->set_state(faces_on_.get());

    if ((faces_on_.get()) && (face_id_ == 0)) {
      faces_dirty_ = true;
      maybe_execute(FACE);
    } else {
      if (ogeom_) ogeom_->flushViews();
    }
  } else if (args[1] == "toggle_display_vectors"){
    // Toggle the GeomSwitch.
    vectors_on_.reset();
    if (renderer_.get_rep() && renderer_->data_switch_ && now) 
      renderer_->data_switch_->set_state(vectors_on_.get());

    if ((vectors_on_.get()) && (data_id_ == 0)) {
      data_dirty_ = true;
      maybe_execute(DATA);
    } else {
      if (ogeom_) ogeom_->flushViews();
    }
  } else if (args[1] == "toggle_normalize"){
    // Toggle the GeomSwitch.
    normalize_vectors_.reset();
    data_dirty_ = true;
    maybe_execute(FACE); // Must redraw the vectors.
  } else if (args[1] == "execute_policy"){
  } else if (args[1] == "calcdefs") {
    
    if (bounding_vector_) {
      //0.00896657
      double fact = 0.01 * bounding_vector_->length();
      node_scale_.set(fact);
      edge_scale_.set(fact * 0.5);
      vectors_scale_.set(fact * 10);
      nodes_dirty_ = true;
      edges_dirty_ = true;
      data_dirty_ = true;
      maybe_execute(DATA_AT);
    } else {
      warning("Cannot calculate defaults without a valid field input.");
    }

  } else {
    Module::tcl_command(args, userdata);
  }
}

DECLARE_MAKER(ShowField)
} // End namespace SCIRun


