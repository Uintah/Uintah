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
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomSwitch.h>
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
  int                      field_generation_;
  int                      mesh_generation_;
  int                      colormap_generation_;
  int                      vector_generation_;

  //! output port
  GeometryOPort           *ogeom_;  

  //! Scene graph ID's
  int                      node_id_;
  int                      edge_id_;
  int                      face_id_;
  int                      data_id_;
  int                      text_id_;

  //! top level nodes for switching on and off..
  //! Options for rendering nodes.
  GuiInt                   nodes_on_;
  GuiInt                   nodes_transparency_;
  GuiInt                   nodes_as_disks_;
  bool                     nodes_dirty_;

  //! Options for rendering edges.
  GuiInt                   edges_on_;
  GuiInt                   edges_transparency_;
  bool                     edges_dirty_;

  //! Options for rendering faces.
  GuiInt                   faces_on_;
  GuiInt                   faces_normals_;
  GuiInt                   faces_transparency_;
  bool                     faces_dirty_;

  //! Options for rendering non-scalar data.
  GuiInt                   vectors_on_;
  GuiInt                   normalize_vectors_;
  GuiInt                   has_vector_data_;
  GuiInt                   bidirectional_;
  GuiInt                   arrow_heads_on_;
  bool                     data_dirty_;
  string                   cur_field_data_type_;
  Field::data_location     cur_field_data_at_;

  GuiInt                   tensors_on_;
  GuiInt                   has_tensor_data_;
  
  //! Options for rendering text.
  GuiInt                   text_on_;
  GuiInt                   text_use_default_color_;
  GuiDouble                text_color_r_;
  GuiDouble                text_color_g_;
  GuiDouble                text_color_b_;
  GuiInt                   text_backface_cull_;
  GuiInt                   text_fontsize_;
  GuiInt                   text_precision_;
  GuiInt                   text_render_locations_;
  GuiInt                   text_show_data_;
  GuiInt                   text_show_nodes_;
  GuiInt                   text_show_edges_;
  GuiInt                   text_show_faces_;
  GuiInt                   text_show_cells_;
  bool                     text_dirty_;
  
  //! default color and material
  GuiDouble                def_color_r_;
  GuiDouble                def_color_g_;
  GuiDouble                def_color_b_;
  GuiDouble                def_color_a_;
  MaterialHandle           def_mat_handle_;

  RenderFieldBase::ind_mat_t idx_mats_;

  //! re-render all the material handles?
  bool                     data_at_dirty_;

  //! holds options for how to visualize nodes.
  GuiString                node_display_type_;
  GuiString                edge_display_type_;
  GuiString                data_display_type_;
  GuiString                tensor_display_type_;
  GuiString                active_tab_; //! for saving nets state
  GuiDouble                node_scale_;
  GuiDouble                edge_scale_;
  GuiDouble                vectors_scale_;
  GuiDouble                tensors_scale_;
  GuiInt                   showProgress_;
  GuiString                interactive_mode_;

  //! Refinement resolution for cylinders and spheres
  GuiInt                   gui_node_resolution_;
  GuiInt                   gui_edge_resolution_;
  GuiInt                   gui_data_resolution_;
  int                      node_resolution_;
  int                      edge_resolution_;
  int                      data_resolution_;
  LockingHandle<RenderFieldBase>  renderer_;
  LockingHandle<RenderVectorFieldBase>  data_renderer_;
  LockingHandle<RenderTensorFieldBase>  data_tensor_renderer_;

  enum toggle_type_e {
    NODE = 0,
    EDGE,
    FACE,
    DATA,
    TEXT,
    DATA_AT
  };
  vector<bool>               render_state_;
  void maybe_execute(toggle_type_e dis_type);
  Vector                  *bounding_vector_;

public:
  ShowField(GuiContext* ctx);
  virtual ~ShowField();
  virtual void execute();
  bool check_for_vector_data(FieldHandle fld_handle);
  bool fetch_typed_algorithm(FieldHandle fld_handle, FieldHandle vfld_handle,
			     bool recompile_nonvector);
  bool determine_dirty(FieldHandle fld_handle, FieldHandle vfld_handle);
  virtual void tcl_command(GuiArgs& args, void* userdata);
};

ShowField::ShowField(GuiContext* ctx) : 
  Module("ShowField", ctx, Filter, "Visualization", "SCIRun"), 
  field_generation_(-1),
  mesh_generation_(-1),
  colormap_generation_(-1),
  vector_generation_(-1),
  ogeom_(0),
  node_id_(0),
  edge_id_(0),
  face_id_(0),
  data_id_(0),
  text_id_(0),
  nodes_on_(ctx->subVar("nodes-on")),
  nodes_transparency_(ctx->subVar("nodes-transparency")),
  nodes_as_disks_(ctx->subVar("nodes-as-disks")),
  nodes_dirty_(true),
  edges_on_(ctx->subVar("edges-on")),
  edges_transparency_(ctx->subVar("edges-transparency")),
  edges_dirty_(true),
  faces_on_(ctx->subVar("faces-on")),
  faces_normals_(ctx->subVar("use-normals")),
  faces_transparency_(ctx->subVar("use-transparency")),
  faces_dirty_(true),
  vectors_on_(ctx->subVar("vectors-on")),
  normalize_vectors_(ctx->subVar("normalize-vectors")),
  has_vector_data_(ctx->subVar("has_vector_data")),
  bidirectional_(ctx->subVar("bidirectional")),
  arrow_heads_on_(ctx->subVar("arrow-heads-on")),
  data_dirty_(true),
  cur_field_data_type_("none"),
  cur_field_data_at_(Field::NONE),
  tensors_on_(ctx->subVar("tensors-on")),
  has_tensor_data_(ctx->subVar("has_tensor_data")),
  text_on_(ctx->subVar("text-on")),
  text_use_default_color_(ctx->subVar("text-use-default-color")),
  text_color_r_(ctx->subVar("text-color-r")),
  text_color_g_(ctx->subVar("text-color-g")),
  text_color_b_(ctx->subVar("text-color-b")),
  text_backface_cull_(ctx->subVar("text-backface-cull")),
  text_fontsize_(ctx->subVar("text-fontsize")),
  text_precision_(ctx->subVar("text-precision")),
  text_render_locations_(ctx->subVar("text-render_locations")),
  text_show_data_(ctx->subVar("text-show-data")),
  text_show_nodes_(ctx->subVar("text-show-nodes")),
  text_show_edges_(ctx->subVar("text-show-edges")),
  text_show_faces_(ctx->subVar("text-show-faces")),
  text_show_cells_(ctx->subVar("text-show-cells")),
  text_dirty_(true),
  def_color_r_(ctx->subVar("def-color-r")),
  def_color_g_(ctx->subVar("def-color-g")),
  def_color_b_(ctx->subVar("def-color-b")),
  def_color_a_(ctx->subVar("def-color-a")),
  def_mat_handle_(scinew Material(Color(0.5, 0.5, 0.5))),
  data_at_dirty_(true),
  node_display_type_(ctx->subVar("node_display_type")),
  edge_display_type_(ctx->subVar("edge_display_type")),
  data_display_type_(ctx->subVar("data_display_type")),
  tensor_display_type_(ctx->subVar("tensor_display_type")),
  active_tab_(ctx->subVar("active_tab")),
  node_scale_(ctx->subVar("node_scale")),
  edge_scale_(ctx->subVar("edge_scale")),
  vectors_scale_(ctx->subVar("vectors_scale")),
  tensors_scale_(ctx->subVar("tensors_scale")),
  showProgress_(ctx->subVar("show_progress")),
  interactive_mode_(ctx->subVar("interactive_mode")),
  gui_node_resolution_(ctx->subVar("node-resolution")),
  gui_edge_resolution_(ctx->subVar("edge-resolution")),
  gui_data_resolution_(ctx->subVar("data-resolution")),
  node_resolution_(0),
  edge_resolution_(0),
  data_resolution_(0),
  renderer_(0),
  data_renderer_(0),
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
  tensors_on_.reset();
  render_state_[DATA] = vectors_on_.get() || tensors_on_.get();
  text_on_.reset();
  render_state_[TEXT] = text_on_.get();
}


ShowField::~ShowField()
{
}


bool
ShowField::check_for_vector_data(FieldHandle fld_handle) {
  // Test for vector data possibility
  if (fld_handle.get_rep() == 0) { return false; }

  has_vector_data_.reset();
  has_tensor_data_.reset();
  nodes_as_disks_.reset();
  if (fld_handle->query_vector_interface(this).get_rep() != 0)
  {
    if (! has_vector_data_.get())
    { 
      has_vector_data_.set(1); 
    }
    if (fld_handle->data_at() == Field::NODE && nodes_as_disks_.get() == 0)
    {
      nodes_as_disks_.set(1); 
    }
    return true;
  }
  else if (fld_handle->query_tensor_interface(this).get_rep() != 0)
  {
    if (! has_tensor_data_.get())
    {
      has_tensor_data_.set(1);
    }
  }
  if (nodes_as_disks_.get() == 1)
  {
    nodes_as_disks_.set(0);
  }
  return false;
}


bool
ShowField::fetch_typed_algorithm(FieldHandle fld_handle,
				 FieldHandle vfld_handle,
				 bool recompile_nonvector) 
{
  const TypeDescription *ftd = fld_handle->get_type_description();
  const TypeDescription *ltd = fld_handle->data_at_type_description();
  // description for just the data in the field
  cur_field_data_type_ = fld_handle->get_type_description(1)->get_name();
  cur_field_data_at_ = fld_handle->data_at();

  if (recompile_nonvector)
  {
    // Get the Algorithm.
    CompileInfoHandle ci = RenderFieldBase::get_compile_info(ftd, ltd);
    if (!module_dynamic_compile(ci, renderer_))
    {
      field_generation_ = -1;
      mesh_generation_ = -1;
      vector_generation_ = -1;
      return false;
    }
  }

  if (vfld_handle.get_rep() && 
      vfld_handle->query_vector_interface(this).get_rep())
  {
    const TypeDescription *vftd = vfld_handle->get_type_description();
    CompileInfoHandle dci =
      RenderVectorFieldBase::get_compile_info(vftd, ftd, ltd);
    if (!module_dynamic_compile(dci, data_renderer_))
    {
      field_generation_ = -1;
      mesh_generation_ = -1;
      vector_generation_ = -1;
      data_renderer_ = 0;
      return false;
    }
  }

  if (vfld_handle.get_rep() && 
      vfld_handle->query_tensor_interface(this).get_rep())
  {
    const TypeDescription *vftd = vfld_handle->get_type_description();
    CompileInfoHandle dci =
      RenderTensorFieldBase::get_compile_info(vftd, ftd, ltd);
    if (!module_dynamic_compile(dci, data_tensor_renderer_))
    {
      field_generation_ = -1;
      mesh_generation_ = -1;
      vector_generation_ = -1;
      data_tensor_renderer_ = 0;
      return false;
    }
  }

  return true;
}


bool
ShowField::determine_dirty(FieldHandle fld_handle, FieldHandle vfld_handle) 
{
  bool mesh_new = fld_handle->mesh()->generation != mesh_generation_;
  bool field_new = fld_handle->generation != field_generation_;
  bool vector_new = 
    vfld_handle.get_rep() && vfld_handle->generation != vector_generation_;

  if (mesh_new) {
    // completely new, all dirty, or just new geometry, so data_at invalid too.
    if (!check_for_vector_data(fld_handle))
    {
      check_for_vector_data(vfld_handle);
    }
    if (!fetch_typed_algorithm(fld_handle, vfld_handle, true))
    {
      return false;
    }
    field_generation_  = fld_handle->generation;  
    mesh_generation_ = fld_handle->mesh()->generation; 
    vector_generation_ =
      (vfld_handle.get_rep())?(vfld_handle->mesh()->generation):-1;
    nodes_dirty_ = true; 
    edges_dirty_ = true; 
    faces_dirty_ = true; 
    data_dirty_ = true;
    data_at_dirty_ = true;
    text_dirty_ = true;
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
    if (text_id_) ogeom_->delObj(text_id_);
    text_id_ = 0;

    if (bounding_vector_) delete bounding_vector_;
    bounding_vector_ = scinew Vector();
    *bounding_vector_ = fld_handle->mesh()->get_bounding_box().diagonal();
    
  } else if (!mesh_new && (field_new || vector_new)) {
    // same geometry, new data.
    if (!check_for_vector_data(fld_handle))
    {
      check_for_vector_data(vfld_handle);
    }

    const TypeDescription *data_type_description = 
      fld_handle->get_type_description(1);
    string fdt = data_type_description->get_name();
    Field::data_location at = fld_handle->data_at();
    if (!fetch_typed_algorithm(fld_handle, vfld_handle,
			       (cur_field_data_type_ != fdt) ||
			       (cur_field_data_at_ != at)))
    { 
      return false;
    }
    data_at_dirty_ = true; //we need to rerender colors..
    nodes_dirty_ = true; // Nodes don't cache color.
    edges_dirty_ = true; // Edges don't cache color.
    faces_dirty_ = true; // Faces don't cache color.
    data_dirty_ = true; // Data doesn't cache color.
    text_dirty_ = true; // Text doesn't cache color.
  } //else both are the same as last time, nothing dirty.
  
  return true;
}


void 
ShowField::execute()
{
  // tell module downstream to delete everything we have sent it before.
  // This is typically viewer, it owns the scene graph memory we create here.
  FieldIPort *field_iport = (FieldIPort *)get_iport("Field");
  ColorMapIPort *color_iport = (ColorMapIPort *)get_iport("ColorMap");
  ogeom_ = (GeometryOPort *)get_oport("Scene Graph");

  if (!field_iport) {
    error("Unable to initialize iport 'Field'.");
    return;
  }
  if (!color_iport) {
    error("Unable to initialize iport 'ColorMap'.");
    return;
  }
  if (!ogeom_) {
    error("Unable to initialize oport 'Scene Graph'.");
    return;
  }

  FieldHandle fld_handle;
  field_iport->get(fld_handle);
  if(!fld_handle.get_rep())
  {
    warning("No Data in port 1 field.");
    return;
  }

  FieldIPort *vfield_iport = (FieldIPort *)get_iport("Orientation Field");
  FieldHandle vfld_handle;
  vfield_iport->get(vfld_handle);
  if (vfld_handle.get_rep())
  {
    if (vfld_handle->mesh().get_rep() != fld_handle->mesh().get_rep())
    {
      error("Color and Orientation fields must share the same mesh.");
      return;
    }
    if (vfld_handle->data_at() != fld_handle->data_at())
    {
      warning("Color and Orientation fields must have data at the same location.");
      return;
    }
  }
  else if (fld_handle->query_vector_interface(this).get_rep() ||
	   fld_handle->query_tensor_interface(this).get_rep())
  {
    vfld_handle = fld_handle;
  }
  else
  {
    vfld_handle = 0;
  }

  // What has changed from last time?  A false return value means that we 
  // could not load the algorithm from the dynamic loader.
  if (! determine_dirty(fld_handle, vfld_handle)) { return; }
  
  // if no colormap was attached, the argument doesn't get changed,
  // so we need to set it manually
  // if the user had a colormap attached for a previous execution,
  // and then detached it, we need to set color_handle to be empty

  ColorMapHandle color_handle;
  if(!color_iport->get(color_handle)) color_handle=0;

  if(!color_handle.get_rep()){
    //warning("No ColorMap in port 2 ColorMap.");
    if (colormap_generation_ != -1) {
      data_at_dirty_ = true;
      nodes_dirty_ = true;
      edges_dirty_ = true;
      faces_dirty_ = true;
      text_dirty_ = true;
      data_dirty_ = true;
    }
    colormap_generation_ = -1;
  } else if (colormap_generation_ != color_handle->generation) {
    colormap_generation_ = color_handle->generation;  
    data_at_dirty_ = true;
    nodes_dirty_ = true;
    edges_dirty_ = true;
    faces_dirty_ = true;
    text_dirty_ = true;
    data_dirty_ = true;
  }

  if (gui_node_resolution_.get() != node_resolution_) {
    nodes_dirty_ = true;
  }
  node_resolution_ = gui_node_resolution_.get();

  if (gui_edge_resolution_.get() != edge_resolution_) {
    edges_dirty_ = true;
  }
  edge_resolution_ = gui_edge_resolution_.get();

  if (gui_data_resolution_.get() != data_resolution_) {
    data_dirty_ = true;
  }
  data_resolution_ = gui_data_resolution_.get();

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
  data_display_type_.reset();
  string vdt = data_display_type_.get();
  tensor_display_type_.reset();
  string tdt = tensor_display_type_.get();
  vectors_scale_.reset();
  double vscale = vectors_scale_.get();
  tensors_scale_.reset();
  double tscale = tensors_scale_.get();

  nodes_on_.reset();
  edges_on_.reset();
  faces_on_.reset();
  vectors_on_.reset();
  tensors_on_.reset();
  text_on_.reset();
  bool do_nodes = nodes_on_.get() && nodes_dirty_;
  bool do_edges = edges_on_.get() && edges_dirty_;
  bool do_faces = faces_on_.get() && faces_dirty_;
  bool do_data  = (vectors_on_.get() || tensors_on_.get()) && data_dirty_;
  bool do_text  = text_on_.get() && text_dirty_;

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
  if (render_state_[DATA] != (vectors_on_.get() || tensors_on_.get())) {
    if (data_id_) ogeom_->delObj(data_id_);
    data_id_ = 0;
    render_state_[DATA] = vectors_on_.get() || tensors_on_.get();
  }  
  if (render_state_[TEXT] != text_on_.get()) {
    if (text_id_) ogeom_->delObj(text_id_);
    text_id_ = 0;
    render_state_[TEXT] = text_on_.get();
  }  

  normalize_vectors_.reset();
  if (renderer_.get_rep())
  {
    if (faces_normals_.get()) fld_handle->mesh()->synchronize(Mesh::NORMALS_E);

    renderer_->set_mat_map(&idx_mats_);
    renderer_->render(fld_handle, 
		      do_nodes, do_edges, do_faces, false,
		      def_mat_handle_, data_at_dirty_, color_handle,
		      ndt, edt, ns, es, vscale, normalize_vectors_.get(),
		      node_resolution_, edge_resolution_,
		      faces_normals_.get(),
		      nodes_transparency_.get(),
		      edges_transparency_.get(),
		      faces_transparency_.get(),
		      bidirectional_.get(), arrow_heads_on_.get());
  }

  // cleanup...
  if (do_nodes) {
    nodes_dirty_ = false;
    if (renderer_.get_rep() && nodes_on_.get()) {
      const char *name = nodes_transparency_.get()?"TransParent Nodes":"Nodes";
      if (node_id_) ogeom_->delObj(node_id_);
      node_id_ = ogeom_->addObj(renderer_->node_switch_, name);
    }
  }
  if (do_edges) {
    edges_dirty_ = false;
    if (renderer_.get_rep() && edges_on_.get()) {
      const char *name = edges_transparency_.get()?"TransParent Edges":"Edges";
      if (edge_id_) ogeom_->delObj(edge_id_);
      edge_id_ = ogeom_->addObj(renderer_->edge_switch_, name);
    }
  }
  if (do_faces) {
    faces_dirty_ = false;
    if (renderer_.get_rep() && faces_on_.get()) 
    {
      const char *name = faces_transparency_.get()?"TransParent Faces":"Faces";
      if (face_id_) ogeom_->delObj(face_id_);
      face_id_ = ogeom_->addObj(renderer_->face_switch_, name);
    }
  }  
  if (do_data)
  {
    data_dirty_ = false;
    if (vfld_handle.get_rep() &&
	data_renderer_.get_rep() &&
	vectors_on_.get())
    {
      if (data_id_) ogeom_->delObj(data_id_);
      GeomHandle data = data_renderer_->render_data(vfld_handle,
						    fld_handle,
						    color_handle,
						    def_mat_handle_,
						    vdt, vscale,
						    normalize_vectors_.get(),
						    bidirectional_.get(),
						    arrow_heads_on_.get(),
						    data_resolution_);
      data_id_ = ogeom_->addObj(data, "Vectors");
    }
    else if (vfld_handle.get_rep() &&
	     data_tensor_renderer_.get_rep() &&
	     tensors_on_.get())
    {
      if (data_id_) ogeom_->delObj(data_id_);
      GeomHandle data = data_tensor_renderer_->render_data(vfld_handle,
							   fld_handle,
							   color_handle,
							   def_mat_handle_,
							   tdt, tscale,
							   data_resolution_);
      data_id_ = ogeom_->addObj(data, "Tensors");
    }
  }
  if (do_text) {
    text_dirty_ = false;
    if (renderer_.get_rep() && text_on_.get()) {
      if (text_id_) ogeom_->delObj(text_id_);
      MaterialHandle m = scinew Material(Color(text_color_r_.get(),
					       text_color_g_.get(),
					       text_color_b_.get()));
      GeomSwitch *text =
	renderer_->render_text(fld_handle, text_use_default_color_.get(), m,
			       text_backface_cull_.get(),
			       text_fontsize_.get(),
			       text_precision_.get(),
			       text_render_locations_.get(),
			       text_show_data_.get(),
			       text_show_nodes_.get(),
			       text_show_edges_.get(),
			       text_show_faces_.get(),
			       text_show_cells_.get());

      const char *name =
	text_backface_cull_.get()?"Culled Text Data":"Text Data";
      text_id_ = ogeom_->addObj(text, name);
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
      do_execute = vectors_on_.get() || tensors_on_.get();
	break;
    case TEXT :
      do_execute = text_on_.get();
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
  } else if (args[1] == "node_resolution_scale") {
    nodes_dirty_ = true;
    if (nodes_on_.get())
    {
      maybe_execute(NODE);
    }
  } else if (args[1] == "edge_resolution_scale") {
    edges_dirty_ = true;
    if (edges_on_.get())
    {
      maybe_execute(EDGE);
    }
  } else if (args[1] == "data_resolution_scale") {
    if (tensors_on_.get() && tensor_display_type_.get() == "Ellipsoids")
    {
      data_dirty_ = true;
      maybe_execute(DATA);
    }
    else
    {
      data_dirty_ = true;
      if (vectors_on_.get())
      {
	maybe_execute(DATA);
      }
    }
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
    edges_dirty_ = true;
    faces_dirty_ = true;
    maybe_execute(DATA_AT);
  } else if (args[1] == "text_color_change") {
    text_color_r_.reset();
    text_color_g_.reset();
    text_color_b_.reset();
    text_dirty_ = true;
    maybe_execute(TEXT);
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
  } else if (args[1] == "data_display_type") {
    data_dirty_ = true;
    if (now && data_id_) {
      ogeom_->delObj(data_id_);
      data_id_ = 0;
    }
    maybe_execute(DATA);
  } else if (args[1] == "toggle_display_nodes"){
    // Toggle the GeomSwitches.
    nodes_on_.reset();
    if ((nodes_on_.get()) && (node_id_ == 0))
    {
      nodes_dirty_ = true;
      maybe_execute(NODE);
    }
    else if (!nodes_on_.get() && node_id_)
    {
      ogeom_->delObj(node_id_);
      ogeom_->flushViews();
      node_id_ = 0;
    }
  } else if (args[1] == "toggle_display_edges"){
    // Toggle the GeomSwitch.
    edges_on_.reset();
    if ((edges_on_.get()) && (edge_id_ == 0))
    {
      edges_dirty_ = true;
      maybe_execute(EDGE);
    }
    else if (!edges_on_.get() && edge_id_)
    {
      ogeom_->delObj(edge_id_);
      ogeom_->flushViews();
      edge_id_ = 0;
    }
  } else if (args[1] == "rerender_nodes"){
    nodes_dirty_ = true;
    if (now && node_id_) {
      ogeom_->delObj(node_id_);
      node_id_ = 0;
    }
    maybe_execute(NODE);
  } else if (args[1] == "rerender_edges"){
    edges_dirty_ = true;
    if (now && edge_id_) {
      ogeom_->delObj(edge_id_);
      edge_id_ = 0;
    }
    maybe_execute(EDGE);
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
    if ((faces_on_.get()) && (face_id_ == 0))
    {
      faces_dirty_ = true;
      maybe_execute(FACE);
    }
    else if (!faces_on_.get() && face_id_)
    {
      ogeom_->delObj(face_id_);
      ogeom_->flushViews();
      face_id_ = 0;
    }
  } else if (args[1] == "toggle_display_vectors"){
    // Toggle the GeomSwitch.
    vectors_on_.reset();
    if ((vectors_on_.get()) && (data_id_ == 0))
    {
      data_dirty_ = true;
      maybe_execute(DATA);
    }
    else if (!vectors_on_.get() && data_id_)
    {
      ogeom_->delObj(data_id_);
      ogeom_->flushViews();
      data_id_ = 0;
    }
  } else if (args[1] == "toggle_display_tensors"){
    // Toggle the GeomSwitch.
    tensors_on_.reset();
    if ((tensors_on_.get()) && (data_id_ == 0))
    {
      data_dirty_ = true;
      maybe_execute(DATA);
    }
    else if (!tensors_on_.get() && data_id_)
    {
      ogeom_->delObj(data_id_);
      ogeom_->flushViews();
      data_id_ = 0;
    }
  } else if (args[1] == "toggle_display_text"){
    // Toggle the GeomSwitch.
    text_on_.reset();
    if ((text_on_.get()) && (text_id_ == 0))
    {
      text_dirty_ = true;
      maybe_execute(TEXT);
    }
    else if (!text_on_.get() && text_id_)
    {
      ogeom_->delObj(text_id_);
      ogeom_->flushViews();
      text_id_ = 0;
    }
  } else if (args[1] == "rerender_text"){
    text_dirty_ = true;
    if (now && text_id_) {
      ogeom_->delObj(text_id_);
      text_id_ = 0;
    }
    maybe_execute(TEXT);
  } else if (args[1] == "toggle_normalize"){
    // Toggle the GeomSwitch.
    normalize_vectors_.reset();
    data_dirty_ = true;
    maybe_execute(DATA); // Must redraw the vectors.
  } else if (args[1] == "toggle_bidirectional"){
    // Toggle the GeomSwitch.
    bidirectional_.reset();
    data_dirty_ = true;
    maybe_execute(DATA); // Must redraw the vectors.
  } else if (args[1] == "toggle_arrowheads"){
    // Toggle the GeomSwitch.
    arrow_heads_on_.reset();
    data_dirty_ = true;
    maybe_execute(DATA); // Must redraw the vectors.
  } else if (args[1] == "execute_policy"){
  } else if (args[1] == "calcdefs") {
    
    if (bounding_vector_) {
      //0.00896657
      double fact = 0.01 * bounding_vector_->length();
      node_scale_.set(fact);
      edge_scale_.set(fact * 0.5);
      vectors_scale_.set(fact * 10);
      tensors_scale_.set(fact * 10);
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


