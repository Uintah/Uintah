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
  int                      vector_generation_;
  int                      color_map_generation_;

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
  bool                     data_dirty_;
  string                   cur_field_data_type_;
  Field::data_location     cur_field_data_at_;

  GuiInt                   tensors_on_;
  GuiInt                   has_tensor_data_;

  GuiInt                   scalars_on_;
  GuiInt                   scalars_transparency_;
  GuiInt                   has_scalar_data_;
  
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
  MaterialHandle           text_material_;
  bool                     text_dirty_;
  
  //! default color and material
  GuiDouble                def_color_r_;
  GuiDouble                def_color_g_;
  GuiDouble                def_color_b_;
  GuiDouble                def_color_a_;
  MaterialHandle           def_material_;
  ColorMapHandle           color_map_;

  //! holds options for how to visualize nodes.
  GuiString                node_display_type_;
  GuiString                edge_display_type_;
  GuiString                data_display_type_;
  GuiString                tensor_display_type_;
  GuiString                scalar_display_type_;
  GuiString                active_tab_; //! for saving nets state
  GuiDouble                node_scale_;
  GuiDouble                edge_scale_;
  GuiDouble                vectors_scale_;
  GuiDouble                tensors_scale_;
  GuiDouble                scalars_scale_;
  GuiInt                   showProgress_;
  GuiString                interactive_mode_;
  GuiString                gui_field_name_;
  GuiInt                   gui_field_name_update_;

  //! Refinement resolution for cylinders and spheres
  GuiInt                   gui_node_resolution_;
  GuiInt                   gui_edge_resolution_;
  GuiInt                   gui_data_resolution_;
  int                      node_resolution_;
  int                      edge_resolution_;
  int                      data_resolution_;
  LockingHandle<RenderFieldBase>  renderer_;
  LockingHandle<RenderScalarFieldBase>  data_scalar_renderer_;
  LockingHandle<RenderVectorFieldBase>  data_vector_renderer_;
  LockingHandle<RenderTensorFieldBase>  data_tensor_renderer_;

  GeomHandle text_geometry_;
  GeomHandle data_geometry_;

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

public:
  ShowField(GuiContext* ctx);
  virtual ~ShowField();
  virtual void execute();
  bool check_for_svt_data(FieldHandle fld_handle);
  bool fetch_typed_algorithm(FieldHandle fld_handle, FieldHandle vfld_handle,
			     bool recompile_nonvector);
  bool determine_dirty(FieldHandle fld_handle, FieldHandle vfld_handle);
  virtual void tcl_command(GuiArgs& args, void* userdata);
};

ShowField::ShowField(GuiContext* ctx) : 
  Module("ShowField", ctx, Filter, "Visualization", "SCIRun"), 
  field_generation_(-1),
  mesh_generation_(-1),
  vector_generation_(-1),
  color_map_generation_(-1),
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
  data_dirty_(true),
  cur_field_data_type_("none"),
  cur_field_data_at_(Field::NONE),
  tensors_on_(ctx->subVar("tensors-on")),
  has_tensor_data_(ctx->subVar("has_tensor_data")),
  scalars_on_(ctx->subVar("scalars-on")),
  scalars_transparency_(ctx->subVar("scalars-transparency")),
  has_scalar_data_(ctx->subVar("has_scalar_data")),
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
  text_material_(scinew Material(Color(0.75, 0.75, 0.75))),
  text_dirty_(true),
  def_color_r_(ctx->subVar("def-color-r")),
  def_color_g_(ctx->subVar("def-color-g")),
  def_color_b_(ctx->subVar("def-color-b")),
  def_color_a_(ctx->subVar("def-color-a")),
  def_material_(scinew Material(Color(0.5, 0.5, 0.5))),
  color_map_(0),
  node_display_type_(ctx->subVar("node_display_type")),
  edge_display_type_(ctx->subVar("edge_display_type")),
  data_display_type_(ctx->subVar("data_display_type")),
  tensor_display_type_(ctx->subVar("tensor_display_type")),
  scalar_display_type_(ctx->subVar("scalar_display_type")),
  active_tab_(ctx->subVar("active_tab")),
  node_scale_(ctx->subVar("node_scale")),
  edge_scale_(ctx->subVar("edge_scale")),
  vectors_scale_(ctx->subVar("vectors_scale")),
  tensors_scale_(ctx->subVar("tensors_scale")),
  scalars_scale_(ctx->subVar("scalars_scale")),
  showProgress_(ctx->subVar("show_progress")),
  interactive_mode_(ctx->subVar("interactive_mode")),
  gui_field_name_(ctx->subVar("field-name")),
  gui_field_name_update_(ctx->subVar("field-name-update")),
  gui_node_resolution_(ctx->subVar("node-resolution")),
  gui_edge_resolution_(ctx->subVar("edge-resolution")),
  gui_data_resolution_(ctx->subVar("data-resolution")),
  node_resolution_(0),
  edge_resolution_(0),
  data_resolution_(0),
  renderer_(0),
  data_scalar_renderer_(0),
  data_vector_renderer_(0),
  data_tensor_renderer_(0),
  text_geometry_(0),
  data_geometry_(0),
  render_state_(5)
{
  def_material_->transparency = 0.5;
  nodes_on_.reset();
  render_state_[NODE] = nodes_on_.get();
  edges_on_.reset();
  render_state_[EDGE] = edges_on_.get();
  faces_on_.reset();
  render_state_[FACE] = faces_on_.get();
  vectors_on_.reset();
  tensors_on_.reset();
  scalars_on_.reset();
  render_state_[DATA] =
    vectors_on_.get() || tensors_on_.get() || scalars_on_.get();
  text_on_.reset();
  render_state_[TEXT] = text_on_.get();
}


ShowField::~ShowField()
{
}


bool
ShowField::check_for_svt_data(FieldHandle fld_handle)
{
  // Test for vector data possibility
  if (fld_handle.get_rep() == 0) { return false; }

  has_vector_data_.set(0);
  has_tensor_data_.set(0);
  has_scalar_data_.set(0);
  nodes_as_disks_.reset();
  bool disks = false;
  bool result = false;
  if (fld_handle->query_scalar_interface(this).get_rep() != 0)
  {
    if (! has_scalar_data_.get())
    {
      has_scalar_data_.set(1);
    }
    result = true;
  }
  else if (fld_handle->query_vector_interface(this).get_rep() != 0)
  {
    if (! has_vector_data_.get())
    { 
      has_vector_data_.set(1); 
    }
    if (fld_handle->data_at() == Field::NODE)
    {
      disks = true;
    }
    result = true;
  }
  else if (fld_handle->query_tensor_interface(this).get_rep() != 0)
  {
    if (! has_tensor_data_.get())
    {
      has_tensor_data_.set(1);
    }
    result = true;
  }
  if (nodes_as_disks_.get() != disks)
  {
    nodes_as_disks_.set(disks);
  }
  return result;
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
      vfld_handle->query_scalar_interface(this).get_rep())
  {
    const TypeDescription *vftd = vfld_handle->get_type_description();
    CompileInfoHandle dci =
      RenderScalarFieldBase::get_compile_info(vftd, ftd, ltd);
    if (!module_dynamic_compile(dci, data_scalar_renderer_))
    {
      field_generation_ = -1;
      mesh_generation_ = -1;
      vector_generation_ = -1;
      data_scalar_renderer_ = 0;
      return false;
    }
  }

  if (vfld_handle.get_rep() && 
      vfld_handle->query_vector_interface(this).get_rep())
  {
    const TypeDescription *vftd = vfld_handle->get_type_description();
    CompileInfoHandle dci =
      RenderVectorFieldBase::get_compile_info(vftd, ftd, ltd);
    if (!module_dynamic_compile(dci, data_vector_renderer_))
    {
      field_generation_ = -1;
      mesh_generation_ = -1;
      vector_generation_ = -1;
      data_vector_renderer_ = 0;
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
  const bool mesh_new = fld_handle->mesh()->generation != mesh_generation_;
  const bool field_new = fld_handle->generation != field_generation_;
  const bool vector_new =
    (vfld_handle.get_rep())?
    (vfld_handle->generation != vector_generation_):
    (vector_generation_ != -1);
  
  // Update the field name.
  if ((field_new || vector_new) && gui_field_name_update_.get())
  {
    string fname;
    if (vfld_handle.get_rep() &&
	vfld_handle.get_rep() != fld_handle.get_rep() &&
	vfld_handle->get_property("name", fname))
    {
      gui_field_name_.set(fname);      
    }
    else if (fld_handle->get_property("name", fname))
    {
      gui_field_name_.set(fname);
    }
    else if (fld_handle->mesh()->get_property("name", fname))
    {
      gui_field_name_.set(fname);
    }
  }

  if (mesh_new || field_new || vector_new)
  {
    if (!check_for_svt_data(vfld_handle))
    {
      check_for_svt_data(fld_handle);
    }
    
    const TypeDescription *data_type_description = 
      fld_handle->get_type_description(1);
    const string fdt = data_type_description->get_name();
    Field::data_location at = fld_handle->data_at();
    if (!fetch_typed_algorithm(fld_handle, vfld_handle,
			       mesh_new ||
			       (cur_field_data_type_ != fdt) ||
			       (cur_field_data_at_ != at)))
    {
      return false;
    }

    field_generation_  = fld_handle->generation;  
    mesh_generation_ = fld_handle->mesh()->generation; 
    vector_generation_ = (vfld_handle.get_rep())?(vfld_handle->generation):-1;

    nodes_dirty_ = true; 
    edges_dirty_ = true; 
    faces_dirty_ = true; 
    data_dirty_ = true;
    text_dirty_ = true;

    // Set default color here.  Probably bogus.
    def_material_->diffuse =
      Color(def_color_r_.get(), def_color_g_.get(), def_color_b_.get());
    def_material_->transparency = def_color_a_.get();

    // Clear display here.  Probably redundant.
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
  }
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
  else if (fld_handle->query_scalar_interface(this).get_rep() ||
	   fld_handle->query_vector_interface(this).get_rep() ||
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
  
  // Simply update the colormap handle.  If the colormap gets connected
  // or disconnected then we may have to do a redraw.
  const bool was_color_map = color_map_.get_rep();
  if (!color_iport->get(color_map_)) color_map_ = 0;

  bool color_map_changed = false;
  if (((bool)(color_map_.get_rep())) != was_color_map ||
      color_map_.get_rep() && color_map_->generation != color_map_generation_)
  {
    color_map_changed = true;
    color_map_generation_ = color_map_.get_rep()?color_map_->generation:-1;
    // Colormap was added or went away.
    if (vectors_on_.get() || tensors_on_.get()) { data_dirty_ = true; }
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
      (!faces_dirty_) && (!data_dirty_) &&
      (!text_dirty_) && (!color_map_changed))
  {
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
  scalar_display_type_.reset();
  string sdt = scalar_display_type_.get();
  vectors_scale_.reset();
  double vscale = vectors_scale_.get();
  tensors_scale_.reset();
  double tscale = tensors_scale_.get();
  scalars_scale_.reset();
  double sscale = scalars_scale_.get();


  nodes_on_.reset();
  edges_on_.reset();
  faces_on_.reset();
  vectors_on_.reset();
  tensors_on_.reset();
  scalars_on_.reset();
  text_on_.reset();
  bool do_nodes = nodes_on_.get() && nodes_dirty_;
  bool do_edges = edges_on_.get() && edges_dirty_;
  bool do_faces = faces_on_.get() && faces_dirty_;
  bool do_data  = (vectors_on_.get() || tensors_on_.get() || scalars_on_.get()) && data_dirty_;
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
  if (render_state_[DATA] != (vectors_on_.get() ||
			      tensors_on_.get() ||
			      scalars_on_.get()))
  {
    if (data_id_) ogeom_->delObj(data_id_);
    data_id_ = 0;
    render_state_[DATA] =
      vectors_on_.get() || tensors_on_.get() || scalars_on_.get();
  }  
  if (render_state_[TEXT] != text_on_.get()) {
    if (text_id_) ogeom_->delObj(text_id_);
    text_id_ = 0;
    render_state_[TEXT] = text_on_.get();
  }  
  
  string fname = gui_field_name_.get();
  if (fname != "" && fname[fname.size()-1] != ' ') { fname = fname + " "; }

  normalize_vectors_.reset();
  if (renderer_.get_rep())
  {
    if (faces_normals_.get()) fld_handle->mesh()->synchronize(Mesh::NORMALS_E);
    renderer_->render(fld_handle, 
		      do_nodes, do_edges, do_faces,
		      color_map_, def_material_,
		      ndt, edt, ns, es, vscale, normalize_vectors_.get(),
		      node_resolution_, edge_resolution_,
		      faces_normals_.get(),
		      nodes_transparency_.get(),
		      edges_transparency_.get(),
		      faces_transparency_.get(),
		      bidirectional_.get());
  }

  // Cleanup.
  if (do_nodes || color_map_changed) {
    nodes_dirty_ = false;
    if (renderer_.get_rep() && nodes_on_.get()) {
      const char *name = nodes_transparency_.get()?"Transparent Nodes":"Nodes";
      if (node_id_) ogeom_->delObj(node_id_);

      GeomHandle gmat =
	scinew GeomMaterial(renderer_->node_switch_, def_material_);
      GeomHandle geom =
	scinew GeomSwitch(scinew GeomColorMap(gmat, color_map_));
      node_id_ = ogeom_->addObj(geom, fname + name);
    }
  }
  if (do_edges || color_map_changed) {
    edges_dirty_ = false;
    if (renderer_.get_rep() && edges_on_.get()) {
      const char *name = edges_transparency_.get()?"Transparent Edges":"Edges";
      if (edge_id_) ogeom_->delObj(edge_id_);
      GeomHandle gmat =
	scinew GeomMaterial(renderer_->edge_switch_, def_material_);
      GeomHandle geom =
	scinew GeomSwitch(scinew GeomColorMap(gmat, color_map_));
      edge_id_ = ogeom_->addObj(geom, fname + name);
    }
  }
  if (do_faces || color_map_changed) {
    faces_dirty_ = false;
    if (renderer_.get_rep() && faces_on_.get()) 
    {
      const char *name = faces_transparency_.get()?"Transparent Faces":"Faces";
      if (face_id_) ogeom_->delObj(face_id_);
      GeomHandle gmat =
	scinew GeomMaterial(renderer_->face_switch_, def_material_);
      GeomHandle geom =
	scinew GeomSwitch(scinew GeomColorMap(gmat, color_map_));
      face_id_ = ogeom_->addObj(geom, fname + name);
    }
  }  
  if (do_data || color_map_changed)
  {
    data_dirty_ = false;
    if (vfld_handle.get_rep() &&
	data_vector_renderer_.get_rep() &&
	vectors_on_.get())
    {
      if (data_id_) ogeom_->delObj(data_id_);
      if (do_data)
      {
	data_geometry_ = 
	  data_vector_renderer_->render_data(vfld_handle,
					     fld_handle,
					     color_map_,
					     def_material_,
					     vdt, vscale,
					     normalize_vectors_.get(),
					     bidirectional_.get(),
					     data_resolution_);
      }
      const string vdname = (vdt=="Needles")?"Transparent Vectors":"Vectors";
      GeomHandle gmat =
	scinew GeomMaterial(data_geometry_, def_material_);
      GeomHandle geom =
	scinew GeomSwitch(scinew GeomColorMap(gmat, color_map_));
      data_id_ = ogeom_->addObj(geom, fname + vdname);
    }
    else if (vfld_handle.get_rep() &&
	     data_tensor_renderer_.get_rep() &&
	     tensors_on_.get())
    {
      if (data_id_) ogeom_->delObj(data_id_);
      GeomHandle data = data_tensor_renderer_->render_data(vfld_handle,
							   fld_handle,
							   color_map_,
							   def_material_,
							   tdt, tscale,
							   data_resolution_);
      data_id_ = ogeom_->addObj(data, fname + "Tensors");
    }
    else if (vfld_handle.get_rep() &&
	     data_scalar_renderer_.get_rep() &&
	     scalars_on_.get())
    {
      if (data_id_) ogeom_->delObj(data_id_);
      const bool transp = scalars_transparency_.get();
      if (do_data)
      {
	data_geometry_ = data_scalar_renderer_->render_data(vfld_handle,
							    fld_handle,
							    color_map_,
							    def_material_,
							    sdt, sscale,
							    data_resolution_,
							    transp);
      }

      GeomHandle gmat =
	scinew GeomMaterial(data_geometry_, def_material_);
      GeomHandle geom =
	scinew GeomSwitch(scinew GeomColorMap(gmat, color_map_));
      data_id_ = ogeom_->addObj(geom, fname + (transp?"Transparent Scalars":"Scalars"));
    }
  }
  if (do_text || color_map_changed) {
    text_dirty_ = false;
    if (renderer_.get_rep() && text_on_.get()) {
      if (text_id_) ogeom_->delObj(text_id_);
      text_material_->diffuse =
	Color(text_color_r_.get(), text_color_g_.get(), text_color_b_.get());

      if (do_text)
      {
	text_geometry_ =
	  renderer_->render_text(fld_handle,
				 color_map_.get_rep(),
				 text_use_default_color_.get(),
				 text_backface_cull_.get(),
				 text_fontsize_.get(),
				 text_precision_.get(),
				 text_render_locations_.get(),
				 text_show_data_.get(),
				 text_show_nodes_.get(),
				 text_show_edges_.get(),
				 text_show_faces_.get(),
				 text_show_cells_.get());
      }

      const char *name =
	text_backface_cull_.get()?"Culled Text Data":"Text Data";
      GeomHandle gmat =
	scinew GeomMaterial(text_geometry_, text_material_);
      GeomHandle geom =
	scinew GeomSwitch(scinew GeomColorMap(gmat, color_map_));
      text_id_ = ogeom_->addObj(geom, fname + name);
    }
  }

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
      do_execute = vectors_on_.get() || tensors_on_.get() || scalars_on_.get();
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
    else if (scalars_on_.get() && scalar_display_type_.get() != "Points")
    {
      data_dirty_ = true;
      maybe_execute(DATA);
    }
    else if (vectors_on_.get())
    {
      data_dirty_ = true;
      maybe_execute(DATA);
    }
  } else if (args[1] == "default_color_change") {
    def_color_r_.reset();
    def_color_g_.reset();
    def_color_b_.reset();
    def_color_a_.reset();
    def_material_->diffuse = 
      Color(def_color_r_.get(), def_color_g_.get(), def_color_b_.get());
    def_material_->transparency = def_color_a_.get();
    ogeom_->flushViews();
  } else if (args[1] == "text_color_change") {
    text_color_r_.reset();
    text_color_g_.reset();
    text_color_b_.reset();
    text_material_->diffuse =
      Color(text_color_r_.get(), text_color_g_.get(), text_color_b_.get());
    ogeom_->flushViews();
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
  } else if (args[1] == "toggle_display_scalars"){
    // Toggle the GeomSwitch.
    scalars_on_.reset();
    if ((scalars_on_.get()) && (data_id_ == 0))
    {
      data_dirty_ = true;
      maybe_execute(DATA);
    }
    else if (!scalars_on_.get() && data_id_)
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
  } else if (args[1] == "execute_policy"){
  } else if (args[1] == "calcdefs") {
    
    if (false) { //if (bounding_vector_) {
      //0.00896657
      double fact = 0.01; // * bounding_vector_->length();
      node_scale_.set(fact);
      edge_scale_.set(fact * 0.5);
      vectors_scale_.set(fact * 10);
      tensors_scale_.set(fact * 10);
      scalars_scale_.set(fact * 10);
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


