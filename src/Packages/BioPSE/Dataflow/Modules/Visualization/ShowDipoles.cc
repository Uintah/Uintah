/*
 *  ShowDipoles.cc:  Builds the RHS of the FE matrix for current sources
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   May 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Widgets/ArrowWidget.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <iostream>
#include <vector>


#include <Packages/BioPSE/share/share.h>

namespace BioPSE {

using namespace SCIRun;

class BioPSESHARE ShowDipoles : public Module {
public:
  ShowDipoles(GuiContext *context);
  virtual ~ShowDipoles();

  virtual void execute();
  virtual void widget_moved(bool last);
  virtual void tcl_command(GuiArgs& args, void* userdata);
private:
  void new_input_data(PointCloudField<Vector> *in);
  void scale_changed();
  void scale_mode_changed();
  bool generate_output_field();
  void draw_lines();
  void load_gui();
  void last_as_vec();

  CrowdMonitor             widget_lock_;
  vector<int>              widget_id_;
  Array1<ArrowWidget*>     widget_;
  int                      gidx_;
  GuiInt                   num_dipoles_;

  MaterialHandle           greenMatl_;
  MaterialHandle           deflMatl_;

  FieldIPort              *ifield_;
  FieldOPort              *ofield_;
  GeometryOPort           *ogeom_;

  FieldHandle              dipoleFldH_;
  GuiDouble                widgetSizeGui_;
  GuiString                scaleModeGui_;
  GuiInt                   showLastVecGui_;
  GuiInt                   showLinesGui_;
  int                      lastGen_;

  string                   execMsg_;
  vector<GeomHandle>       widget_switch_;
  vector<GuiPoint*>        new_positions_;
  vector<GuiVector*>       new_directions_;
  vector<GuiDouble*>       new_magnitudes_;
  GuiDouble                max_len_;
  bool                     output_dirty_;
  bool                     been_executed_;
  bool                     reset_;
  double                   last_scale_;
  string                   last_scale_mode_;
};


DECLARE_MAKER(ShowDipoles)


ShowDipoles::ShowDipoles(GuiContext *context) :
  Module("ShowDipoles", context, Filter, "Visualization", "BioPSE"),
  widget_lock_("ShowDipoles widget lock"),
  num_dipoles_(context->subVar("num-dipoles")),
  widgetSizeGui_(context->subVar("widgetSizeGui_")),
  scaleModeGui_(context->subVar("scaleModeGui_")),
  showLastVecGui_(context->subVar("showLastVecGui_")),
  showLinesGui_(context->subVar("showLinesGui_")),
  max_len_(context->subVar("max-len")),
  output_dirty_(true),
  been_executed_(false),
  reset_(false),
  last_scale_(0.0),
  last_scale_mode_("fixed")
{
  lastGen_=-1;
  greenMatl_ = new Material(Color(0.2, 0.8, 0.2));
  gidx_=0;
}

ShowDipoles::~ShowDipoles(){
}


void
ShowDipoles::execute()
{
  ifield_ = (FieldIPort *)get_iport("dipoleFld");
  ofield_ = (FieldOPort *)get_oport("dipoleFld");
  ogeom_ = (GeometryOPort *)get_oport("Geometry");

  if (!ifield_) {
    error("Unable to initialize iport 'dipoleFld'.");
    return;
  }
  if (!ofield_) {
    error("Unable to initialize oport 'dipoleFld'.");
    return;
  }
  if (!ogeom_) {
    error("Unable to initialize oport 'Geometry'.");
    return;
  }
  
  // if this is the first execution then try loading all the values
  // from saved GuiVars, load the widgets from these values
  if (!been_executed_) {
    load_gui();
    been_executed_ = true;
  }
  FieldHandle fieldH;
  PointCloudField<Vector> *field_pcv;
  if (!ifield_->get(fieldH) || 
      !(field_pcv=dynamic_cast<PointCloudField<Vector>*>(fieldH.get_rep()))) {
    error("No vald input in ShowDipoles Field port.");
    return;
  }
  PointCloudMeshHandle field_mesh = field_pcv->get_typed_mesh();
  
  int gen = fieldH->generation;
  
  reset_vars();
  if (reset_ || (gen != lastGen_))
  {
    lastGen_ = gen;
    if (reset_ || 
	(field_pcv->fdata().size() != (unsigned)num_dipoles_.get())) {
      new_input_data(field_pcv);
    }
  }

  //widget_moved(true);
  last_as_vec();
  draw_lines();
  generate_output_field();
  ogeom_->flushViews();
  ofield_->send(dipoleFldH_);
}

void 
ShowDipoles::load_gui()
{
  num_dipoles_.reset();
  new_positions_.resize(num_dipoles_.get(), 0);
  new_directions_.resize(num_dipoles_.get(), 0);
  new_magnitudes_.resize(num_dipoles_.get(), 0);
  for (int i = 0; i < num_dipoles_.get(); i++) {
    if (!new_positions_[i]) {
      ostringstream str;
      str << "newpos" << i;
      new_positions_[i] = new GuiPoint(ctx->subVar(str.str()));
      new_positions_[i]->reset();
    }
    if (!new_directions_[i]) {
      ostringstream str;
      str << "newdir" << i;
      new_directions_[i] = new GuiVector(ctx->subVar(str.str()));
      new_directions_[i]->reset();
    }
    if (!new_magnitudes_[i]) {
      ostringstream str;
      str << "newmag" << i;
      new_magnitudes_[i] = new GuiDouble(ctx->subVar(str.str()));
      new_magnitudes_[i]->reset();
    }

    // it is possible that these were created already, dont do it twice.
    if (widget_id_.size() != num_dipoles_.get()) {
      ArrowWidget *a = scinew ArrowWidget(this, &widget_lock_, 
					  widgetSizeGui_.get());
      a->Connect(ogeom_);
      a->SetCurrentMode(1);
      widget_.add(a);
      deflMatl_ = widget_[0]->GetMaterial(0);
      widget_switch_.push_back(widget_[i]->GetWidget());
      ((GeomSwitch *)(widget_switch_[i].get_rep()))->set_state(1);
      widget_id_.push_back(ogeom_->addObj(widget_switch_[i],
					  "Dipole" + to_string((int)i),
					  &widget_lock_));

      widget_[i]->SetPosition(new_positions_[i]->get());
      double str = new_magnitudes_[i]->get();
      widget_[i]->SetDirection(new_directions_[i]->get());
      double sc = widgetSizeGui_.get();
      string scaleMode = scaleModeGui_.get();
      if (scaleMode == "normalize") sc *= (str / max_len_.get());
      else if (scaleMode == "scale") sc *= str;
      widget_[i]->SetScale(sc);
      widget_[i]->SetLength(2 * sc);
    }
  }
  last_scale_ = widgetSizeGui_.get();
}

void 
ShowDipoles::new_input_data(PointCloudField<Vector> *in)
{
  num_dipoles_.reset();
  widgetSizeGui_.reset();
  showLastVecGui_.reset();
  reset_ = false;

  if (widget_switch_.size()) {
    widget_[num_dipoles_.get() - 1]->SetCurrentMode(1);
    widget_[num_dipoles_.get() - 1]->SetMaterial(0, deflMatl_);
  }
  // turn off any extra arrow widgets we might have.
  if (in->fdata().size() < (unsigned)num_dipoles_.get()) {
    for (int i = in->fdata().size(); i < num_dipoles_.get(); i++)
      ((GeomSwitch *)(widget_switch_[i].get_rep()))->set_state(0);
    num_dipoles_.set(in->fdata().size());
    num_dipoles_.reset();
  } else {
    unsigned i;
    for (i = num_dipoles_.get(); i < widget_switch_.size(); i++)
      ((GeomSwitch *)(widget_switch_[i].get_rep()))->set_state(1);
    for (; i < in->fdata().size(); i++) {
      ArrowWidget *a = scinew ArrowWidget(this, &widget_lock_, 
					  widgetSizeGui_.get());
      a->Connect(ogeom_);
      a->SetCurrentMode(1);
      widget_.add(a);
      deflMatl_ = widget_[0]->GetMaterial(0);
      widget_switch_.push_back(widget_[i]->GetWidget());
      ((GeomSwitch *)(widget_switch_[i].get_rep()))->set_state(1);
      widget_id_.push_back(ogeom_->addObj(widget_switch_[i],
					  "Dipole" + to_string((int)i),
					  &widget_lock_));

    }
    num_dipoles_.set(in->fdata().size());
    num_dipoles_.reset();
    load_gui();
  }
  
  if (showLastVecGui_.get()) {
    widget_[num_dipoles_.get() - 1]->SetCurrentMode(1);
    widget_[num_dipoles_.get() - 1]->SetMaterial(0, deflMatl_);
  } else {
    widget_[num_dipoles_.get() - 1]->SetCurrentMode(2);
    widget_[num_dipoles_.get() - 1]->SetMaterial(0, greenMatl_);
  }
  
  for (int i = 0; i < num_dipoles_.get(); i++) {
    Vector v(in->fdata()[i]);
    double dv = v.length();
    if (i == 0 || dv > max_len_.get()) {
      max_len_.set(dv);
      max_len_.reset();
    }
  }
  string scaleMode = scaleModeGui_.get();
  for (int i = 0; i < num_dipoles_.get(); i++) {
 
    PointCloudMeshHandle field_mesh = in->get_typed_mesh();
    Point p;
    field_mesh->get_point(p,i);
    new_positions_[i]->set(p);
    widget_[i]->SetPosition(p);
    Vector v(in->fdata()[i]);
    double str = v.length();
    new_magnitudes_[i]->set(str);
    if (str < 0.0000001) v.z(1);
    v.normalize();
    widget_[i]->SetDirection(v);
    new_directions_[i]->set(v);
    double sc = widgetSizeGui_.get();

    if (scaleMode == "normalize") sc *= (str / max_len_.get());
    else if (scaleMode == "scale") sc *= str;
    widget_[i]->SetScale(sc);
    widget_[i]->SetLength(2 * sc);
  }
  last_scale_ = widgetSizeGui_.get();
}

void 
ShowDipoles::last_as_vec()
{
  showLastVecGui_.reset();
  bool slv = showLastVecGui_.get();
  if (!num_dipoles_.get()) return;

  if (slv) {
    widget_[num_dipoles_.get() - 1]->SetCurrentMode(1);
    widget_[num_dipoles_.get() - 1]->SetMaterial(0, deflMatl_);
  } else {
    widget_[num_dipoles_.get() - 1]->SetCurrentMode(2);
    widget_[num_dipoles_.get() - 1]->SetMaterial(0, greenMatl_);
  }

  if (ogeom_) ogeom_->flushViews();
}


void
ShowDipoles::scale_mode_changed()
{
  scaleModeGui_.reset();
  string scaleMode = scaleModeGui_.get();
  if (scaleMode == last_scale_mode_) return;
  widgetSizeGui_.reset();
  max_len_.reset();
  double max = max_len_.get();
  for (int i = 0; i < num_dipoles_.get(); i++) {   
    double sc = widgetSizeGui_.get();

    if (scaleMode == "normalize") {
      if (last_scale_mode_ == "scale") {
	sc *= (widget_[i]->GetScale() / last_scale_) / max;
      } else {
	sc *= widget_[i]->GetScale() / max;
      }
    } else if (scaleMode == "scale") {
      if (last_scale_mode_ == "normalize") {
	sc *= (widget_[i]->GetScale() * max) / last_scale_;
      } else {
	sc *= widget_[i]->GetScale();
      }
    }
    new_magnitudes_[i]->set(sc);
    widget_[i]->SetScale(sc);
    widget_[i]->SetLength(2*sc);
  }
  last_scale_ = widgetSizeGui_.get();
  last_scale_mode_ = scaleMode;
  if (ogeom_) ogeom_->flushViews();
}

void
ShowDipoles::scale_changed()
{
  scaleModeGui_.reset();
  string scaleMode = scaleModeGui_.get();
  widgetSizeGui_.reset();
  max_len_.reset();

  for (int i = 0; i < num_dipoles_.get(); i++) {    
    double sc = widgetSizeGui_.get();
    if (scaleMode != "fixed") {
      sc *= widget_[i]->GetScale() / last_scale_;
    }
    new_magnitudes_[i]->set(sc);
    widget_[i]->SetScale(sc);
    widget_[i]->SetLength(2*sc);
  }
  last_scale_ = widgetSizeGui_.get();
  if (ogeom_) ogeom_->flushViews();
}


bool
ShowDipoles::generate_output_field()
{
  if (output_dirty_) {
    output_dirty_ = false;
    PointCloudMesh *msh = new PointCloudMesh;
    for (int i = 0; i < num_dipoles_.get(); i++) {      
      msh->add_node(new_positions_[i]->get());
    }
    PointCloudMeshHandle mh(msh);
    PointCloudField<Vector> *out = new PointCloudField<Vector>(mh, 
							       Field::NODE);
    scaleModeGui_.reset();
    string scaleMode = scaleModeGui_.get();
    double max = max_len_.get();
    for (int i = 0; i < num_dipoles_.get(); i++) {  
      Vector d = new_directions_[i]->get();
      double sc = 1.0f;

      if (scaleMode == "normalize") {
	sc = widgetSizeGui_.get() / max;
      }
      else if (scaleMode == "scale") {
	sc = widgetSizeGui_.get();
      }

      d *= widget_[i]->GetScale() / sc;
      out->fdata()[i] = d;
    }
    dipoleFldH_ = out;
    return true;
  }
  return false;
}


void
ShowDipoles::widget_moved(bool redraw)
{
  if (redraw) {
    output_dirty_ = true;
    scaleModeGui_.reset();
    string scaleMode = scaleModeGui_.get();
    double max = max_len_.get();
    double old_max = max;
    double new_max = 0.;   
    //may have a new max length for normalize.
    for (int i = 0; i < num_dipoles_.get(); i++) {
      double sc = 1.0f;
      
      if (scaleMode == "normalize") {
	sc = widgetSizeGui_.get() / max;
      }
      else if (scaleMode == "scale") {
	sc = widgetSizeGui_.get();
      }
      double mag = widget_[i]->GetScale() / sc;

      if (mag > new_max) {
	new_max = mag;
      }
    }
    max_len_.set(new_max);
    max_len_.reset();
    max = new_max;

    // dont know which widget moved so update all of them
    for (int i = 0; i < num_dipoles_.get(); i++) {

      new_positions_[i]->set(widget_[i]->GetPosition());
      new_directions_[i]->set(widget_[i]->GetDirection());

      double sc = 1.0f;

      if (scaleMode == "normalize") {
	// need to renormalize...
	double old = widget_[i]->GetScale();
	// undo the old normalize
	double vec = (old * old_max) / widgetSizeGui_.get();
	sc = widgetSizeGui_.get() * (vec / max);
	widget_[i]->SetScale(sc); 
	new_magnitudes_[i]->set(sc);
      } else {
	new_magnitudes_[i]->set(widget_[i]->GetScale());
      }

      new_positions_[i]->reset();
      new_directions_[i]->reset();
      new_magnitudes_[i]->reset();
    }
    last_scale_ = widgetSizeGui_.get();
    draw_lines();
    if (ogeom_) ogeom_->flushViews();
  }
}

void 
ShowDipoles::draw_lines()
{
  showLinesGui_.reset();
  if (gidx_ && ogeom_) { 
    ogeom_->delObj(gidx_); 
    gidx_=0; 
  }
  if (showLinesGui_.get()) 
  {
    GeomLines *g = new GeomLines;
    for (unsigned i = 0; i < new_positions_.size() - 2; i++) 
      for (unsigned j = i+1; j < new_positions_.size() - 1; j++) 
	g->add(new_positions_[i]->get(), new_positions_[j]->get());
    GeomMaterial *gm = new GeomMaterial(g, new Material(Color(.8,.8,.2)));
    gidx_ = ogeom_->addObj(gm, string("ShowDipole Lines"));
  }
  if(ogeom_) ogeom_->flushViews();
}

void 
ShowDipoles::tcl_command(GuiArgs& args, void* userdata) {
  if(args.count() < 2){
    args.error("ShowDipoles needs a minor command");
    return;
  }

  if (args[1] == "widget_scale") {
    scale_changed();
  } else if (args[1] == "scale_mode") {
    scale_mode_changed();
  } else if (args[1] == "show_last_vec") {
    last_as_vec();
  } else  if (args[1] == "show_lines") {
    draw_lines();
  } else  if (args[1] == "reset") {
    reset_ = true;
    want_to_execute();
  } else{
    Module::tcl_command(args, userdata);
  }
}


 
} // End namespace BioPSE
