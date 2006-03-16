//
//  For more information, please see: http://software.sci.utah.edu
//
//  The MIT License
//
//  Copyright (c) 2005 Scientific Computing and Imaging Institute,
//  University of Utah.
//
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//    File   : ParticleFlow.cc
//    Author : Kurt Zimmerman
//    Date   : Tues April 19 10:36:18 2005

#include <unistd.h>

#include <Core/Geom/ShaderProgramARB.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/ImageMesh.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Scheduler.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <Core/Util/NotFinished.h>
#include <Core/Volume/VideoCardInfo.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/FrameWidget.h>

#include <Packages/Uintah/Dataflow/Modules/Visualization/ParticleFlowRenderer.h>

namespace Uintah {
using namespace SCIRun;

class ParticleFlow : public Module
{
public:
  ParticleFlow(GuiContext*);
  virtual ~ParticleFlow();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);

  // This is a callback made by the scheduler when the network
  // finishes.  It should ask for a reexecute if the module and
  // increment the timestep if animate is on.
  static bool network_finished(void* ts_);

  void update_animate();

  // Inherited from Module.  I need this to setup the callback
  // function for the scheduler.
  virtual void set_context(Network* network);
  
  virtual void widget_moved( bool last, BaseWidget*);
  
private:
  int vfield_prev_generation_;
  int cmap_prev_generation_;
  int card_mem_;
  GuiInt gui_animate_;
  GuiDouble gui_time_;

  ParticleFlowRenderer *flow_ren_;

  CrowdMonitor widget_lock_;
  FrameWidget *frame_;
  int widgetid_;
  BBox frame_bbox_;

  void build_widget(FieldHandle f);
  bool bbox_similar_to(const BBox &a, const BBox &b);
};

DECLARE_MAKER(ParticleFlow)

static bool
check_ratio(double x, double y, double lower, double upper)
{
  if (fabs(x) < 1e-6) {
    if (!(fabs(y) < 1e-6))
      return false;
  } else {
    const double ratio = y / x;
    if (ratio < lower || ratio > upper)
      return false;
  }
  return true;
}

ParticleFlow::ParticleFlow(GuiContext* ctx)
  : Module("ParticleFlow", ctx, Source, "Visualization", "Uintah"),
    vfield_prev_generation_(-1),
    cmap_prev_generation_(-1),
    card_mem_(video_card_memory_size()),
    gui_animate_(ctx->subVar("animate")),
    gui_time_(ctx->subVar("time")),
    flow_ren_(0),
    widget_lock_("ParticleFlow widget lock")
{
  frame_ = scinew FrameWidget(this, &widget_lock_, 1.0);
  frame_->Connect((GeometryOPort*)get_oport("Geometry"));
  frame_bbox_.reset();
  frame_bbox_.extend(Point(-1.0, -1.0, -1.0));
  frame_bbox_.extend(Point(1.0, 1.0, 1.0));


}

ParticleFlow::~ParticleFlow()
{
 delete flow_ren_;
 delete frame_;
  // Remove the callback
  sched->remove_callback(network_finished, this);
}

// This is a callback made by the scheduler when the network finishes.
// It should ask for a reexecute if the module and increment the
// timestep if animate is on.
bool ParticleFlow::network_finished(void* ts_) {
  ParticleFlow* ts = (ParticleFlow*)ts_;
  ts->update_animate();
  return true;
}
void
ParticleFlow::widget_moved(bool last, BaseWidget*)
{
  if (last) {
    if( flow_ren_ ){
      Point c, r, d;
      frame_->GetPosition( c, r, d);
      flow_ren_->update_transform( c, r, d );
    }
    want_to_execute();
  }
}

void ParticleFlow::update_animate() {
  if( gui_animate_.get() ) {
    want_to_execute();
  }    
}

void ParticleFlow::set_context(Network* network) {
  Module::set_context(network);
  // Set up a callback to call after we finish
  sched->add_callback(network_finished, this);
}

void
ParticleFlow::execute()
{
  static GeomID geomID  = 0;

  FieldIPort* ivfield = (FieldIPort *)get_iport("Vector Field");
  ColorMapIPort* icmap = (ColorMapIPort*)get_iport("ColorMap");
  GeometryOPort* ogeom = (GeometryOPort *)get_oport("Geometry");
  ColorMapOPort* ocmap = (ColorMapOPort*)get_oport("ColorMap");

  FieldHandle vfield;
  ivfield->get(vfield);
  if (!vfield.get_rep())
  {
    error("Field has no representation.");
    return;
  }

  bool field_dirty = false;

  if (vfield->generation != vfield_prev_generation_)
  {
    // new field or range change
    VectorFieldInterfaceHandle vfi = vfield->query_vector_interface(this);
    if (!vfi.get_rep())
    {
      error("Input field does not contain vector data.");
      return;
    }

    const TypeDescription *td = vfield->get_type_description();

    if( td->get_name().find("Vector") != string::npos )
    {
      remark("Input data is Vector data.");
    } else {
      error("Input is not Vector data.");
      return;
    }

    if( td->get_name().find("LatVolMesh") != string::npos)
    {
      remark("LatVolMesh is allowed.");
    } else {
      error("Can only handle LatVolMeshes at this point.");
      return;
    }

    vfield_prev_generation_ = vfield->generation;
    field_dirty = true;

    build_widget(vfield);

  }

  // check for shaders
  if( !ShaderProgramARB::shaders_supported())
  {
    error("Shaders are not supported on this platform. Nothing drawn.");
    return;
  }

  if( flow_ren_ == 0 ){
    flow_ren_ = scinew ParticleFlowRenderer();
    if( flow_ren_ == 0 ) return;
    flow_ren_->update_vector_field( vfield );
    geomID = ogeom->addObj(flow_ren_, "ParticleFlowRenderer" );
    Point c, r, d;
    frame_->GetPosition(c,r,d);
    flow_ren_->update_transform( c, r, d );
  }
  
 

  flow_ren_->set_animation( bool(gui_animate_.get()) );
  if( gui_animate_.get() ){
        reset_vars();
        flow_ren_->set_time_increment( float(gui_time_.get()));
  }

  ColorMapHandle cmap;
  bool have_cmap = (icmap->get(cmap) && cmap.get_rep());
  if( have_cmap ){
    flow_ren_->update_colormap( cmap );
  }

  bool cmap_dirty = false;
  if( have_cmap && (cmap->generation != cmap_prev_generation_))
  {
    cmap_prev_generation_ = cmap->generation;
    cmap_dirty = true;
  }

  flow_ren_->update_vector_field( vfield );

  
  ogeom->flushViews();
  
}

void
ParticleFlow::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


void
ParticleFlow::build_widget(FieldHandle f)
{

  // for now default position is minimum Z value, and Normal = (0,0,1);
  BBox b = f->mesh()->get_bounding_box();
  bool resize = frame_bbox_.valid() && !bbox_similar_to(frame_bbox_, b);


  if( resize ) {
    double s;
    Point c, r, d;
    Vector diag(b.max() - b.min());
    c = (b.min() + diag * 0.5);
    c.y( b.min().y());
    r = c + Vector(c.x(), 0, 0);
    d = c + Vector(0.0, 0.0, c.z());
    s = diag.y() * 0.03;


    // Apply the new coordinates.
    frame_->SetScale(s); // do first, widget_moved resets
    frame_->SetPosition(c, r, d);
    frame_bbox_ = b;
  }

  
//   frame_->SetPosition( Point( c.x(), c.y(), b.min().z()), Vector(0,0,1),
//                        diag.x(), diag.y());

  GeomGroup *wg = scinew GeomGroup;
  wg->add(frame_->GetWidget());

  GeometryOPort *ogport = (GeometryOPort*)get_oport("Geometry");
  widgetid_ = ogport->addObj(wg,"ParticleFlow widget",
			     &widget_lock_);
  ogport->flushViews();


}

bool
ParticleFlow::bbox_similar_to(const BBox &a, const BBox &b)
{
  return
    a.valid() &&
    b.valid() &&
    check_ratio(a.min().x(), b.min().x(), 0.5, 2.0) &&
    check_ratio(a.min().y(), b.min().y(), 0.5, 2.0) &&
    check_ratio(a.min().z(), b.min().z(), 0.5, 2.0) &&
    check_ratio(a.min().x(), b.min().x(), 0.5, 2.0) &&
    check_ratio(a.min().y(), b.min().y(), 0.5, 2.0) &&
    check_ratio(a.min().z(), b.min().z(), 0.5, 2.0);
}

} // end namespace Uintah

