/*
 *  ParticleFlow.cc:
 *
 *  Written by:
 *   kuzimmer
 *   TODAY'S DATE HERE
 *
 */

#include <unistd.h>

#include <Core/Geom/ShaderProgramARB.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/ImageMesh.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Scheduler.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>

#include <Core/Util/NotFinished.h>
#include <Core/Volume/VideoCardInfo.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/FrameWidget.h>

#include <Packages/Kurt/Dataflow/Modules/Visualization/ParticleFlowRenderer.h>

namespace Kurt {

using namespace SCIRun;

class ParticleFlow : public Module {
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
  GuiInt gui_nsteps_;
  GuiDouble gui_step_size_;
  GuiInt gui_n_particles_;
  GuiInt gui_recompute_points_;
  GuiInt gui_freeze_particles_;

  GuiPoint gui_center_;
  GuiPoint gui_right_;
  GuiPoint gui_down_;
  GuiDouble gui_widget_scale_;
  GuiInt   gui_widget_reset_;


  ParticleFlowRenderer *flow_ren_;

  CrowdMonitor widget_lock_;
  FrameWidget *frame_;
  int widgetid_;
  BBox frame_bbox_;

  void build_widget(FieldHandle f);
  bool bbox_similar_to(const BBox &a, const BBox &b);
};


DECLARE_MAKER(ParticleFlow)
ParticleFlow::ParticleFlow(GuiContext* ctx)
  : Module("ParticleFlow", ctx, Source, "Visualization", "Kurt"),
    vfield_prev_generation_(-1),
    cmap_prev_generation_(-1),
    card_mem_(video_card_memory_size()),
    gui_animate_(ctx->subVar("animate"), 0),
    gui_time_(ctx->subVar("time"), -0.05),
    gui_nsteps_(ctx->subVar("nsteps"), 32),
    gui_step_size_(ctx->subVar("step-size"), 1.0),
    gui_n_particles_(ctx->subVar("nparticles"), 1),
    gui_recompute_points_(ctx->subVar("recompute-points"), 0),
    gui_freeze_particles_(ctx->subVar("freeze-particles"), 0),
    gui_center_(ctx->subVar("center"), Point(0,0,0)),
    gui_right_(ctx->subVar("right"), Point(0,0,0)),
    gui_down_(ctx->subVar("down"), Point(0,0,0)),
    gui_widget_scale_(ctx->subVar("widget-scale"), 0),
    gui_widget_reset_(ctx->subVar("widget-reset"), 0),
    flow_ren_(0),
    widget_lock_("ParticleFlow widget lock"),
    frame_(0)
{
  frame_ = scinew FrameWidget(this, &widget_lock_, 1.0);
  frame_->Connect((GeometryOPort*)get_oport("ParticleFlow Renderer"));
  frame_bbox_.reset();
  frame_bbox_.extend(Point(-1.0, -1.0, -1.0));
  frame_bbox_.extend(Point(1.0, 1.0, 1.0));
}


ParticleFlow::~ParticleFlow(){
 delete flow_ren_;
 delete frame_;
  // Remove the callback
  sched_->remove_callback(network_finished, this);
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
      double s;
      Point c, r, d;
      frame_->GetPosition( c, r, d);
      s = frame_->GetScale();

      gui_center_.set( c );
      gui_right_.set(r);
      gui_down_.set(d);
      gui_widget_scale_.set( s );

//       cerr<<"frame state is "<<frame_->GetStateString()<<"\n";
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
  sched_->add_callback(network_finished, this);
}

void
 ParticleFlow::execute()
{
  static GeomID geomID  = 0;

  FieldIPort* ivfield = (FieldIPort *)get_iport("Vector Flow Field");
  ColorMapIPort* icmap = (ColorMapIPort*)get_iport("ColorMap");
  GeometryOPort* ogeom = (GeometryOPort *)get_oport("ParticleFlow Renderer");
  ColorMapOPort* ocmap = (ColorMapOPort*)get_oport("ColorMap");

  FieldHandle vfield;
  ivfield->get(vfield);
  if (!vfield.get_rep())
  {
    error("Field has no representation.");
    return;
  }

  bool field_dirty = false;

  if (vfield->generation != vfield_prev_generation_ || gui_widget_reset_.get())
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
//     cerr<<"frame state is "<<frame_->GetStateString()<<"\n";
    flow_ren_->update_transform( c, r, d );
  } else {
    if( field_dirty ){
      flow_ren_->update_vector_field( vfield );
    }
  }
  
 

  flow_ren_->set_animation( bool(gui_animate_.get()) );
  flow_ren_->set_time_increment( float(gui_time_.get()));
  flow_ren_->set_nsteps( (unsigned int)( gui_nsteps_.get()));
  flow_ren_->set_step_size( gui_step_size_.get());
  flow_ren_->recompute_points( bool(gui_recompute_points_.get()));
  flow_ren_->freeze( bool( gui_freeze_particles_.get()));
  flow_ren_->set_particle_exponent( gui_n_particles_.get() );
  gui_recompute_points_.set(0);  // default to false
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

  ogeom->flushViews();


}


// goes with bbox_similar_to()  below...
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

void
ParticleFlow::build_widget(FieldHandle f)
{

  // for now default position is minimum Z value, and Normal = (0,0,1);
  BBox b = f->mesh()->get_bounding_box();
//   bool resize = frame_bbox_.valid() && !bbox_similar_to(frame_bbox_, b);


//   if( resize ) {

  if( gui_widget_reset_.get() == 1 || gui_widget_scale_.get() <= 0 ||
      (gui_center_.get() == Point(0,0,0) &&
       gui_right_.get() == Point(0,0,0) &&
       gui_down_.get() == Point(0,0,0) )) {

    if(!b.valid()){
      warning("Input box is invalid -- using unit cube.");
      b.extend(Point(0,0,0));
      b.extend(Point(1,1,1));
    }

    double s;
    Point c, r, d;
    Vector diag(b.max() - b.min());
    c = (b.min() + diag * 0.5);
    //     c.y( b.min().y());
    r = c + Vector(diag.x()*0.5, 0, 0);
    d = c + Vector(0.0, 0.0, diag.z()*0.5);
    // Apply the new coordinates.

    frame_->SetPosition(c, r, d);

    s = diag.length() * 0.015;

    frame_->SetScale(s); // do first, widget_moved resets
    frame_bbox_ = b;
    gui_center_.set( c );
    gui_right_.set( r );
    gui_down_.set( d );
    gui_widget_scale_.set( s );
    gui_widget_reset_.set( 0 );

  } else {
    const double l2norm = (gui_right_.get().vector() +
			   Cross(gui_down_.get().vector(), 
                                 gui_right_.get().vector()) +
			   gui_down_.get().vector()).length();
    const double s = l2norm * 0.015;
    
    double scale = gui_widget_scale_.get();
    if( scale < s * 1e-2 || scale > s * 1e2 ){
      scale = s;
    }

    frame_->SetPosition(gui_center_.get(), 
                        gui_right_.get(), 
                        gui_down_.get());
    frame_->SetScale( scale );
    
 
  }
//   }

  

  GeomGroup *wg = scinew GeomGroup;
  wg->add(frame_->GetWidget());

  GeometryOPort *ogport = (GeometryOPort*)get_oport("ParticleFlow Renderer");
  widgetid_ = ogport->addObj(wg,"ParticleFlow widget",
			     &widget_lock_);
  ogport->flushViews();


}

void
 ParticleFlow::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Kurt


