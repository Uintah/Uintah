/*
 *  AddLight.cc:
 *
 *  Written by:
 *   kuzimmer
 *   TODAY'S DATE HERE
 *
 */

#include <Core/Malloc/Allocator.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/share/share.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Geom/PointLight.h>
#include <Core/Geom/DirectionalLight.h>
#include <Core/Geom/SpotLight.h>
#include <Core/Math/Trig.h>

#include <iostream>
using std::cerr;
using std::endl;


#include <Dataflow/Widgets/LightWidget.h>


namespace SCIRun {


static string control_name("Control Widget");

class SCICORESHARE AddLight : public Module {
public:
  AddLight(GuiContext*);

  virtual ~AddLight();

  virtual void widget_moved(bool last);   

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  CrowdMonitor     control_lock_;
  LightWidget      *light_widget_;
  LightHandle      l;
  GeomID           control_id_;
  LightID          light_id_;
  GuiInt           control_pos_saved_;
  GuiDouble        control_x_;
  GuiDouble        control_y_;
  GuiDouble        control_z_;
  GuiDouble        at_x_;
  GuiDouble        at_y_;
  GuiDouble        at_z_;
  GuiDouble        rad_;
  GuiInt           light_type_;
  GuiInt           light_on_;
};

} // End namespace SCIRun

using namespace SCIRun;


DECLARE_MAKER(AddLight)
AddLight::AddLight(GuiContext* ctx)
  : Module("AddLight", ctx, Source, "Visualization", "SCIRun"),
    control_lock_("AddLight Position Control"),
    light_widget_(0),
    l(0),
    control_id_(-1),
    light_id_(-1),
    control_pos_saved_(ctx->subVar("control_pos_saved")),
    control_x_(ctx->subVar("control_x")),
    control_y_(ctx->subVar("control_y")),
    control_z_(ctx->subVar("control_z")),
    at_x_(ctx->subVar("at_x")),
    at_y_(ctx->subVar("at_y")),
    at_z_(ctx->subVar("at_z")),
    rad_(ctx->subVar("rad")),
    light_type_(ctx->subVar("type")),
    light_on_(ctx->subVar("on"))
{
  
}

AddLight::~AddLight()
{
}

void
AddLight::execute()
{
  static LightType lt = DirectionalLight;
  bool need_new_light = false;
  GeometryOPort *ogeom_ = (GeometryOPort *)get_oport("Geometry");
  
  if (!ogeom_) {
    error("Unable to initialize oport 'Geometry'.");
    return;
  }
  
  if(!light_widget_){
    light_widget_=scinew LightWidget(this, &control_lock_, 0.2);
    
    if( control_pos_saved_.get()== 1 ) {
      light_widget_->Move(Point(control_x_.get(),
				control_y_.get(),
				control_z_.get()));
      light_widget_->SetPointAt( Point(at_x_.get(),
				       at_y_.get(),
				       at_z_.get())); 
      light_widget_->SetRadius( rad_.get() );
      lt = (LightType)light_type_.get();
      light_widget_->SetLightType( lt );
    }
    
    light_widget_->Connect(ogeom_);
  }

  if( control_id_ == -1 ){
    GeomObj *w=light_widget_->GetWidget();
    control_id_ = ogeom_->addObj( w, control_name, &control_lock_);
  }
  
  if( light_id_ == -1 ||
      lt != light_widget_->GetLightType() || l.get_rep() == 0 ){
    need_new_light = true;    
//     ogeom_->delLight( light_id_, 0);
//     light_id_ = -1;
  }
  
  if( light_widget_->GetLightType() !=  (LightType)light_type_.get() ){
    light_widget_->SetLightType((LightType)light_type_.get());
    return;
  }

  Point location( light_widget_->GetSource() );
  Point dirpoint( light_widget_->GetPointAt() );
  Vector direction( light_widget_->GetAxis() );
  double radius = light_widget_->GetRadius();
  double d = (dirpoint - location).length();
  float cutoff = (float)(( d == 0 ) ? 0 : atan( radius/d )* 180/Pi);
  
  switch( light_widget_->GetLightType() ){
  case DirectionalLight:
//     cerr<<"DirectionalLight with location = "<< location <<", direction = "
//      	<<direction<<"\n";
    if (need_new_light){
      l = new class DirectionalLight( "my directional light", (-direction),
				      Color(1,1,1) );
      
    } else {
      class DirectionalLight* dl;
      dl = dynamic_cast<class DirectionalLight *>(l.get_rep());
      dl->move( (-direction) );
      ogeom_->flushViews();
    }
    break;
  case PointLight:
//     cerr<<"PointLight with location = "<<location<<"\n";
    if (need_new_light){
      l = new class PointLight( "my point light", location, Color(1,1,1));
    } else {
      class PointLight* pl;
      pl = dynamic_cast<class PointLight *>(l.get_rep());
      pl->move( location );
      ogeom_->flushViews();
    }
    break;
  case SpotLight:
//     cerr<<"SpotLight with location = "<<location<<", direction = "<<
//       direction<<", cutoff = "<<cutoff<<"\n";
    if (need_new_light){
      l = new class SpotLight( "my spot light", location, direction, cutoff,
			       Color(1,1,1));
    } else {
      class SpotLight * sl;
      sl = dynamic_cast<class SpotLight *>(l.get_rep());
      sl->move( location );
      sl->setDirection( direction );
      sl->setCutoff( cutoff );
      ogeom_->flushViews();
    }
    break;
  default:
    warning("Unknown Light type");
  }
  
  if( light_on_.get() == 1)
    l->on = true;
  else
    l->on = false;

  if ( need_new_light ){
    if( light_id_ != -1 ){
      ogeom_->delLight( light_id_, 0);
    }
    light_id_ = ogeom_->addLight( l, "my light", &control_lock_);
    lt = light_widget_->GetLightType();
  } 
  
}

void
AddLight::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

void 
AddLight::widget_moved(bool)
{
  Point w(light_widget_->ReferencePoint());
  control_x_.set( w.x() );
  control_y_.set( w.y() );
  control_z_.set( w.z() );
  Point dp( light_widget_->GetPointAt() );
  at_x_.set( dp.x() );
  at_y_.set( dp.y() );
  at_z_.set( dp.z() );
  rad_.set( light_widget_->GetRadius());
  light_type_.set( (int)light_widget_->GetLightType() );
  control_pos_saved_.set( 1 );
  want_to_execute();
}




