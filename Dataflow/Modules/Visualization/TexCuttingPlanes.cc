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
 * TexCuttingPlanes.cc
 *
 * Simple interface to volume rendering stuff
 */

#include "TexCuttingPlanes.h"

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/View.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Dataflow/Widgets/PointWidget.h>
#include <iostream>
#include <algorithm>
#include <Core/GLVolumeRenderer/VolumeUtils.h>



namespace SCIRun {


static string control_name("Control Widget");
			 
extern "C" Module* make_TexCuttingPlanes( const string& id) {
  return scinew TexCuttingPlanes(id);
}


TexCuttingPlanes::TexCuttingPlanes(const string& id) : 
  Module("TexCuttingPlanes", id, Filter, "Visualization", "SCIRun"), 
  tex_(0),
  control_lock_("TexCuttingPlanes resolution lock"),
  control_widget_(0),
  control_id_(-1),
  drawX_("drawX", id, this),
  drawY_("drawY", id, this),
  drawZ_("drawZ", id, this),
  drawView_("drawView", id, this),
  interp_mode_("interp_mode", id, this),
  draw_phi0_("draw_phi_0", id, this),
  draw_phi1_("draw_phi_1", id, this),
  phi0_("phi_0", id, this),
  phi1_("phi_1", id, this),
  cyl_active_("cyl_active", id, this),
  volren_(0)
{
}

TexCuttingPlanes::~TexCuttingPlanes()
{

}
void 
TexCuttingPlanes::tcl_command( TCLArgs& args, void* userdata)
{
  if (args[1] == "MoveWidget") {
      if (!control_widget_) return;
      Point w(control_widget_->ReferencePoint());
      if (args[2] == "xplus") {
	w+=ddx_*atof(args[3].c_str());
      } else if (args[2] == "xat") {
	w=dmin_+ddx_*atof(args[3].c_str());
      } else if (args[2] == "yplus") {
	w+=ddy_*atof(args[3].c_str());
      } else if (args[2] == "yat") {
	w=dmin_+ddy_*atof(args[3].c_str());
      } else if (args[2] == "zplus") {
	w+=ddz_*atof(args[3].c_str());
      } else if (args[2] == "zat") {
	w=dmin_+ddz_*atof(args[3].c_str());
      } else if (args[2] == "vplus"){
	GeometryData* data = ogeom_->getData( 0, 1);
	Vector view = data->view->lookat() - data->view->eyep();
	view.normalize();
	w += view*ddview_*atof(args[3].c_str());
      }
      control_widget_->SetPosition(w);
      widget_moved(1);
      ogeom_->flushViews();				  
  } else {
    Module::tcl_command(args, userdata);
  }
}

void TexCuttingPlanes::widget_moved(int)
{
  if( volren_ ){
      volren_->SetControlPoint(tex_->get_field_transform().unproject(control_widget_->ReferencePoint()));
    }
}


void TexCuttingPlanes::execute(void)
{
  intexture_ = (GLTexture3DIPort *)get_iport("GL Texture");
  incolormap_ = (ColorMapIPort *)get_iport("Color Map");
  ogeom_ = (GeometryOPort *)get_oport("Geometry");

  if (!intexture_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!incolormap_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!ogeom_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  
  //AuditAllocator(default_allocator);
  static GLTexture3DHandle oldtex = 0;
  if (!intexture_->get(tex_)) {
    return;
  }
  else if (!tex_.get_rep()) {
    return;
  }
  
  ColorMapHandle cmap;
  if( !incolormap_->get(cmap)){
    return;
  }


  if(!control_widget_){
    control_widget_=scinew PointWidget(this, &control_lock_, 0.2);
    
    BBox b;
    tex_->get_bounds(b);
    Vector dv(b.diagonal());
    int nx, ny, nz;
    Transform t(tex_->get_field_transform());
    dmin_=t.project(Point(0,0,0));
    ddx_=t.project(Point(1,0,0))-dmin_;
    ddy_=t.project(Point(0,1,0))-dmin_;
    ddz_=t.project(Point(0,0,1))-dmin_;
    tex_->get_dimensions(nx,ny,nz);
    ddview_ = (dv.length()/(std::max(nx, std::max(ny,nz)) -1));
    control_widget_->SetPosition(Interpolate(b.min(), b.max(), 0.5));
    control_widget_->SetScale(dv.length()/80.0);
  }


  //AuditAllocator(default_allocator);
  if( !volren_ ){
    volren_ = scinew GLVolumeRenderer(0x12345676,
				  tex_,
				  cmap);

    volren_->SetControlPoint(tex_->get_field_transform().unproject(control_widget_->ReferencePoint()));
    volren_->SetInterp( bool(interp_mode_.get()));

    if(tex_->CC()){
      volren_->SetInterp(false);
      interp_mode_.set(0);
    } else {
      volren_->SetInterp(interp_mode_.get());
    }

    ogeom_->addObj( volren_, "Volume Slicer");
    volren_->set_tex_ren_state(GLVolumeRenderer::TRS_GLPlanes);
    volren_->DrawPlanes();
    cyl_active_.reset();
    draw_phi0_.reset();
    phi0_.reset();
    draw_phi1_.reset();
    phi1_.reset();
    volren_->set_cylindrical(cyl_active_.get(), draw_phi0_.get(), phi0_.get(), 
			     draw_phi1_.get(), phi1_.get());
  } else {    
    if( tex_.get_rep() != oldtex.get_rep() ){
      oldtex = tex_;
      BBox b;
      tex_->get_bounds(b);
      Vector dv(b.diagonal());
      int nx, ny, nz;
      Transform t(tex_->get_field_transform());
      dmin_=t.project(Point(0,0,0));
      ddx_=t.project(Point(1,0,0))-dmin_;
      ddy_=t.project(Point(0,1,0))-dmin_;
      ddz_=t.project(Point(0,0,1))-dmin_;
      tex_->get_dimensions(nx,ny,nz);
      ddview_ = (dv.length()/(std::max(nx, std::max(ny,nz)) -1));
      if (!b.inside(control_widget_->GetPosition())) {
	control_widget_->SetPosition(Interpolate(b.min(), b.max(), 0.5));
	control_widget_->SetScale(dv.length()/80.0);
      }
      volren_->SetVol( tex_.get_rep() );
      volren_->SetControlPoint(tex_->get_field_transform().unproject(control_widget_->ReferencePoint()));
    }

    volren_->SetInterp( bool(interp_mode_.get()));
    volren_->SetColorMap( cmap.get_rep() );
  }
 
  //AuditAllocator(default_allocator);
  if(drawX_.get() || drawY_.get() || drawZ_.get()){
    if( control_id_ == -1 ){
      GeomObj *w=control_widget_->GetWidget();
      control_id_ = ogeom_->addObj( w, control_name, &control_lock_);
    }
  } else {
    if( control_id_ != -1){
      ogeom_->delObj( control_id_, 0);
      control_id_ = -1;
    }
  }  

  volren_->SetX(drawX_.get());
  volren_->SetY(drawY_.get());
  volren_->SetZ(drawZ_.get());
  volren_->SetView(drawView_.get());
  cyl_active_.reset();
  draw_phi0_.reset();
  phi0_.reset();
  draw_phi1_.reset();
  phi1_.reset();
  volren_->set_cylindrical(cyl_active_.get(), draw_phi0_.get(), phi0_.get(), 
			   draw_phi1_.get(), phi1_.get());
  //AuditAllocator(default_allocator);

  ogeom_->flushViews();				  
  //AuditAllocator(default_allocator);
}

} // End namespace SCIRun


