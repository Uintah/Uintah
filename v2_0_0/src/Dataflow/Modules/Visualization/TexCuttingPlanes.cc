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
#include <Core/Geom/ColorMap.h>
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
			 
DECLARE_MAKER(TexCuttingPlanes)

TexCuttingPlanes::TexCuttingPlanes(GuiContext* ctx) : 
  Module("TexCuttingPlanes", ctx, Filter, "Visualization", "SCIRun"), 
  tex_(0),
  control_lock_("TexCuttingPlanes resolution lock"),
  control_widget_(0),
  control_id_(-1),
  control_pos_saved_(ctx->subVar("control_pos_saved")),
  control_x_(ctx->subVar("control_x")),
  control_y_(ctx->subVar("control_y")),
  control_z_(ctx->subVar("control_z")),
  drawX_(ctx->subVar("drawX")),
  drawY_(ctx->subVar("drawY")),
  drawZ_(ctx->subVar("drawZ")),
  drawView_(ctx->subVar("drawView")),
  interp_mode_(ctx->subVar("interp_mode")),
  draw_phi0_(ctx->subVar("draw_phi_0")),
  draw_phi1_(ctx->subVar("draw_phi_1")),
  phi0_(ctx->subVar("phi_0")),
  phi1_(ctx->subVar("phi_1")),
  cyl_active_(ctx->subVar("cyl_active")),
  volren_(0)
{
}

TexCuttingPlanes::~TexCuttingPlanes()
{

}
void 
TexCuttingPlanes::tcl_command( GuiArgs& args, void* userdata)
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
	GeometryData* data = ogeom_->getData(0, 0, 1);
	Vector view = data->view->lookat() - data->view->eyep();
	view.normalize();
	w += view*ddview_*atof(args[3].c_str());
      }
      control_widget_->SetPosition(w);
      widget_moved(true);
      control_x_.set( w.x() );
      control_y_.set( w.y() );
      control_z_.set( w.z() );
      control_pos_saved_.set( 1 );
      ogeom_->flushViews();				  
  } else {
    Module::tcl_command(args, userdata);
  }
}

void TexCuttingPlanes::widget_moved(bool)
{
  if( volren_ ){
      volren_->SetControlPoint(tex_->get_field_transform().unproject(control_widget_->ReferencePoint()));
    }
  Point w(control_widget_->ReferencePoint());
  control_x_.set( w.x() );
  control_y_.set( w.y() );
  control_z_.set( w.z() );
  control_pos_saved_.set( 1 );
}


void TexCuttingPlanes::execute(void)
{
  intexture_ = (GLTexture3DIPort *)get_iport("GL Texture");
  icmap_ = (ColorMapIPort *)get_iport("ColorMap");
  ogeom_ = (GeometryOPort *)get_oport("Geometry");
  ocmap_ = (ColorMapOPort *)get_oport("ColorMap");

  if (!intexture_) {
    error("Unable to initialize iport 'GL Texture'.");
    return;
  }
  if (!icmap_) {
    error("Unable to initialize iport 'Color Map'.");
    return;
  }
  if (!ogeom_) {
    error("Unable to initialize oport 'Geometry'.");
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
  if( !icmap_->get(cmap)){
    return;
  }
  
  
  if(!control_widget_){
    control_widget_=scinew PointWidget(this, &control_lock_, 0.2);
    control_widget_->Connect(ogeom_);
    
    BBox b;
    tex_->get_bounds(b);
    Vector dv(b.diagonal());
    if( control_pos_saved_.get() ) {
      control_widget_->SetPosition(Point(control_x_.get(),
					 control_y_.get(),
					 control_z_.get()));
      control_widget_->SetScale(dv.length()/80.0);
    } else {
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
  }
    
    //AuditAllocator(default_allocator);
  if( !volren_ ){
    volren_ = scinew GLVolumeRenderer(tex_, cmap);
    
    volren_->SetControlPoint(tex_->get_field_transform().unproject(control_widget_->ReferencePoint()));
    volren_->SetInterp( bool(interp_mode_.get()));
    
//     if(tex_->CC()){
//       volren_->SetInterp(false);
//       interp_mode_.set(0);
//     } else {
      volren_->SetInterp(interp_mode_.get());
//     }

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
      GeomHandle w = control_widget_->GetWidget();
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

  if (!ocmap_) {
    error("Unable to initialize oport 'Color Map'.");
    return;
  } else {
    ColorMapHandle outcmap;
    outcmap = new ColorMap(*cmap.get_rep()); 
    double min, max;
    tex_->getminmax(min, max);
    outcmap->Scale(min, max);
    ocmap_->send(outcmap);
  }    

}

} // End namespace SCIRun


