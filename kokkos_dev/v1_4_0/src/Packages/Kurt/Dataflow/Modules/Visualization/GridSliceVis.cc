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
 * GridSliceVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include "GridSliceVis.h"

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/View.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Uintah/Core/Datatypes/LevelMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Dataflow/Widgets/PointWidget.h>
#include <iostream>
#ifdef __sgi
#include <ios>
#endif
#include <algorithm>
#include <Core/GLVolumeRenderer/VolumeUtils.h>



namespace Kurt {

using SCIRun::postMessage;
using SCIRun::Field;
using SCIRun::TCL;
using SCIRun::to_string;
using SCIRun::GeometryData;
using SCIRun::Allocator;
using SCIRun::AuditAllocator;
using SCIRun::DumpAllocator;
using SCIRun::LatVolMesh;
using SCIRun::Interpolate;
using Uintah::LevelMesh;

static string control_name("Control Widget");
			 
extern "C" Module* make_GridSliceVis( const string& id) {
  return scinew GridSliceVis(id);
}


GridSliceVis::GridSliceVis(const string& id)
  : Module("GridSliceVis", id, Filter, "Visualization", "Kurt"), 
  tex(0),
  control_lock("GridSliceVis resolution lock"),
  control_widget(0),
  control_id(-1),
  is_fixed_("is_fixed_", id, this),
  max_brick_dim_("max_brick_dim_",id, this),
  min_("min_", id ,this),
  max_("max_", id , this),
  drawX("drawX", id, this),
  drawY("drawY", id, this),
  drawZ("drawZ", id, this),
  drawView("drawView", id, this),
  interp_mode("interp_mode", id, this),
  point_x("point_x", id, this),
  point_y("point_y", id, this),
  point_z("point_z", id, this),
  point_init("pointk_init", id, this),
  sliceren(0),
  svr(0)
{
}

GridSliceVis::~GridSliceVis()
{

}
void 
GridSliceVis::tcl_command( TCLArgs& args, void* userdata)
{
  if (args[1] == "MoveWidget") {
      if (!control_widget) return;
      Point w(control_widget->ReferencePoint());
      if (args[2] == "xplus") {
	w+=Vector(ddv.x(), 0, 0);
      } else if (args[2] == "xminus") {
	w-=Vector(ddv.x(), 0, 0);
      } else if (args[2] == "yplus") {
	w+=Vector(0, ddv.y(), 0);
      } else if (args[2] == "yminus") {
	w-=Vector(0, ddv.y(), 0);
      } else if (args[2] == "zplus") {
	w+=Vector(0, 0, ddv.z());
      } else if (args[2] == "zminus"){
	w-=Vector(0, 0, ddv.z());
      } else if (args[2] == "vplus"){
	GeometryData* data = ogeom->getData( 0, 1);
	Vector view = data->view->lookat() - data->view->eyep();
	view.normalize();
	w += view*ddview;
      } else if (args[2] == "vminus"){
	GeometryData* data = ogeom->getData( 0, 1);
	Vector view = data->view->lookat() - data->view->eyep();
	view.normalize();
	w -= view*ddview;
      }
      control_widget->SetPosition(w);
      widget_moved(1);
  } else {
    Module::tcl_command(args, userdata);
  }
}

void GridSliceVis::widget_moved(int)
{
  if( sliceren ){
    Point w = control_widget->ReferencePoint();
    point_x.set( w.x() );
    point_y.set( w.y() );
    point_z.set( w.z() );
    sliceren->SetControlPoint(w);
  }
}


void GridSliceVis::execute(void)
{
  static FieldHandle old_tex = 0;
  static ColorMapHandle old_cmap = 0;
  static int old_brick_size = 0;
  static double old_min = 0;
  static double old_max = 1;
  static bool old_fixed = false;
  infield = (FieldIPort *)get_iport("Scalar Field");
  incolormap = (ColorMapIPort *)get_iport("Color Map");
  ogeom = (GeometryOPort *)get_oport("Geometry");

  if (!infield) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!incolormap) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!ogeom) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  
  //AuditAllocator(default_allocator);

  if (!infield->get(tex)) {
    return;
  }
  else if (!tex.get_rep()) {
    return;
  }
  
  ColorMapHandle cmap;
  if( !incolormap->get(cmap)){
    return;
  }

  Point Smin, Smax;
  int nx,ny,nz;
  if(!control_widget){
    control_widget=scinew PointWidget(this, &control_lock, 0.2);
    
    if(LevelMesh *mesh = dynamic_cast<LevelMesh *> (tex->mesh().get_rep()))
    {
      BBox bb = mesh->get_bounding_box();
      Smin = bb.min();
      Smax = bb.max();
      nx = mesh->get_nx();
      ny = mesh->get_ny();
      nz = mesh->get_nz();
    } else if( LatVolMesh *mesh =
	       dynamic_cast<LatVolMesh *> (tex->mesh().get_rep()))
    {
      BBox bb = mesh->get_bounding_box();
      Smin = bb.min();
      Smax = bb.max();
      nx = mesh->get_nx();
      ny = mesh->get_ny();
      nz = mesh->get_nz();
    } else {
      std::cerr<<"Unknown Field type\n";
      return;
    }
    Vector dv(Smax - Smin);
    ddv.x(dv.x()/(nx - 1));
    ddv.y(dv.y()/(ny - 1));
    ddv.z(dv.z()/(nz - 1));
    ddview = (dv.length()/(std::max(nx, std::max(ny,nz)) -1));
    if( point_init.get() ){
      control_widget->SetPosition( Point( point_x.get(),
					  point_y.get(),
					  point_z.get()) );
    } else {
      Point w( Interpolate(Smin,Smax,0.5) );
      control_widget->SetPosition( w );
      point_init.set(1);
      point_x.set( w.x() );
      point_y.set( w.y() );
      point_z.set( w.z() );

    }
    control_widget->SetScale(dv.length()/80.0);
  }


  //AuditAllocator(default_allocator);
  if( !sliceren ){
    svr = scinew GridSliceRen();
    sliceren = scinew SliceRenderer(0x12345676,svr, tex, cmap,
				   (is_fixed_.get() == 1),
				   min_.get(), max_.get());

    if(tex->data_at() == Field::CELL){
      sliceren->SetInterp(false);
      interp_mode.set(0);
    }
    cerr<<"Need to initialize sliceren\n";
    old_cmap = cmap;
    old_tex = tex;
    TCL::execute(id + " SetDims " + to_string( sliceren->get_brick_size()));
    max_brick_dim_.set(  sliceren->get_brick_size() );
    old_brick_size = max_brick_dim_.get();
    if( is_fixed_.get() !=1 ){
      sliceren->GetRange( old_min, old_max);
      min_.set(old_min);
      max_.set(old_max);
    }
    old_fixed = ( is_fixed_.get() == 1);
    ogeom->delAll();

    ogeom->addObj( sliceren, "Volume Slicer");
    //    sliceren->_GLPlanes();
    //    sliceren->DrawPlanes();
  } else {  
    bool needbuild = false;
    if( tex.get_rep() != old_tex.get_rep() ){
      old_tex = tex;
    if(LevelMesh *mesh = dynamic_cast<LevelMesh *> (tex->mesh().get_rep()))
    {
      BBox bb = mesh->get_bounding_box();
      Smin = bb.min();
      Smax = bb.max();
      nx = mesh->get_nx();
      ny = mesh->get_ny();
      nz = mesh->get_nz();
    } else if( LatVolMesh *mesh =
	       dynamic_cast<LatVolMesh *> (tex->mesh().get_rep()))
    {
      BBox bb = mesh->get_bounding_box();
      Smin = bb.min();
      Smax = bb.max();
      nx = mesh->get_nx();
      ny = mesh->get_ny();
      nz = mesh->get_nz();
    } else {
      std::cerr<<"Unknown Field type\n";
      return;
    }
      Vector dv(Smax - Smin);
      ddv.x(dv.x()/(nx - 1));
      ddv.y(dv.y()/(ny - 1));
      ddv.z(dv.z()/(nz - 1));
      ddview = (dv.length()/(std::max(nx, std::max(ny,nz)) -1));
      needbuild = true;
      sliceren->SetVol( tex.get_rep() );
    }

    if( max_brick_dim_.get() != old_brick_size ){
      sliceren->SetBrickSize(  max_brick_dim_.get() );
      old_brick_size = max_brick_dim_.get();
      needbuild = true;
    }
    if( is_fixed_.get() == 1 &&
	(old_min != min_.get() || old_max!= max_.get()) ){
      sliceren->SetRange( min_.get(), max_.get() );
      sliceren->FixedRange( true );
      old_min = min_.get();
      old_max = max_.get();
      old_fixed = ( is_fixed_.get() == 1);
      needbuild = true;
    }
    if( old_fixed == true && is_fixed_.get() == 0 ){
      old_fixed = ( is_fixed_.get() == 1);
      needbuild = true;
    }
    if( cmap.get_rep() != old_cmap.get_rep() ){
      sliceren->SetColorMap( cmap );
      old_cmap = cmap;
    }

    if( needbuild ) {
      sliceren->Build();
      sliceren->GetRange( old_min, old_max);
      min_.set(old_min);
      max_.set(old_max);
    }


    sliceren->SetInterp( bool(interp_mode.get()));

  }
 
  //AuditAllocator(default_allocator);
  if(drawX.get() || drawY.get() || drawZ.get()){
    if( control_id == -1 ){
      GeomObj *w=control_widget->GetWidget();
      control_id = ogeom->addObj( w, control_name, &control_lock);
      widget_moved(1);
    }
  } else {
    if( control_id != -1){
      ogeom->delObj( control_id, 0);
      control_id = -1;
    }
  }  

  sliceren->SetX(drawX.get());
  sliceren->SetY(drawY.get());
  sliceren->SetZ(drawZ.get());
  sliceren->SetView(drawView.get());
  //AuditAllocator(default_allocator);

  ogeom->flushViews();				  
  //AuditAllocator(default_allocator);
}

} // End namespace SCIRun


