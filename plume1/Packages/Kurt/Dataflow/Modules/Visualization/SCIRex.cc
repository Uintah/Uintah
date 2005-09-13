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
 * SCIRex.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Packages/Kurt/Dataflow/Modules/Visualization/SCIRex.h>
#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/Field.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/GuiContext.h>

#include <vector>
#include <iostream>
#include <algorithm>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <sstream>
#include <string>

using std::cerr;
using std::endl;
using std::hex;
using std::dec;
using std::ostringstream;
using std::istringstream;
using std::vector;
using std::string;

using SCIRun::Field;
using SCIRun::to_string;
using SCIRun::Allocator;
using SCIRun::AuditAllocator;
using SCIRun::DumpAllocator;
using SCIRun::GuiContext;

using SCIRun::default_allocator;

using namespace Kurt;

DECLARE_MAKER(SCIRex)

SCIRex::SCIRex(GuiContext* ctx)
  : Module("SCIRex", ctx,  Filter, "Visualization", "Kurt"),
    tex_(0),
    volren_(0),
    texH_(0),
    is_fixed_(ctx->subVar("is_fixed_")),
    max_brick_dim_(ctx->subVar("max_brick_dim_")),
    min_(ctx->subVar("min_")), max_(ctx->subVar("max_")),
    draw_mode_(ctx->subVar("draw_mode")),
    num_slices_(ctx->subVar("num_slices")),
    render_style_(ctx->subVar("render_style")),
    alpha_scale_(ctx->subVar("alpha_scale")),
    interp_mode_(ctx->subVar("interp_mode")),
    dump_frames_(ctx->subVar("dump_frames")),
    use_depth_(ctx->subVar("use_depth")),
    displays_(ctx->subVar("displays")),
    compositers_(ctx->subVar("compositers"))
{
}

SCIRex::~SCIRex()
{

}

void SCIRex::execute(void)
{
  static FieldHandle old_tex = 0;
  static ColorMapHandle old_cmap = 0;
  static int old_brick_size = 0;
  static double old_min = 0;
  static double old_max = 1;
  static bool old_fixed = false;
  static GeomID geom_id = -1;
  static int old_compositers = 0;
//   static int dumpcounter = 0;

  infield_ = (FieldIPort *)get_iport("Texture Field");
  incolormap_ = (ColorMapIPort *)get_iport("Color Map");
  ogeom_ = (GeometryOPort *)get_oport("Geometry");

  if (!infield_) {
    gui->postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!incolormap_) {
    gui->postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!ogeom_) {
    gui->postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  if (!infield_->get(tex_)) {
    warning("Did not get FieldHandle!");
    return;
  }
  else if (!tex_.get_rep()) {
    warning("FieldHandle does not have data.");
    return;
  }
  
  ColorMapHandle cmap;
  if( !incolormap_->get(cmap)){
    return;
  }

  if(tex_->get_type_name(0) != "LatVolField"){
    warning("SCIRex only works with LatVolFields, no action.");
    return;
  }

  istringstream iss(displays_.get());
  string dpy;
  vector<char*> dpys; // hard code for now
  while(iss >> dpy){
    char *ds = new char[dpy.length()];
//     cerr<<"using display "<<dpy<<endl;
    strcpy(ds, dpy.c_str());
    dpys.push_back( ds );
  }

  if( dpys.size() < 2 ) return;

//   dpys.push_back(":0.0");
//   dpys.push_back(":0.0");
//   int ncomp = 3;
  //AuditAllocator(default_allocator);
//   if(geom_id != -1)
//     ogeom_->delObj(geom_id, 0);
  
  if( !volren_ ){
    cerr<<"min = "<<min_.get()<<" max = "<<max_.get()<<endl;
    volren_ =
      scinew SCIRexRenderer(dpys, compositers_.get(), tex_, cmap,
			    (is_fixed_.get() == 1),
			    min_.get(), max_.get());
    //   if( tex_->data_at() == Field::CELL ){
    //     volren_->SetInterp(false);
    //     interp_mode_.set(0);
    //   }
//     std::cerr<<"Need to initialize volren\n";
    old_cmap = cmap;
    old_tex = tex_;
    gui->execute(id + " SetDims " + to_string( volren_->getBrickSize()));
    max_brick_dim_.set(  volren_->getBrickSize() );
    old_brick_size = max_brick_dim_.get();
    if( is_fixed_.get() !=1 ){
      volren_->getRange( old_min, old_max);
      min_.set(old_min);
      max_.set(old_max);
    }
    old_fixed = ( is_fixed_.get() == 1);
    old_compositers = compositers_.get();
    ogeom_->delAll();
    geom_id = ogeom_->addObj( volren_, "SCIRex TransParent");
  
  } else {
    bool needbuild = false;
    if( tex_.get_rep() != old_tex.get_rep() ){
      volren_->SetVol( tex_ );
      old_tex = tex_;
      needbuild = true;
    }
    if( old_compositers != compositers_.get()) {
      volren_->UpdateCompositers( compositers_.get() );
      old_compositers = compositers_.get();
    }
    if( max_brick_dim_.get() != old_brick_size ){
      volren_->SetBrickSize(  max_brick_dim_.get() );
      old_brick_size = max_brick_dim_.get();
      needbuild = true;
    }
    if( is_fixed_.get() == 1 &&
	(old_min != min_.get() || old_max!= max_.get()) ){
      volren_->SetRange( min_.get(), max_.get() );
      volren_->RangeIsFixed( true );
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
      volren_->SetColorMap( cmap );
      old_cmap = cmap;
    }
    if( needbuild ) {
      volren_->Build(); // dpys, ncomp, tex_, cmap);
      volren_->getRange( old_min, old_max);
      min_.set(old_min);
      max_.set(old_max);
    }
    
//    std::cerr<<"Initialized\n";
  }
 
  //AuditAllocator(default_allocator);
  volren_->SetInterp( bool(interp_mode_.get()));
  //AuditAllocator(default_allocator);
  volren_->DumpFrames( bool(dump_frames_.get()));
  volren_->UseDepth( bool(use_depth_.get()));

  switch( render_style_.get() ) {
  case 0:
    volren_->over_op();
    break;
  case 1:
    volren_->mip();
    break;
  case 2:
    volren_->attenuate();
  }
  
  //AuditAllocator(default_allocator);
  volren_->SetNSlices( num_slices_.get() );
  volren_->SetSliceAlpha( alpha_scale_.get() );
  //AuditAllocator(default_allocator);
  //ogeom->flushViews();	
  ogeom_->flushViewsAndWait();
  //AuditAllocator(default_allocator);
//   ostringstream num;
//   num <<dumpcounter++;
//   std::string dump_string("SCIRex dump_string");
//   dump_string += num.str();
  //  DumpAllocator(default_allocator, dump_string.c_str() );

  gui->execute(id + " disableDpy");
}

