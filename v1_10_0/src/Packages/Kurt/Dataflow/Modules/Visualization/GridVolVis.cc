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
 * GridVolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Packages/Kurt/Dataflow/Modules/Visualization/GridVolVis.h>
#include <Packages/Kurt/Core/Geom/VolumeRenderer.h>
#include <Packages/Kurt/Core/Geom/GridVolRen.h>
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

#include <iostream>
#ifdef __sgi
#include <ios>
#endif
#include <algorithm>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <strstream>
using std::hex;
using std::dec;
using std::ostrstream;

using SCIRun::Field;
using SCIRun::to_string;
using SCIRun::Allocator;
using SCIRun::AuditAllocator;
using SCIRun::DumpAllocator;
using SCIRun::GuiContext;

using SCIRun::default_allocator;

using namespace Kurt;

DECLARE_MAKER(GridVolVis)

GridVolVis::GridVolVis(GuiContext* ctx)
  : Module("GridVolVis", ctx,  Filter, "Visualization", "Kurt"),
    tex(0), gvr(0),
    is_fixed_(ctx->subVar("is_fixed_")),
    max_brick_dim_(ctx->subVar("max_brick_dim_")),
    min_(ctx->subVar("min_")), max_(ctx->subVar("max_")),
    num_slices(ctx->subVar("num_slices")),
    render_style(ctx->subVar("render_style")),
    alpha_scale(ctx->subVar("alpha_scale")),
    interp_mode(ctx->subVar("interp_mode")),
    volren(0)
{
}

GridVolVis::~GridVolVis()
{

}

void GridVolVis::execute(void)
{
  static FieldHandle old_tex = 0;
  static ColorMapHandle old_cmap = 0;
  static int old_brick_size = 0;
  static double old_min = 0;
  static double old_max = 1;
  static bool old_fixed = false;
  static int dumpcounter = 0;

  infield = (FieldIPort *)get_iport("Texture Field");
  incolormap = (ColorMapIPort *)get_iport("Color Map");
  ogeom = (GeometryOPort *)get_oport("Geometry");

  if (!infield) {
    gui->postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!incolormap) {
    gui->postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!ogeom) {
    gui->postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

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

  //AuditAllocator(default_allocator);
  if( !volren ){
    gvr = scinew GridVolRen();
    volren = scinew VolumeRenderer( gvr, tex, cmap,
				(is_fixed_.get() == 1),
				min_.get(), max_.get());
    if( tex->data_at() == Field::CELL ){
      volren->SetInterp(false);
      interp_mode.set(0);
    }
    std::cerr<<"Need to initialize volren\n";
    old_cmap = cmap;
    old_tex = tex;
    gui->execute(id + " SetDims " + to_string( volren->get_brick_size()));
    max_brick_dim_.set(  volren->get_brick_size() );
    old_brick_size = max_brick_dim_.get();
    if( is_fixed_.get() !=1 ){
      volren->GetRange( old_min, old_max);
      min_.set(old_min);
      max_.set(old_max);
    }
    old_fixed = ( is_fixed_.get() == 1);
    ogeom->delAll();
    ogeom->addObj( volren, "GridVolVis TransParent");
    
  } else {
    bool needbuild = false;
    if( tex.get_rep() != old_tex.get_rep() ){
      volren->SetVol( tex );
      old_tex = tex;
      needbuild = true;
    }
    if( max_brick_dim_.get() != old_brick_size ){
      volren->SetBrickSize(  max_brick_dim_.get() );
      old_brick_size = max_brick_dim_.get();
      needbuild = true;
    }
    if( is_fixed_.get() == 1 &&
	(old_min != min_.get() || old_max!= max_.get()) ){
      volren->SetRange( min_.get(), max_.get() );
      volren->FixedRange( true );
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
      volren->SetColorMap( cmap );
      old_cmap = cmap;
    }
    if( needbuild ) {
      volren->Build();
      volren->GetRange( old_min, old_max);
      min_.set(old_min);
      max_.set(old_max);
    }
    
    std::cerr<<"Initialized\n";
  }
 
  //AuditAllocator(default_allocator);
  volren->SetInterp( bool(interp_mode.get()));
  //AuditAllocator(default_allocator);

  switch( render_style.get() ) {
  case 0:
    volren->over_op();
    break;
  case 1:
    volren->mip();
    break;
  case 2:
    volren->attenuate();
  }
  
  //AuditAllocator(default_allocator);
  volren->SetNSlices( num_slices.get() );
  volren->SetSliceAlpha( alpha_scale.get() );
  //AuditAllocator(default_allocator);
  //ogeom->flushViews();	
  ogeom->flushViewsAndWait();
  //AuditAllocator(default_allocator);
  char buf[16];
  ostrstream num(buf, 16);
  num <<dumpcounter++;
  std::string dump_string("gd");
  dump_string += num.str();
  //  DumpAllocator(default_allocator, dump_string.c_str() );
}




