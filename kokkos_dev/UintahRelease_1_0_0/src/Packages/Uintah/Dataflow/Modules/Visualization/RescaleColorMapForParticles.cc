/*
 *  RescaleColorMapForParticles.cc.cc:  Generate Color maps
 *
 *  Written by:
 *   Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   December 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include "RescaleColorMapForParticles.h"
#include <Core/Geom/ColorMap.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSubset.h>
#include <Packages/Uintah/Core/Datatypes/ScalarParticles.h>

namespace Uintah {

using namespace SCIRun;

RescaleColorMapForParticles::RescaleColorMapForParticles(GuiContext* ctx)
: Module("RescaleColorMapForParticles", ctx, Filter, "Visualization", "Uintah"),
  minVal(get_ctx()->subVar("minVal")),
  maxVal(get_ctx()->subVar("maxVal")),
  scaleMode(get_ctx()->subVar("scaleMode"))
{
    //    scaleMode.set("auto");

}

RescaleColorMapForParticles::~RescaleColorMapForParticles()
{
}

void RescaleColorMapForParticles::execute()
{
    // Create the input ports
  imap= (ColorMapIPort *) get_iport("ColorMap");
  iPort=  (ScalarParticlesIPort *) get_iport("ScalarParticles");
    // Create the output port
  omap= (ColorMapOPort *) get_oport("ColorMap");

    ColorMapHandle cmap;
    ColorMapHandle out_cmap;
    if(!imap->get(cmap))
	return;

    ScalarParticlesHandle part;
    if (!iPort->get(part))
      return;
    
    if( part.get_rep() == 0) return;

    double mx = -1e30;
    double mn = 1e30;
 
   out_cmap = new ColorMap(*cmap.get_rep());
   if( scaleMode.get() == "auto") {

      part->get_minmax(mn, mx);


      if( mx == mn ){
	mx += 0.001;
	mn -= 0.001;
      }
      
      out_cmap->Scale( mn, mx);
      minVal.set( mn );
      maxVal.set( mx );
   } else {
     out_cmap->Scale( minVal.get(), maxVal.get());
   }
   omap->send_and_dereference(out_cmap);
}
  
DECLARE_MAKER(RescaleColorMapForParticles)
} // End namespace Uintah
