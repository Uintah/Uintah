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
#include <SCICore/Datatypes/ColorMap.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Grid/ParticleSubset.h>
#include <Uintah/Datatypes/ScalarParticles.h>

namespace Uintah {
namespace Modules {

using Uintah::ParticleSubset;

using SCICore::Datatypes::ColorMap;
using SCICore::Datatypes::ColorMapHandle;
using Uintah::Datatypes::ScalarParticles;
using Uintah::Datatypes::ScalarParticlesHandle;
using PSECore::Datatypes::ScalarParticlesIPort;
using PSECore::Datatypes::ColorMapOPort;
using PSECore::Datatypes::ColorMapIPort;
using PSECore::Dataflow::Module;

using namespace SCICore::TclInterface;
using SCICore::Containers::clString;

RescaleColorMapForParticles::RescaleColorMapForParticles(const clString& id)
: Module("RescaleColorMapForParticles", id, Filter),
  minVal("minVal", id, this),
  maxVal("maxVal", id, this),
  scaleMode("scaleMode", id, this)
{
    // Create the output port
    omap=scinew ColorMapOPort(this, "ColorMap", ColorMapIPort::Atomic);

    add_oport(omap);

    // Create the input ports

    iPort=scinew ScalarParticlesIPort(this, "ScalarParticles",
						     ScalarParticlesIPort::Atomic);
    add_iport(iPort);


    imap=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(imap);

    //    scaleMode.set("auto");

}

RescaleColorMapForParticles::~RescaleColorMapForParticles()
{
}

void RescaleColorMapForParticles::execute()
{
    ColorMapHandle cmap;
    if(!imap->get(cmap))
	return;

    ScalarParticlesHandle part;
    if (!iPort->get(part))
      return;

    double mx = -1e30;
    double mn = 1e30;
 
   if( scaleMode.get() == "auto") {
      ParticleSubset *ps = part->getPositions().getParticleSubset();
      part->get_minmax(mn, mx);


      if( mx == mn ){
	mx += 0.001;
	mn -= 0.001;
      }
      
      cmap->Scale( mn, mx);
      minVal.set( mn );
      maxVal.set( mx );
   } else {
     cmap->Scale( minVal.get(), maxVal.get());
   }
   omap->send(cmap);
}
  
extern "C" Module* make_RescaleColorMapForParticles( const clString& id ) {
  return scinew RescaleColorMapForParticles( id );
}

} // End namespace Modules
} // End namespace Kurt

