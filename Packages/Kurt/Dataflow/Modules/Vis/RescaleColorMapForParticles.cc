/*
 *  RescaleColorMapForParticles.cc.cc:  Generate Color maps
 *
 *  Written by:
 *   Packages/Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   December 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include "RescaleColorMapForParticles.h"
#include <Core/Datatypes/ColorMap.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Kurt/Core/Datatypes/VisParticleSet.h>
#include <Packages/Uintah/Grid/ParticleSubset.h>

namespace Kurt {
using Uintah::ParticleSubset;

using Kurt::Datatypes::VisParticleSet;
using Kurt::Datatypes::VisParticleSetHandle;
  //using namespace Core::Datatypes;
using namespace SCIRun;


RescaleColorMapForParticles::RescaleColorMapForParticles(const clString& id)
: Module("RescaleColorMapForParticles", id, Filter),
  minVal("minVal", id, this),
  maxVal("maxVal", id, this),
  scaleMode("scaleMode", id, this)
{
    // Create the output port
    omap=new ColorMapOPort(this, "ColorMap", ColorMapIPort::Atomic);

    add_oport(omap);

    // Create the input ports

    iPort=new VisParticleSetIPort(this, "VisParticleSet",
						     VisParticleSetIPort::Atomic);
    add_iport(iPort);


    imap=new ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
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

    VisParticleSetHandle part;
    if (!iPort->get(part))
      return;

    if( scaleMode.get() == "auto") {
      ParticleSubset *ps = part->getPositions().getParticleSubset();
      double max = -1e30;
      double min = 1e30;
      for(ParticleSubset::iterator iter = ps->begin();
	  iter != ps->end(); iter++){
	max = ( part->getScalars()[ *iter ] > max ) ?
	  part->getScalars()[ *iter ] : max;
	min = ( part->getScalars()[ *iter ] < min ) ?
	  part->getScalars()[ *iter ] : min;
      }
      if( max == min ){
	max += 0.001;
	min -= 0.001;
      }
	
      cmap->Scale( min, max);
      minVal.set( min );
      maxVal.set( max );
    } else {
      cmap->Scale( minVal.get(), maxVal.get());
    }
    omap->send(cmap);
}
  
extern "C" Module* make_RescaleColorMapForParticles( const clString& id ) {
  return new RescaleColorMapForParticles( id );
}
} // End namespace Kurt


