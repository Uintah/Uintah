//static char *id="@(#) $Id$";

/*
 *  RescaleParticleColorMap.cc.cc:  Generate Color maps
 *
 *  Written by:
 *   Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   December 1999
 *
 *  Copyright (C) 1999 SCI Group
 */


#include <Datatypes/Particles/MPVizParticleSet.h>
#include <Util/NotFinished.h>
#include <CoreDatatypes/ColorMap.h>
#include <Malloc/Allocator.h>
#include <RescaleParticleColorMap.h>

namespace Uintah {
namespace Modules {


using namespace Uintah::Datatypes;

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::Containers;

RescaleParticleColorMap::RescaleParticleColorMap(const clString& id)
: Module("RescaleParticleColorMap", id, Filter),
  minVal("minVal", id, this),
  maxVal("maxVal", id, this),
  scaleMode("scaleMode", id, this)
{
    // Create the output port
    omap=scinew ColorMapOPort(this, "ColorMap", ColorMapIPort::Atomic);

    add_oport(omap);

    // Create the input ports

    iPort=scinew ParticleSetIPort(this, "ParticleSet",
						     ParticleSetIPort::Atomic);
    add_iport(iPort);


    imap=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(imap);

    //    scaleMode.set("auto");

}

RescaleParticleColorMap::RescaleParticleColorMap(const RescaleParticleColorMap& copy, int deep)
: Module(copy, deep),
  minVal("minVal", id, this),
  maxVal("maxVal", id, this),
  scaleMode("scaleMode", id, this)
{
    NOT_FINISHED("RescaleParticleColorMap::RescaleParticleColorMap");
}

RescaleParticleColorMap::~RescaleParticleColorMap()
{
}

Module* RescaleParticleColorMap::clone(int deep)
{
    return scinew RescaleParticleColorMap(*this, deep);
}

void RescaleParticleColorMap::execute()
{
    ColorMapHandle cmap;
    if(!imap->get(cmap))
	return;

    ParticleSetHandle psh;
    if (!iPort->get(psh))
      return;

    if( scaleMode.get() == "auto") {
      Array1<double> timesteps;
      psh->list_natural_times(timesteps);
      int timestep = 0;
      if(timesteps.size() > 0){
	int sid, i;
	Array1<double>scalars;
	ParticleSet *ps = psh.get_rep();
	if( MPVizParticleSet *mpvps = dynamic_cast <MPVizParticleSet *> (ps)){
	  sid = psh->find_scalar( mpvps->getScalarId());
	  psh->get(timestep, sid, scalars);
	  double max = -1e30;
	  double min = 1e30;
	  for( i = 0; i < scalars.size(); i++ ) {
	    max = ( scalars[i] > max ) ? scalars[i] : max;
	    min = ( scalars[i] < min ) ? scalars[i] : min;
	  }
	  if( max == min ){
	    max += 1e-6;
	    min -= 1e-6;
	  }
	
	  cmap.detach();
	  cmap->Scale( min, max);
	  minVal.set( min );
	  maxVal.set( max );
	} else {
	  return;
	}
      } else {
	return;
      }
    } else {
      cmap.detach();
      cmap->Scale( minVal.get(), maxVal.get());
    }
    omap->send(cmap);
 }
  
Module* make_RescaleParticleColorMap( const clString& id ) {
  return new RescaleParticleColorMap( id );
}

} // End namespace Modules
} // End namespace Uintah

//
// $Log$
// Revision 1.1  1999/07/27 17:08:57  mcq
// Initial commit
//
// Revision 1.3  1999/06/09 23:23:44  kuzimmer
// Modified the modules to work with the new Material/Particle classes.  When a module needs to determine the type of particleSet that is incoming, the new stl dynamic type testing is used.  Works good so far.
//
// Revision 1.2  1999/04/27 23:18:41  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//
