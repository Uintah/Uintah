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


#include <SCICore/Datatypes/ColorMap.h>
#include <SCICore/Malloc/Allocator.h>

#include "RescaleParticleColorMap.h"
#include <Uintah/Datatypes/Particles/MPVizParticleSet.h>

namespace Uintah {
namespace Modules {


using namespace Uintah::Datatypes;

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::Containers;

RescaleParticleColorMap::RescaleParticleColorMap(const clString& id)
: Module("RescaleParticleColorMap", id, Filter),
  minVal("minVal", id, this),
  maxVal("maxVal", id, this),
  scaleMode("scaleMode", id, this)
{
    // Create the output port
    omap=new ColorMapOPort(this, "ColorMap", ColorMapIPort::Atomic);

    add_oport(omap);

    // Create the input ports

    iPort=new ParticleSetIPort(this, "ParticleSet",
						     ParticleSetIPort::Atomic);
    add_iport(iPort);


    imap=new ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(imap);

    //    scaleMode.set("auto");

}

RescaleParticleColorMap::~RescaleParticleColorMap()
{
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
  
extern "C" Module* make_RescaleParticleColorMap( const clString& id ) {
  return new RescaleParticleColorMap( id );
}

} // End namespace Modules
} // End namespace Uintah

//
// $Log$
// Revision 1.7  2000/03/17 09:30:11  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/09/21 16:12:25  kuzimmer
// changes made to support binary/ASCII file IO
//
// Revision 1.5  1999/08/25 03:49:04  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:18:08  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:23  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:40:11  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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
