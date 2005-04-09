
/*
 *  RescaleParticleColorMap.cc.cc:  Generate Color maps
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Module.h>
#include <Classlib/NotFinished.h>
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/ColorMap.h>
#include <Datatypes/ParticleSetPort.h>
#include <Datatypes/ParticleSetExtensionPort.h>
#include <Datatypes/ParticleSet.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

class RescaleParticleColorMap : public Module {
    ColorMapOPort* omap;
    ParticleSetIPort* iPort;
    ParticleSetExtensionIPort* iePort;
    ParticleSetOPort* oPort;
    ParticleSetExtensionOPort* oePort;
    ColorMapIPort* imap;
 public:
    RescaleParticleColorMap(const clString& id);
    RescaleParticleColorMap(const RescaleParticleColorMap&, int deep);
    virtual ~RescaleParticleColorMap();
    virtual Module* clone(int deep);
    virtual void execute();
 protected:
    TCLdouble minVal;
    TCLdouble maxVal;
    TCLstring scaleMode;
};

extern "C" {
Module* make_RescaleParticleColorMap(const clString& id)
{
    return scinew RescaleParticleColorMap(id);
}
}

RescaleParticleColorMap::RescaleParticleColorMap(const clString& id)
: Module("RescaleParticleColorMap", id, Filter),
  minVal("minVal", id, this),
  maxVal("maxVal", id, this),
  scaleMode("scaleMode", id, this)
{
    // Create the output port
    oPort = scinew ParticleSetOPort(this, "ParticleSet",
						     ParticleSetIPort::Atomic);
    add_oport(oPort);
    oePort= scinew ParticleSetExtensionOPort(this, "ParticleSetExtension",
                                           ParticleSetExtensionIPort::Atomic);
    add_oport(oePort);

    omap=scinew ColorMapOPort(this, "ColorMap", ColorMapIPort::Atomic);

    add_oport(omap);

    // Create the input ports

    iPort=scinew ParticleSetIPort(this, "ParticleSet",
						     ParticleSetIPort::Atomic);
    add_iport(iPort);
    iePort= scinew ParticleSetExtensionIPort(this, "ParticleSetExtension",
                                           ParticleSetExtensionIPort::Atomic);
    add_iport(iePort);
    scaleMode.set("auto");

    imap=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(imap);

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
    ParticleSetExtensionHandle pseh;
    if (!iPort->get(psh) || !iePort->get(pseh))
      return;

    if( scaleMode.get() == "auto") {
      Array1<double> timesteps;
      psh->list_natural_times(timesteps);
      int timestep = 0;
      if(timesteps.size() > 0){
	int sid, i;
	Array1<double>scalars;
	sid = psh->find_scalar( pseh->getScalarId());
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
      }
    } else {
      cmap.detach();
      cmap->Scale( minVal.get(), maxVal.get());
    }
    omap->send(cmap);
    oPort->send(psh);
    oePort->send(pseh);
}

