
/*
 *  ParticleViz: Run a particle simulation
 *  $Id$
 *
 *  Written 
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Geom/GeomGroup.h>
using SCICore::GeomSpace::GeomGroup;
#include <SCICore/Geom/GeomSphere.h>
using SCICore::GeomSpace::GeomSphere;
#include <SCICore/Geom/Pt.h>
using SCICore::GeomSpace::GeomPts;

#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Dataflow/Module.h>
#include <Uintah/Datatypes/Particles/PIDLObject.h>
#include <Uintah/Datatypes/Particles/PIDLObjectPort.h>
#include <Component/PIDL/URL.h>
#include <Uintah/Datatypes/Particles/Particles_sidl.h>
#include <SCICore/Util/NotFinished.h>
#include <iostream>
using std::cerr;
#include <sstream>

namespace Uintah {
namespace Modules {

using namespace PSECore::Dataflow;

using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;
using namespace PSECore::Datatypes;
using Uintah::Datatypes::PIDLObject;
using Uintah::Datatypes::PIDLObjectHandle;
using Uintah::Datatypes::PIDLObjectIPort;
using Uintah::Datatypes::Particles::Visualization;
using Uintah::Datatypes::Particles::TimestepNotify_interface;
using std::ostringstream;
using std::vector;
using CIA::array1;
using CIA::array2;

class ParticleViz : public Module, public TimestepNotify_interface {
    PIDLObjectIPort* iface;
    Visualization vi;
    GeometryOPort* ogeom;

    virtual void notifyNewTimestep(double time);
    void notifyNewParticleSet(const Visualization& newvi);
    int cbid;
 public:
    ParticleViz(const clString& id);
    virtual ~ParticleViz();

    virtual void execute();
};

ParticleViz::ParticleViz(const clString& id)
  : Module("ParticleViz", id, Filter)
{
    cbid=-1;
    iface=new PIDLObjectIPort(this, "VizualizationInterface",
			      PIDLObjectIPort::Atomic);
    add_iport(iface);
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    addReference(); // So that we won't get blown away through our PIDL interface
}

ParticleViz::~ParticleViz()
{
}

void ParticleViz::execute()
{
    PIDLObjectHandle h;
    if(!iface->get(h))
	return;

    if(h->getObject()){
	Visualization newvi=pidl_cast<Visualization>(h->getObject());
	if(!newvi){
	    cerr << "Wrong object type!\n";
	} else {
	    notifyNewParticleSet(newvi);
	}
    }
}

void ParticleViz::notifyNewParticleSet(const Visualization& newvi)
{
    if(vi.getPointer() != newvi.getPointer()){
	if(vi)
	    vi->unregisterNotify(cbid);
	if(newvi)
	    cbid=newvi->registerNotify(this);
	else
	    cbid=-1;
    }
    vi=newvi;
}

void ParticleViz::notifyNewTimestep(double time)
{
    array1<int> ids;
    array2<double> data;
    vi->getTimestep(time, 3, ids, data);
    //GeomGroup* group=new GeomGroup();
    GeomPts* pts=new GeomPts(0);
    for(int i=0;i<ids.size();i++){
	//group->add(new GeomSphere(Point(data[0][i], data[1][i], data[2][i]), 3.0, 3, 3));
	pts->add(Point(data[0][i], data[1][i], data[2][i]));
    }
    ogeom->delAll();
    //ogeom->addObj(group, "Particles");
    ogeom->addObj(pts, "Particles");
    ogeom->flushViews();
}

Module* make_ParticleViz( const clString& id )
{
  return new ParticleViz( id );
}


} // End namespace Modules
} // End namespace Uintah


//
// $Log$
// Revision 1.1  1999/10/07 02:08:28  sparker
// use standard iostreams and complex type
//
//
