
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
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/Thread/Parallel.h>
using SCICore::Thread::Parallel;
#include <SCICore/Thread/Thread.h>
using SCICore::Thread::Thread;
#include <SCICore/Geom/GeomGroup.h>
using SCICore::GeomSpace::GeomGroup;
#include <SCICore/Geom/GeomSphere.h>
using SCICore::GeomSpace::GeomSphere;
#include <SCICore/Geom/Pt.h>
using SCICore::GeomSpace::GeomPts;
#include <SCICore/Thread/Time.h>
using SCICore::Thread::Time;

#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ColorMapPort.h>
using PSECore::Datatypes::ColorMapIPort;
using SCICore::Datatypes::ColorMapHandle;
#include <PSECore/Dataflow/Module.h>
#include <Uintah/Datatypes/Particles/PIDLObject.h>
#include <Uintah/Datatypes/Particles/PIDLObjectPort.h>
#include <Component/PIDL/URL.h>
#include <Uintah/Datatypes/Particles/Particles_sidl.h>
#include <SCICore/Math/MinMax.h>
using SCICore::Math::Min;
using SCICore::Math::Max;
#include <SCICore/Util/NotFinished.h>
#include <iostream>
using std::cerr;
#include <sstream>
#include <values.h>

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
    ColorMapIPort* cmapport;

    virtual void notifyNewTimestep(double time);
    void notifyNewParticleSet(const Visualization& newvi);
    int cbid;
    Mutex updateLock;
    void update(double time);
    void updateLoop(int);
    Mailbox<double> mailbox;
    ColorMapHandle cmap;
 public:
    ParticleViz(const clString& id);
    virtual ~ParticleViz();

    virtual void execute();

    virtual void tcl_command(TCLArgs& args, void* userdata);

    TCLdouble radius;
    TCLstring type;
    TCLint nu;
    TCLint nv;
};

ParticleViz::ParticleViz(const clString& id)
  : Module("ParticleViz", id, Filter), updateLock("ParticleViz update lock"),
    mailbox("ParticleViz update messages", 1000),
    radius("radius", id, this), type("type", id, this),
    nu("nu", id, this), nv("nv", id, this)
{
    cbid=-1;
    iface=new PIDLObjectIPort(this, "VizualizationInterface",
			      PIDLObjectIPort::Atomic);
    add_iport(iface);
    cmapport=new ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(cmapport);
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
    cmapport->get(cmap);

    if(h->getObject()){
	Visualization newvi=pidl_cast<Visualization>(h->getObject());
	if(!newvi){
	    cerr << "Wrong object type!\n";
	} else {
	    notifyNewParticleSet(newvi);
	}
    }
    TCL::execute(id+" updateCur");
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
    ostringstream str;
    if(vi){
	array1<double> timesteps=vi->listTimesteps();
	str << id << " updateMinMax " << timesteps[0] << " " << timesteps[timesteps.size()-1];
    } else {
	str << id << " updateMinMax 0 0";
    }
    vi=newvi;
    TCL::execute(str.str().c_str());
    Thread::parallel(Parallel<ParticleViz>(this, &ParticleViz::updateLoop),
		     1, false);
}

void ParticleViz::update(double time)
{
    if(!mailbox.trySend(time))
	cerr << "Dropping update: " << time << '\n';
}

#include <stdlib.h>

void ParticleViz::updateLoop(int)
{
    cerr << "updateLoop: " << getpid() << '\n';
    for(;;){
	double time=mailbox.receive();
	while(mailbox.tryReceive(time)) {}
	double t0=Time::currentSeconds();
	
	if(!vi){
	    ogeom->delAll();
	    ogeom->flushViews();
	    continue;
	}
	array1<double> timesteps=vi->listTimesteps();
	int s=0;
	while(s < timesteps.size() && time > timesteps[s])
	    s++;
	time=timesteps[s];

	double t1=Time::currentSeconds();
	array1<int> ids;
	array2<double> data;
	vi->getTimestep(time, 3, ids, data);
	double t2=Time::currentSeconds();
	reset_vars();
	ogeom->delAll();
	clString t=type.get();
	double t3=Time::currentSeconds();
	double t4;
	if(cmap.get_rep()){
	    double min=MAXDOUBLE;
	    double max=-MAXDOUBLE;
	    for(int i=0;i<data.size2();i++){
		min=Min(data[3][i], min);
		max=Max(data[3][i], max);
	    }
	    cmap->Scale(min, max);
	    t4=Time::currentSeconds();
	    if(t == "Points"){
		GeomPts* pts=new GeomPts(0);
		for(int i=0;i<data.size2();i++){
		    pts->add(Point(data[0][i], data[1][i], data[2][i]),
			     cmap->lookup(data[3][i])->diffuse);
		}
		ogeom->addObj(pts, "Particles");
	    } else {
		int nnu=nu.get();
		int nnv=nv.get();
		double r=radius.get();
		GeomGroup* group=new GeomGroup();
		for(int i=0;i<data.size2();i++){
		    group->add(new GeomSphere(Point(data[0][i], data[1][i], data[2][i]), r, nnu, nnv));
		}
		ogeom->addObj(group, "Particles");
	    }
	} else {
	    if(t == "Points"){
		GeomPts* pts=new GeomPts(0);
		for(int i=0;i<data.size2();i++){
		    pts->add(Point(data[0][i], data[1][i], data[2][i]));
		}
		ogeom->addObj(pts, "Particles");
	    } else {
		int nnu=nu.get();
		int nnv=nv.get();
		double r=radius.get();
		GeomGroup* group=new GeomGroup();
		for(int i=0;i<data.size2();i++){
		    group->add(new GeomSphere(Point(data[0][i], data[1][i], data[2][i]), r, nnu, nnv));
		}
		ogeom->addObj(group, "Particles");
	    }
	}
	double t5=Time::currentSeconds();
	ogeom->flushViews();
	double t6=Time::currentSeconds();
	cerr << t1-t0 << ' ' << t2-t1 << ' ' << t3-t2 << ' ' << t4-t3 << ' ' << ' ' << t5-t4 << ' ' << t6-t5 << '\n';
    }
}

void ParticleViz::notifyNewTimestep(double time)
{
    ostringstream str;
    str << id << " updateMax " << time;
    TCL::execute(str.str().c_str());
}

void ParticleViz::tcl_command(TCLArgs& args, void* userdata)
{
    if(args[1] == "getMinMax"){
	if(args.count() != 2){
	    args.error("getMinMax requires no arguments");
	    return;
	}
	if(vi){
	    array1<double> timesteps=vi->listTimesteps();
	    if(timesteps.size() > 0){
		clString str(to_string(timesteps[0])+" "+
			     to_string(timesteps[timesteps.size()-1]));
		args.result(str);
	    } else {
		args.result("0 0");
	    }
	} else {
	    args.result("0 0");
	}
    } else if(args[1] == "update"){
	if(args.count() != 3){
	    args.error("update requires one argument: time");
	    return;
	}
	double time;
	if(!args[2].get_double(time)){
	    args.error("update cannot parse time");
	}
	update(time);
    } else {
	Module::tcl_command(args, userdata);
    }
}

Module* make_ParticleViz( const clString& id )
{
  return new ParticleViz( id );
}


} // End namespace Modules
} // End namespace Uintah


//
// $Log$
// Revision 1.2  1999/10/15 20:23:01  sparker
// Mostly working
//
// Revision 1.1  1999/10/07 02:08:28  sparker
// use standard iostreams and complex type
//
//
