
/*
 *  PartToGeom.cc:  Convert a Partace into geoemtry
 *
 *  Written 
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Pt.h>
#include <Geom/Sphere.h>
#include <Datatypes/ParticleSetPort.h>
#include <Datatypes/ParticleSet.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

class PartToGeom : public Module {
    ParticleSetIPort* iPart;
    GeometryOPort* ogeom;
    TCLdouble current_time;
    int last_idx;
    int last_generation;
public:
    PartToGeom(const clString& id);
    PartToGeom(const PartToGeom&, int deep);
    virtual ~PartToGeom();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_PartToGeom(const clString& id)
{
    return scinew PartToGeom(id);
}
};

PartToGeom::PartToGeom(const clString& id)
: Module("PartToGeom", id, Filter), current_time("current_time", id, this)
{
    // Create the input port
    iPart=scinew ParticleSetIPort(this, "Particles", ParticleSetIPort::Atomic);
    add_iport(iPart);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    last_idx=-1;
    last_generation=-1;
}

PartToGeom::PartToGeom(const PartToGeom&copy, int deep)
: Module(copy, deep), current_time("current_time", id, this)
{
    NOT_FINISHED("PartToGeom::PartToGeom");
}

PartToGeom::~PartToGeom()
{
}

Module* PartToGeom::clone(int deep)
{
    return scinew PartToGeom(*this, deep);
}

void PartToGeom::execute()
{
    ParticleSetHandle part;
    
    if (!iPart->get(part)){
	last_idx=-1;
	return;
    }

    double time=current_time.get();
    Array1<double> timesteps;
    part->list_natural_times(timesteps);
    if(timesteps.size()==0){
	ogeom->delAll();
	last_idx=-1;
	return;
    }
    int timestep=0;
    while(time>timesteps[timestep] && timestep<timesteps.size()-1)
	timestep++;

    if(timestep == last_idx && part->generation == last_generation)
	return;
    last_idx=timestep;
    last_generation=part->generation;

    int posid=part->position_vector();
    Array1<Vector> pos;
    part->get(timestep, posid, pos);

    
    //double radius=1;
    //GeomGroup* group = scinew GeomGroup;
    //    for (int i=0; i<pos.size();i++){
    //	group->add(scinew GeomSphere(pos[i].asPoint(), radius, 8, 6));
    //}
    GeomPts* pts=scinew GeomPts(pos.size());
    for(int i=0;i<pos.size();i++){
	pts->add(pos[i].asPoint());
    }
    GeomMaterial* matl=scinew GeomMaterial(pts,
					   scinew Material(Color(0,0,0),
							   Color(0,.6,0), 
							   Color(.5,.5,.5), 20));
    ogeom->delAll();
    ogeom->addObj(matl, "Particles");
}
