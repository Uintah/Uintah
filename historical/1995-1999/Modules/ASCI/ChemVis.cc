
/*
 *  ChemVis.cc:  Convert a Partace into geoemtry
 *
 *  Written 
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Tester/RigorousTest.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Pt.h>
#include <Geom/Sphere.h>
#include <Geom/TCLGeom.h>
#include <Datatypes/ParticleSetPort.h>
#include <Datatypes/ParticleSet.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include <fstream.h>

struct Prop {
    TCLdouble radius;	
    TCLColor color;
    TCLstring label;
    Prop(const clString& name, const clString& id, TCL* tcl);
};

Prop::Prop(const clString& name, const clString& id, TCL* tcl)
    : radius("radius", id+"-"+name, tcl),
      color("color", id+"-"+name, tcl),
      label("label", id+"-"+name, tcl)
{
}

struct ParticleGroup {
    int groupno;
    Array1<Point> particles;
};

struct TimeStep {
    Array1<ParticleGroup*> pgroups;
};

class ChemVis : public Module {
    GeometryOPort* ogeom;
    TCLstring filename;
    TCLint current_time;
    clString last_file;
    TCLint sphere_res;

    HashTable<int, Prop*> props;
    Array1<TimeStep*> data;
public:
    ChemVis(const clString& id);
    ChemVis(const ChemVis&, int deep);
    virtual ~ChemVis();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_ChemVis(const clString& id)
{
    return scinew ChemVis(id);
}
}

ChemVis::ChemVis(const clString& id)
: Module("ChemVis", id, Filter), current_time("current_time", id, this),
  filename("filename", id, this), sphere_res("sphere_res", id, this)
{
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

ChemVis::ChemVis(const ChemVis&copy, int deep)
: Module(copy, deep), current_time("current_time", id, this),
  filename("filename", id, this), sphere_res("sphere_res", id, this)
{
    NOT_FINISHED("ChemVis::ChemVis");
}

ChemVis::~ChemVis()
{
}

Module* ChemVis::clone(int deep)
{
    return scinew ChemVis(*this, deep);
}

void ChemVis::execute()
{

    clString file=filename.get();
    if(file == "")
	return;

    if(file != last_file){
	ifstream in(file());
	if(!in){
	    cerr << "Error opening file: " << file << '\n';
	    last_file="";
	    return;
	}
	int nsteps;
	in >> nsteps;
	char buf[1000];
	in.getline(buf, 1000); // Rest of line 1
	in.getline(buf, 1000); // Line 2
	in.getline(buf, 1000); // Line 3
	int nparticles;
	in >> nparticles;
	in.getline(buf, 1000); // Rest of line 4
	in.getline(buf, 1000); // Line 5
	in.getline(buf, 1000); // Line 6
	cerr << "nsteps: " << nsteps << '\n';
	cerr << "nparticles: " << nparticles << '\n';

	for(int i=0;i<data.size();i++)
	    delete data[i];

	data.remove_all();
	for(i=0;i<nsteps;i++){
	    TimeStep* ts=new TimeStep;
	    HashTable<int, ParticleGroup*> pgroups;
	    for(int j=0;j<nparticles;j++){
		double x,y,z;
		int pg;
		int junk;
		in >> x >> y >> z >> pg >> junk;
		if(!in){
		    cerr << "Error reading file!\n";
		    return;
		}
		ParticleGroup* pgroup;
		if(!pgroups.lookup(pg, pgroup)){
		    // New group...
		    pgroup=new ParticleGroup;
		    pgroup->groupno=pg;
		    pgroups.insert(pg, pgroup);
		    ts->pgroups.add(pgroup);
		    Prop* prop;
		    if(!props.lookup(pg, prop)){
			prop=new Prop(to_string(pg), id, this);
			props.insert(pg, prop);
		    }
		}
		pgroup->particles.add(Point(x,y,z));
	    }
	    data.add(ts);
	}
	last_file=file;
	current_time.set(0);
    }

    ogeom->delAll();
    int time=current_time.get();
    if(time<0)
	time=0;
    if(time>=data.size())
	time=data.size()-1;
    TimeStep* ts=data[time];
    int sr=sphere_res.get();
    for(int j=0;j<ts->pgroups.size();j++){
	ParticleGroup* pgroup=ts->pgroups[j];
	int groupno=pgroup->groupno;
	Prop* prop;
	if(!props.lookup(groupno, prop)){
	    cerr << "Cannot find property for group number " << groupno << '\n';
	    continue;
	}
	double radius=prop->radius.get();
	GeomGroup* group=scinew GeomGroup;
	double Ka=0.2;
	double Kd=0.8;
	Color c=prop->color.get();
	Color specular(.6,.6,.6);
	double specpow=50;
	MaterialHandle m=scinew Material(c*Ka, c*Kd, specular, specpow);
	GeomMaterial* matl=scinew GeomMaterial(group, m);

	for(int i=0;i<pgroup->particles.size();i++){
	    Point& p(pgroup->particles[i]);
	    group->add(scinew GeomSphere(p, radius, sr, sr));
	}
	ogeom->addObj(matl, prop->label.get());
    }
}
