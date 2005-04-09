
/*
 *  MeshReader.cc: Mesh Reader class
 *
 *  Written by:
 *   Carole Gitlin
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>
#include <iostream.h>
#include <fstream.h>
#include <stdio.h>
#include <stdlib.h>

class PointsReader : public Module {
    MeshOPort* outport;
    TCLstring ptsname, tetname;   
    MeshHandle mesh;
    clString old_filename, old_Tname;
public:
    PointsReader(const clString& id);
    PointsReader(const PointsReader&, int deep=0);
    virtual ~PointsReader();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_PointsReader(const clString& id)
{
    return new PointsReader(id);
}

#include "PointsRegister.h"

PointsReader::PointsReader(const clString& id)
: Module("PointsReader", id, Source), ptsname("ptsname", id, this),
tetname("tetname", id, this)
{
    // Create the output data handle and port
    outport=new MeshOPort(this, "Output Data", MeshIPort::Atomic);
    add_oport(outport);

    mesh = new Mesh;
}

PointsReader::PointsReader(const PointsReader& copy, int deep)
: Module(copy, deep), ptsname("ptsname", id, this), 
tetname("tetname", id, this)
{
    NOT_FINISHED("PointsReader::PointsReader");
}

PointsReader::~PointsReader()
{
}

Module* PointsReader::clone(int deep)
{
    return new PointsReader(*this, deep);
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
    PointsReader* reader=(PointsReader*)cbdata;
    if(TCLTask::try_lock()){
	// Try the malloc lock once before we call update_progress
	// If we can't get it, then back off, since our caller might
	// have it locked
	if(!Task::test_malloc_lock()){
	    TCLTask::unlock();
	    return;
	}
	reader->update_progress(pd);
	TCLTask::unlock();
    }
}
#endif

void PointsReader::execute()
{

    clString pn(ptsname.get());
    if(!mesh.get_rep() || pn != old_filename){
	old_filename=pn;
	char *str1=pn();
	ifstream ptsfile(str1);
	if(!ptsfile){
	    error(clString("Error reading points file: ")+ptsname.get());
	    return; // Can't open file...
	}

	// Read the file...

	int n;
	ptsfile >> n;
	cerr << "nnodes=" << n << endl;
	while(ptsfile){
	    double x,y,z;
	    ptsfile >> x >> y >> z;
	    if (ptsfile)
	    {
		mesh->nodes.add(new Node(Point(x, y, z)));
	    }
	}
	cerr << "nnodes=" << mesh->nodes.size() << endl;

    }
    outport->send(mesh);
}
