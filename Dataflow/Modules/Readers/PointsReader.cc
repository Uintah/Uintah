
/*
 *  PointsReader.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/TclInterface/TCLTask.h>
#include <Core/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <fstream>
using std::ifstream;
#include <stdio.h>
#include <stdlib.h>

namespace SCIRun {


class PointsReader : public Module {
    MeshOPort* outport;
    TCLstring ptsname, tetname;   
    MeshHandle mesh;
    clString old_filename, old_Tname;
public:
    PointsReader(const clString& id);
    virtual ~PointsReader();
    virtual void execute();
};

extern "C" Module* make_PointsReader(const clString& id) {
  return new PointsReader(id);
}

PointsReader::PointsReader(const clString& id)
: Module("PointsReader", id, Source), ptsname("ptsname", id, this),
  tetname("tetname", id, this)
{
    // Create the output data handle and port
    outport=new MeshOPort(this, "Output Data", MeshIPort::Atomic);
    add_oport(outport);

    mesh = new Mesh;
}

PointsReader::~PointsReader()
{
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
	const char *str1=pn();
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
		mesh->nodes.add(NodeHandle(new Node(Point(x, y, z))));
	    }
	}
	cerr << "nnodes=" << mesh->nodesize() << endl;

    }
    outport->send(mesh);
}

} // End namespace SCIRun

