//static char *id="@(#) $Id$";

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

#include <SCICore/Util/NotFinished.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/MeshPort.h>
#include <SCICore/CoreDatatypes/Mesh.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream.h>
#include <fstream.h>
#include <stdio.h>
#include <stdlib.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

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

Module* make_PointsReader(const clString& id) {
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
    using SCICore::Containers::Pio;

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
	cerr << "nnodes=" << mesh->nodes.size() << endl;

    }
    outport->send(mesh);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  1999/08/17 06:37:35  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:49  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:26  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:57:53  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
