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

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <fstream>
using std::ifstream;
#include <stdio.h>
#include <stdlib.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

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
// Revision 1.7  2000/03/17 09:27:12  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:06:55  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:47:55  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:51  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:50  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
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
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
