//static char *id="@(#) $Id$";

/*
 *  TetraWriter.cc: Mesh Writer class
 *
 *  Written by:
 *   Carole Gitlin
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <SCICore/Persistent/Pstreams.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <fstream>
using std::endl;
using std::ofstream;
#include <string.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class TetraWriter : public Module {
    MeshIPort* inport;
    TCLstring filename;
public:
    TetraWriter(const clString& id);
    virtual ~TetraWriter();
    virtual void execute();
};

Module* make_TetraWriter(const clString& id) {
  return new TetraWriter(id);
}

TetraWriter::TetraWriter(const clString& id)
: Module("TetraWriter", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    inport=new MeshIPort(this, "Input Data", MeshIPort::Atomic);
    add_iport(inport);
}

TetraWriter::~TetraWriter()
{
}

void TetraWriter::execute()
{
    MeshHandle handle;
    if(!inport->get(handle))
	return;
    clString fn(filename.get());
    if(fn == "")
	return;

    const char *str1 = fn();
    char str2[80];

    strcpy(str2, str1);
    strcat(str2, ".pts");
    ofstream outfile(str2);

    outfile << handle -> nodes.size() << endl;
    int i;
    for (i = 0; i < handle -> nodes.size(); i++)
    {
	outfile << handle -> nodes[i] -> p.x() << " " <<
	           handle -> nodes[i] -> p.y() << " " <<
	           handle -> nodes[i] -> p.z() << endl;
    }

    strcpy(str2, str1);
    strcat(str2,".tetra");
    ofstream tetfile(str2);

    tetfile << handle ->elems.size() << endl;

    for (i = 0; i < handle -> elems.size(); i++)
    {
	for (int j = 0; j < 4; j++)
	    tetfile << handle -> elems[i] -> n[j] << " ";
	tetfile << endl;
    }

}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  1999/10/07 02:07:13  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:48:16  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:18:02  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:16  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:57  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:21  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:33  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:58:05  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
