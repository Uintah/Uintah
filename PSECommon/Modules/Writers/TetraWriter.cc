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

#include <Util/NotFinished.h>
#include <Persistent/Pstreams.h>
#include <Dataflow/Module.h>
#include <CommonDatatypes/MeshPort.h>
#include <CoreDatatypes/Mesh.h>
#include <TclInterface/TCLvar.h>
#include <fstream.h>
#include <string.h>

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class TetraWriter : public Module {
    MeshIPort* inport;
    TCLstring filename;
public:
    TetraWriter(const clString& id);
    TetraWriter(const TetraWriter&, int deep=0);
    virtual ~TetraWriter();
    virtual Module* clone(int deep);
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

TetraWriter::TetraWriter(const TetraWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
    NOT_FINISHED("TetraWriter::TetraWriter");
}

TetraWriter::~TetraWriter()
{
}

Module* TetraWriter::clone(int deep)
{
    return new TetraWriter(*this, deep);
}

#if 0
static void watcher(double pd, void* cbdata)
{
    TetraWriter* writer=(TetraWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

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
// Revision 1.1  1999/07/27 16:58:21  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:33  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:58:05  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
