
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

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/TclInterface/TCLvar.h>
#include <fstream>
using std::endl;
using std::ofstream;
#include <string.h>

namespace SCIRun {


class TetraWriter : public Module {
    MeshIPort* inport;
    TCLstring filename;
public:
    TetraWriter(const clString& id);
    virtual ~TetraWriter();
    virtual void execute();
};

extern "C" Module* make_TetraWriter(const clString& id) {
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

} // End namespace SCIRun

