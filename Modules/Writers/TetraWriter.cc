
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

#include <Classlib/NotFinished.h>
#include <Classlib/Pstreams.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <TCL/TCLvar.h>
#include <fstream.h>
#include <string.h>

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

extern "C" {
Module* make_TetraWriter(const clString& id)
{
    return new TetraWriter(id);
}
};

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

static void watcher(double pd, void* cbdata)
{
    TetraWriter* writer=(TetraWriter*)cbdata;
    writer->update_progress(pd);
}

void TetraWriter::execute()
{
    MeshHandle handle;
    if(!inport->get(handle))
	return;
    clString fn(filename.get());
    if(fn == "")
	return;

    char *str1 = fn();
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
