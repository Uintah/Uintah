
/*
 *  VectorFieldOcean.cc: VectorField Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/VectorFieldPort.h>
#include <Datatypes/VectorFieldOcean.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>

class VectorFieldOceanReader : public Module {
    VectorFieldOPort* outport;
    TCLstring filename;
    VectorFieldHandle handle;
    clString old_filename;
    TCLint downsample;
    int old_downsample;
    int surfid;
    GeometryOPort* ogeom;
public:
    VectorFieldOceanReader(const clString& id);
    VectorFieldOceanReader(const VectorFieldOceanReader&, int deep=0);
    virtual ~VectorFieldOceanReader();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_VectorFieldOceanReader(const clString& id)
{
    return scinew VectorFieldOceanReader(id);
}
};

VectorFieldOceanReader::VectorFieldOceanReader(const clString& id)
: Module("VectorFieldOceanReader", id, Source), filename("filename", id, this),
  downsample("downsample", id, this)
{
    // Create the output data handle and port
    outport=scinew VectorFieldOPort(this, "Output Data", VectorFieldIPort::Atomic);
    add_oport(outport);
    ogeom=scinew GeometryOPort(this, "Ocean Floor", GeometryIPort::Atomic);
    add_oport(ogeom);
    old_downsample = -1;
    surfid = 0;
}

VectorFieldOceanReader::VectorFieldOceanReader(const VectorFieldOceanReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  downsample("downsample", id, this)
{
    NOT_FINISHED("VectorFieldOceanReader::VectorFieldOceanReader");
}

VectorFieldOceanReader::~VectorFieldOceanReader()
{
}

Module* VectorFieldOceanReader::clone(int deep)
{
    return scinew VectorFieldOceanReader(*this, deep);
}

void VectorFieldOceanReader::execute()
{
    clString fn(filename.get());
    if(!handle.get_rep() || fn != old_filename || downsample.get() != old_downsample){
	old_filename=fn;
	clString fn2("/u0/parker/kmt.ieee");
	VectorFieldOcean* field=new VectorFieldOcean(fn, fn2);
	if(!field->data){
	  delete field;
	  return;
	}
	handle=field;
	outport->send(handle);
	if(surfid != 0)
	  ogeom->delObj(surfid);
	surfid=ogeom->addObj(field->makesurf(downsample.get()), "Ocean Floor");
	old_downsample=downsample.get();
    } else {
      outport->send(handle);
    }
}
