
/*
 *  ScalarFieldReader.cc: ScalarField Reader class
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
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRGchar.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>

#include <stdio.h>

class ImageReader : public Module {
    ScalarFieldOPort* outport;
    TCLstring filename;
    ScalarFieldHandle handle;
    clString old_filename;
public:
    ImageReader(const clString& id);
    ImageReader(const ImageReader&, int deep=0);
    virtual ~ImageReader();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_ImageReader(const clString& id)
{
    return scinew ImageReader(id);
}
};

ImageReader::ImageReader(const clString& id)
: Module("ImageReader", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew ScalarFieldOPort(this, "Output Data", ScalarFieldIPort::Atomic);
    add_oport(outport);
}

ImageReader::ImageReader(const ImageReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
    NOT_FINISHED("ImageReader::ImageReader");
}

ImageReader::~ImageReader()
{
}

Module* ImageReader::clone(int deep)
{
    return scinew ImageReader(*this, deep);
}

void ImageReader::execute()
{
    clString fn(filename.get());
    if(!handle.get_rep() || fn != old_filename){
	old_filename=fn;

	FILE *f = fopen(fn(),"rb");

	if (!f) {
	  error("No such file");
	  return;
	}
	int dims[2];

	if (!fread(dims,sizeof(int),2,f)) {
	  error ("file is hosed...");
	  return;
	}

	if (dims[0] < 0 || dims[1] < 0 ||
	    dims[0] > 4096 || dims[1] > 4096) {
	  error("Dimensions are hosed...");
	  return;
	}

	int xdim=dims[0],ydim=dims[1],zdim=1;
	
	ScalarFieldRGchar* sf=new ScalarFieldRGchar;
	sf->resize(xdim, ydim, zdim);
	sf->compute_bounds();
	Point pmin(0,0,0),pmax(xdim,ydim,zdim);
	sf->set_bounds(pmin,pmax);  // something more reasonable later...
	
	for (int y=0;y<ydim;y++) {
	  for(int x=0;x<xdim;x++) {
	    char newval[3];
	    if (!fread(newval,sizeof(char),3,f)) {
	      printf("Choked...\n");
	    }
	    sf->grid(x,y,0) = newval[0];
	  }
	}

	handle = sf;
    }
    outport->send(handle);
}

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template void Pio(Piostream&, ScalarFieldHandle&);

#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/LockingHandle.cc>

static void _dummy_(Piostream& p1, ScalarFieldHandle& p2)
{
    Pio(p1, p2);
}

#endif
#endif

