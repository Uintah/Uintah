//static char *id="@(#) $Id$";

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

#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <CommonDatatypes/ScalarFieldPort.h>
#include <CoreDatatypes/ScalarField.h>
#include <CoreDatatypes/ScalarFieldRGchar.h>
#include <Malloc/Allocator.h>
#include <TclInterface/TCLTask.h>
#include <TclInterface/TCLvar.h>

#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

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

Module* make_ImageReader(const clString& id) {
  return new ImageReader(id);
}

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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:57:48  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:52  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
