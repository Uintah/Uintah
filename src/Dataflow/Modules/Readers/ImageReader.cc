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

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class ImageReader : public Module {
    ScalarFieldOPort* outport;
    TCLstring filename;
    ScalarFieldHandle handle;
    clString old_filename;
public:
    ImageReader(const clString& id);
    virtual ~ImageReader();
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

ImageReader::~ImageReader()
{
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
// Revision 1.5  1999/08/25 03:47:54  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:50  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:48  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:34  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:48  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:52  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
