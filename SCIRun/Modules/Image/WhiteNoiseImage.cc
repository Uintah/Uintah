//static char *id="@(#) $Id"

/*
 *  WhiteNoiseImage.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <SCIRun/Datatypes/Image/ImagePort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MusilRNG.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace SCIRun {
namespace Modules {

using namespace SCICore::TclInterface;

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCIRun::Datatypes;

class WhiteNoiseImage : public Module {
    ImageOPort* oport;
    TCLint xres;
    TCLint yres;
public:
    WhiteNoiseImage(const clString& id);
    virtual ~WhiteNoiseImage();
    virtual void execute();
};

Module* make_WhiteNoiseImage(const clString& id)
{
    return scinew WhiteNoiseImage(id);
}

WhiteNoiseImage::WhiteNoiseImage(const clString& id)
: Module("WhiteNoiseImage", id, Source), xres("xres", id, this),
  yres("yres", id, this)
{
    // Create the output port
    oport=scinew ImageOPort(this, "Image", ImageIPort::Atomic);
    add_oport(oport);
}

WhiteNoiseImage::~WhiteNoiseImage()
{
}

void WhiteNoiseImage::execute()
{
    MusilRNG rng;
    int xr=xres.get();
    int yr=yres.get();
    Image* image=new Image(xr, yr);
    for(int y=0;y<yr;y++){
	float* p=image->rows[y];
	for(int x=0;x<xr;x++){
	    *p++=rng()*2-1;
	    *p++=0;
	}
    }
    oport->send(image);
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.5  1999/09/08 02:27:04  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/31 08:55:36  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:49:00  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:04  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:56  mcq
// Initial commit
//
// Revision 1.2  1999/04/30 01:11:54  dav
// moved TiffReader to SCIRun from PSECore
//
// Revision 1.1  1999/04/29 22:26:36  dav
// Added image files to SCIRun
//
//
