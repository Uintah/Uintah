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

#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/Image/ImagePort.h>
#include <Malloc/Allocator.h>
#include <Math/MusilRNG.h>
#include <TclInterface/TCLvar.h>

namespace SCIRun {
namespace Modules {

using namespace SCICore::TclInterface;
using namespace SCICore::Multitask;

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;

using namespace SCIRun::Datatypes;

class WhiteNoiseImage : public Module {
    ImageOPort* oport;
    TCLint xres;
    TCLint yres;
public:
    WhiteNoiseImage(const clString& id);
    WhiteNoiseImage(const WhiteNoiseImage&, int deep);
    virtual ~WhiteNoiseImage();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_WhiteNoiseImage(const clString& id)
{
    return scinew WhiteNoiseImage(id);
}
}

WhiteNoiseImage::WhiteNoiseImage(const clString& id)
: Module("WhiteNoiseImage", id, Source), xres("xres", id, this),
  yres("yres", id, this)
{
    // Create the output port
    oport=scinew ImageOPort(this, "Image", ImageIPort::Atomic);
    add_oport(oport);
}

WhiteNoiseImage::WhiteNoiseImage(const WhiteNoiseImage& copy, int deep)
: Module(copy, deep), xres("xres", id, this), yres("yres", id, this)
{
    NOT_FINISHED("WhiteNoiseImage::WhiteNoiseImage");
}

WhiteNoiseImage::~WhiteNoiseImage()
{
}

Module* WhiteNoiseImage::clone(int deep)
{
    return scinew WhiteNoiseImage(*this, deep);
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
