//static char *id="@(#) $Id"

/*
 *  IFFTImage.cc:  Unfinished modules
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
#include <SCICore/Math/fft.h>

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCIRun::Datatypes;

class IFFTImage : public Module {
    ImageIPort* iport;
    ImageOPort* oport;
public:
    IFFTImage(const clString& id);
    virtual ~IFFTImage();
    virtual void execute();
};

  extern "C" Module* make_IFFTImage(const clString& id)
    {
      return scinew IFFTImage(id);
    }

IFFTImage::IFFTImage(const clString& id)
: Module("IFFTImage", id, Filter)
{
    iport=scinew ImageIPort(this, "Frequency domain", ImageIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=scinew ImageOPort(this, "Spatial domain", ImageIPort::Atomic);
    add_oport(oport);
}

IFFTImage::~IFFTImage()
{
}

void IFFTImage::execute()
{
    ImageHandle in;
    if(!iport->get(in))
	return;
    ImageHandle out=in->clone();
    unsigned long flops, refs;
    int xres=out->xres();
    int yres=out->yres();
    fft2d_float(out->rows[0], out->xres(), -1, &flops, &refs);
    float* p=out->rows[0];
    float scale=1./(xres*yres);
    //cout << "scale=" << scale << endl;
    for(int y=0;y<yres;y++){
	for(int x=0;x<xres;x++){
	    *p++ *= scale;
	    *p++ *= scale;
	}
    }
    oport->send(out);
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.6  2000/03/17 09:29:03  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.5  1999/09/08 02:26:59  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/31 08:55:32  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:48:56  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:39:59  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:52  mcq
// Initial commit
//
// Revision 1.2  1999/04/30 01:11:53  dav
// moved TiffReader to SCIRun from PSECore
//
// Revision 1.1  1999/04/29 22:26:32  dav
// Added image files to SCIRun
//
//
