//static char *id="@(#) $Id"

/*
 *  FFTImage.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCIRun/Datatypes/Image/ImagePort.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/fft.h>

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCIRun::Datatypes;

class FFTImage : public Module {
    ImageIPort* iport;
    ImageOPort* oport;
public:
    FFTImage(const clString& id);
    virtual ~FFTImage();
    virtual void execute();
};

Module* make_FFTImage(const clString& id)
{
    return scinew FFTImage(id);
}

FFTImage::FFTImage(const clString& id)
: Module("FFTImage", id, Filter)
{
    iport=scinew ImageIPort(this, "Spatial domain", ImageIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=scinew ImageOPort(this, "Frequency domain", ImageIPort::Atomic);
    add_oport(oport);
}

FFTImage::~FFTImage()
{
}

void FFTImage::execute()
{
    ImageHandle in;
    if(!iport->get(in))
	return;
    ImageHandle out=in->clone();
    unsigned long flops, refs;

    fft2d_float(out->rows[0], out->xres(), 1, &flops, &refs);
    oport->send(out);
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.5  1999/09/08 02:26:58  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/31 08:55:31  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:48:55  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:39:58  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:51  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:31  dav
// Added image files to SCIRun
//
//

