//static char *id="@(#) $Id"

/*
 *  FilterImage.cc:  Unfinished modules
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

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCIRun::Datatypes;

class FilterImage : public Module {
    ImageIPort* iimage;
    ImageIPort* ifilter;
    ImageOPort* oport;
public:
    FilterImage(const clString& id);
    virtual ~FilterImage();
    virtual void execute();
};

Module* make_FilterImage(const clString& id)
{
    return scinew FilterImage(id);
}

FilterImage::FilterImage(const clString& id)
: Module("FilterImage", id, Filter)
{
    iimage=scinew ImageIPort(this, "Freq. domain image", ImageIPort::Atomic);
    add_iport(iimage);
    ifilter=scinew ImageIPort(this, "Freq. domain filter", ImageIPort::Atomic);
    add_iport(ifilter);
    // Create the output port
    oport=scinew ImageOPort(this, "Freq. domain filtered image", ImageIPort::Atomic);
    add_oport(oport);
}

FilterImage::~FilterImage()
{
}

void filter_image(int xres, int yres, float* out, float* image, float* filter)
{
    for(int y=0;y<yres;y++){
	for(int x=0;x<xres;x++){
	    float ir=*image++;
	    float ii=*image++;
	    float fr=*filter++;
	    float fi=*filter++;
	    float or=ir*fr-ii*fi;
	    float oi=ii*fr-ir*fi;
	    *out++=or;
	    *out++=oi;
	}
    }
}

void FilterImage::execute()
{
    ImageHandle image;
    ImageHandle filter;
    if(!iimage->get(image))
	return;
    if(!ifilter->get(filter))
	return;
    ImageHandle out=new Image(image->xres(), image->yres());
    filter_image(image->xres(), image->yres(),
		 out->rows[0], image->rows[0], filter->rows[0]);
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
// Revision 1.2  1999/04/30 01:11:52  dav
// moved TiffReader to SCIRun from PSECore
//
// Revision 1.1  1999/04/29 22:26:31  dav
// Added image files to SCIRun
//
//

