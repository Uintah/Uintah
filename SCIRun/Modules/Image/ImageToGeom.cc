//static char *id="@(#) $Id"

/*
 *  ImageToGeom.cc:  Unfinished modules
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
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCIRun/Datatypes/Image/ImagePort.h>
#include <SCICore/Geom/GeomGrid.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace SCIRun {
namespace Modules {

using namespace SCICore::TclInterface;

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCIRun::Datatypes;

class ImageToGeom : public Module {
    ImageIPort* iport;
    GeometryOPort* ogeom;
    double scale;
    int have_scale;
public:
    TCLint format;
    TCLdouble heightscale;
    TCLint downsample;
    ImageToGeom(const clString& id);
    virtual ~ImageToGeom();
    virtual void execute();

    int oldid;
};

  Module* make_ImageToGeom(const clString& id)
    {
      return scinew ImageToGeom(id);
    }

ImageToGeom::ImageToGeom(const clString& id)
: Module("ImageToGeom", id, Filter), format("format", id, this),
  heightscale("heightscale", id, this), downsample("downsample", id, this)
{
    iport=scinew ImageIPort(this, "Image", ImageIPort::Atomic);
    add_iport(iport);
    // Create the output port
    ogeom=scinew GeometryOPort(this, "Geom object", ImageIPort::Atomic);
    add_oport(ogeom);
    oldid=0;
    have_scale=0;
}

ImageToGeom::~ImageToGeom()
{
}

void ImageToGeom::execute()
{
    ImageHandle image;
    if(!iport->get(image))
	return;
    //cerr << "generation=" << image->generation << endl;
    int form=format.get();
    GeomObj* obj=0;
    double hs(heightscale.get());
    switch(form){
    case 0:
	{
	    int xres=image->xres();
	    int yres=image->yres();
#if 0
	    GeomGrid* grid=new GeomGrid(image->xres(), image->yres(),
					Point(0,0,0),
					Vector(xres-1,0,0),
					Vector(0,yres-1,0),
					GeomGrid::WithNormals);
	    for(int j=0;j<yres;j++){
		for(int i=0;i<xres;i++){
		    float value=image->getr(i,j);
		    grid->setn(i,j,hs*value);
		}
	    }
	    grid->compute_normals();
	    obj=grid;
#endif
	}
	break;
    default:
	error("Unknown format");
	break;
    }

    if(oldid != 0)
	ogeom->delObj(oldid);
    if(obj != 0){	
	oldid=ogeom->addObj(obj, "Image");
    } else {
	oldid=0;
    }
    ogeom->flushViews();
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.5  1999/09/08 02:27:00  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/31 08:55:33  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:48:57  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:01  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:53  mcq
// Initial commit
//
// Revision 1.2  1999/04/30 01:11:53  dav
// moved TiffReader to SCIRun from PSECore
//
// Revision 1.1  1999/04/29 22:26:33  dav
// Added image files to SCIRun
//
//
