//static char *id="@(#) $Id$";

/*
 *  TiffWriter.cc: TiffWriter class
 *  Written by:
 *
 *    Scott Morris
 *    July 1997
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <stdio.h>
#include <stdlib.h>
#include "tiffio.h"

namespace PSECore {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class TiffWriter : public Module {
    ScalarFieldIPort *inscalarfield;
    TCLstring filename;

    clString old_filename;
    TIFF *tif;
    unsigned long imagelength;
    unsigned char *buf;
    unsigned short *buf16;
    long row;
    int x,gen;
    ScalarFieldRG *ingrid,*newgrid;

    TCLstring resolution;
  
public:
    TiffWriter(const clString& id);
    virtual ~TiffWriter();
    virtual void execute();
};

extern "C" Module* make_TiffWriter(const clString& id)
{
    return scinew TiffWriter(id);
}

TiffWriter::TiffWriter(const clString& id)
: Module("TiffWriter", id, Source), filename("filename", id, this),
  resolution("resolution",id,this)
{
    // Create the input data handle and port
    inscalarfield=scinew ScalarFieldIPort(this, "Scalar Field", ScalarFieldIPort::Atomic);
    add_iport(inscalarfield);
    newgrid = new ScalarFieldRG;
}

TiffWriter::~TiffWriter()
{
}

void TiffWriter::execute()
{
    clString fn(filename.get());
 //   if(!handle.get_rep() || fn != old_filename){

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
      return;

    ingrid = sfield->getRG();

    if (!ingrid)
      return;

    gen=ingrid->generation;
    
    if  ((gen!=newgrid->generation) || (fn != old_filename) || \
	 (!sfield.get_rep())) {

        newgrid=new ScalarFieldRG(*ingrid);

	int nx=ingrid->grid.dim1();
	int ny=ingrid->grid.dim2();
	int nz=ingrid->grid.dim3();
	newgrid->resize(nx,ny,nz);

	double maxval,minval;

	ingrid->get_minmax(minval,maxval);  // I don't know what I
	                                    // was thinking here.. max?
	
        old_filename=fn;

	tif = TIFFOpen(fn(), "w");
	if (!tif) {
	  error("Something is wrong with your filename.\n");
	  return;
	}

	int xdim=ingrid->grid.dim2(),ydim=ingrid->grid.dim1(), \
	  zdim=ingrid->grid.dim3();
	uint16 bps,spp;

	clString res(resolution.get());
	
	if ((res=="RGB") && (zdim==3)) {   //RGB
	  bps=8;
	  spp=3;
	  maxval=255;
	};
	if (res=="8bit") {
	  bps=8;
	  spp=1;
	  maxval=255;
	}
        if (res=="16bit") {
	  bps=16;
	  spp=1;
	  maxval=65535;
	}

	// These options will auto-scale the image to fit in 8 or 16 bits
	
	if ((res=="ScaleRGB") && (zdim==3)) {
	  bps=8;
	  spp=3;
	}
	if (res=="Scale8bit") {
	  bps=8;
	  spp=1;
	}
	if (res=="Scale16bit") {
	  bps=16;
	  spp=1;
	}
	
	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, xdim);
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, ydim);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bps);
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, spp);
	TIFFSetField(tif, TIFFTAG_PLANARCONFIG, 1);
	TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 1);
	
	cerr << "--TiffWriter--\n";
	cerr << "Writing to : " << fn << "\n";
	cerr << "Dimensions: " << xdim << " " << ydim << "\n";
	cerr << "spp: " << spp << "  bps: " << bps << "\n";
	cerr << "Max value: " << maxval << "\n";
	
	if (bps==16) {
	  buf16 = new unsigned short[xdim];

	  for (row = 0; row < ydim; row++){
	    for (x=0; x<(xdim); x++)   
	      buf16[x]=(ingrid->grid(ydim-1-row,x,0)/maxval)*65535;  
	    TIFFWriteScanline(tif, buf16, row, 0);
	  }
	  
	} else
	{
	  buf = new unsigned char[xdim*spp];

	  if (spp==1) {
	    for (row = 0; row < ydim; row++){
	      for (x=0; x<xdim; x++)   
		buf[x]=(ingrid->grid(ydim-row-1,x,0)/maxval)*255;
	      TIFFWriteScanline(tif, buf, row, 0);
	    }	    
	  }
	  if (spp==3) {  //Let's try RGB
	    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 2);
	    for (row = 0; row < ydim; row++){
	      for (x=0; x<(xdim*3); x+=3) {
		buf[x] = (ingrid->grid(ydim-row-1,x/3,0)/maxval)*255;
	        buf[x+1] = (ingrid->grid(ydim-row-1,x/3,1)/maxval)*255;
	        buf[x+2] = (ingrid->grid(ydim-row-1,x/3,2)/maxval)*255;
	      }
	      TIFFWriteScanline(tif, buf, row, 0);
	    }
	  }
	}
	TIFFClose(tif);
	cerr << "--ENDTiffWriter--\n";
       
    }

}

} // End namespace Modules
} // End namespace PSECore

//
// $Log$
// Revision 1.6  2000/03/17 09:29:20  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.5  1999/09/08 02:27:06  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/31 08:55:38  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:49:02  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:04  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:57  mcq
// Initial commit
//
// Revision 1.1  1999/06/21 20:27:01  dav
// added TiffWriter.cc to SCIRun/Modules/Writers
//
// Revision 1.3  1999/04/27 22:58:06  dav
// updates in Modules for Datatypes
//
// Revision 1.2  1999/04/25 03:11:11  dav
// picking up anychanges I have made.
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
