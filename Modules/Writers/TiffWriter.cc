/*
 *  TiffWriter.cc: TiffWriter class
 *  Written by:
 *
 *    Scott Morris
 *    July 1997
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>

#include <stdio.h>
#include <stdlib.h>
#include "tiffio.h"

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
    TiffWriter(const TiffWriter&, int deep=0);
    virtual ~TiffWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_TiffWriter(const clString& id)
{
    return scinew TiffWriter(id);
}
};

TiffWriter::TiffWriter(const clString& id)
: Module("TiffWriter", id, Source), filename("filename", id, this),
  resolution("resolution",id,this)
{
    // Create the input data handle and port
    inscalarfield=scinew ScalarFieldIPort(this, "Scalar Field", ScalarFieldIPort::Atomic);
    add_iport(inscalarfield);
    newgrid = new ScalarFieldRG;
}

TiffWriter::TiffWriter(const TiffWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  resolution("resolution",id,this)
{
    NOT_FINISHED("TiffWriter::TiffWriter");
}

TiffWriter::~TiffWriter()
{
}

Module* TiffWriter::clone(int deep)
{
    return scinew TiffWriter(*this, deep);
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




