//static char *id="@(#) $Id"

/*
 *  HistEq.cc:  Histogram Equalization
 *
 *  Written by:
 *    Scott Morris
 *    June 1998
 */

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <SCICore/Geom/GeomGrid.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Thread.h>
#include <iostream>
using std::cerr;

#include <math.h>
#include <stdlib.h>

using namespace SCICore::Thread;

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;

class HistEq : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;

   ScalarFieldRG* newgrid;
   ScalarFieldRG*  rg;

   double min,max;       // Max and min values of the scalarfield

  TCLdouble clip,bins,conx,cony;
  
   int np; // number of proccesors
  
public:
   HistEq(const clString& id);
   virtual ~HistEq();
   virtual void execute();

//   void tcl_command( TCLArgs&, void *);

   void do_HistEq(int proc);

};

  extern "C" Module* make_HistEq(const clString& id)
    {
      return scinew HistEq(id);
    }

static clString module_name("HistEq");

HistEq::HistEq(const clString& id)
: Module("HistEq", id, Filter),
  bins("bins",id,this),clip("clip",id,this),conx("conx",id,this),
  cony("cony",id,this)
{
    // Create the input ports
    // Need a scalar field
  
    inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_iport( inscalarfield);

    // Create the output port
    outscalarfield = scinew ScalarFieldOPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_oport( outscalarfield);
    newgrid=new ScalarFieldRG;
}

HistEq::~HistEq()
{
}

void HistEq::do_HistEq(int proc)    
{
  int start = (rg->grid.dim2())*proc/np;
  int end   = (proc+1)*(rg->grid.dim2())/np;

  for(int x=start; x<end; x++) {
    for(int y=0; y<rg->grid.dim1(); y++) {
      if (rg->grid(y,x,0)>0)
	newgrid->grid(0,rg->grid(y,x,0),0)++;
    }
  }
}

typedef unsigned char kz_pixel_t;
#define uiNR_OF_GREY (256)

const unsigned int uiMAX_REG_X = 16;
const unsigned int uiMAX_REG_Y = 16;


void ClipHistogram (unsigned long* pulHistogram, unsigned int
		     uiNrGreylevels, unsigned long ulClipLimit)
/* This function performs clipping of the histogram and redistribution of bins.
 * The histogram is clipped and the number of excess pixels is counted. Afterwards
 * the excess pixels are equally redistributed across the whole histogram (providing
 * the bin count is smaller than the cliplimit).
 */
{
    unsigned long* pulBinPointer, *pulEndPointer, *pulHisto;
    unsigned long ulNrExcess, ulUpper, ulBinIncr, ulStepSize, i;
    long lBinExcess;

    ulNrExcess = 0;  pulBinPointer = pulHistogram;
    for (i = 0; i < uiNrGreylevels; i++) { /* calculate total number of excess pixels */
	lBinExcess = (long) pulBinPointer[i] - (long) ulClipLimit;
	if (lBinExcess > 0) ulNrExcess += lBinExcess;	  /* excess in current bin */
    };

    /* Second part: clip histogram and redistribute excess pixels in each bin */
    ulBinIncr = ulNrExcess / uiNrGreylevels;		  /* average binincrement */
    ulUpper =  ulClipLimit - ulBinIncr;	 /* Bins larger than ulUpper set to cliplimit */

    for (i = 0; i < uiNrGreylevels; i++) {
      if (pulHistogram[i] > ulClipLimit) pulHistogram[i] = ulClipLimit; /* clip bin */
      else {
	  if (pulHistogram[i] > ulUpper) {		/* high bin count */
	      ulNrExcess -= pulHistogram[i] - ulUpper; pulHistogram[i]=ulClipLimit;
	  }
	  else {					/* low bin count */
	      ulNrExcess -= ulBinIncr; pulHistogram[i] += ulBinIncr;
	  }
       }
    }

    while (ulNrExcess) {   /* Redistribute remaining excess  */
	pulEndPointer = &pulHistogram[uiNrGreylevels]; pulHisto = pulHistogram;

	while (ulNrExcess && pulHisto < pulEndPointer) {
	    ulStepSize = uiNrGreylevels / ulNrExcess;
	    if (ulStepSize < 1) ulStepSize = 1;		  /* stepsize at least 1 */
	    for (pulBinPointer=pulHisto; pulBinPointer < pulEndPointer && ulNrExcess;
		 pulBinPointer += ulStepSize) {
		if (*pulBinPointer < ulClipLimit) {
		    (*pulBinPointer)++;	 ulNrExcess--;	  /* reduce excess */
		}
	    }
	    pulHisto++;		  /* restart redistributing on other bin location */
	}
    }
}
void MakeHistogram (kz_pixel_t* pImage, unsigned int uiXRes,
		unsigned int uiSizeX, unsigned int uiSizeY,
		unsigned long* pulHistogram,
		unsigned int uiNrGreylevels, kz_pixel_t* pLookupTable)
/* This function classifies the greylevels present in the array image into
 * a greylevel histogram. The pLookupTable specifies the relationship
 * between the greyvalue of the pixel (typically between 0 and 4095) and
 * the corresponding bin in the histogram (usually containing only 128 bins).
 */
{
    kz_pixel_t* pImagePointer;
    unsigned int i;

    for (i = 0; i < uiNrGreylevels; i++) pulHistogram[i] = 0L; /* clear histogram */

    for (i = 0; i < uiSizeY; i++) {
	pImagePointer = &pImage[uiSizeX];
	while (pImage < pImagePointer) pulHistogram[pLookupTable[*pImage++]]++;
	pImagePointer += uiXRes;
	pImage = &pImagePointer[-uiSizeX];
    }
}

void MapHistogram (unsigned long* pulHistogram, kz_pixel_t Min, kz_pixel_t Max,
	       unsigned int uiNrGreylevels, unsigned long ulNrOfPixels)
/* This function calculates the equalized lookup table (mapping) by
 * cumulating the input histogram. Note: lookup table is rescaled in range [Min..Max].
 */
{
    unsigned int i;  unsigned long ulSum = 0;
    const float fScale = ((float)(Max - Min)) / (float)ulNrOfPixels;
    const unsigned long ulMin = (unsigned long) Min;

    for (i = 0; i < uiNrGreylevels; i++) {
	ulSum += pulHistogram[i]; pulHistogram[i]=(unsigned long)(ulMin+(float)ulSum*fScale);
	if (pulHistogram[i] > Max) pulHistogram[i] = Max;
    }
}

void MakeLut (kz_pixel_t * pLUT, kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrBins)
/* To speed up histogram clipping, the input image [Min,Max] is scaled down to
 * [0,uiNrBins-1]. This function calculates the LUT.
 */
{
    int i;
    const kz_pixel_t BinSize = (kz_pixel_t) (1 + (Max - Min) / uiNrBins);

    for (i = Min; i <= Max; i++)  pLUT[i] = (i - Min) / BinSize;
}

void Interpolate (kz_pixel_t * pImage, int uiXRes, unsigned long * pulMapLU,
     unsigned long * pulMapRU, unsigned long * pulMapLB,  unsigned long * pulMapRB,
     unsigned int uiXSize, unsigned int uiYSize, kz_pixel_t * pLUT)
/* pImage      - pointer to input/output image
 * uiXRes      - resolution of image in x-direction
 * pulMap*     - mappings of greylevels from histograms
 * uiXSize     - uiXSize of image submatrix
 * uiYSize     - uiYSize of image submatrix
 * pLUT	       - lookup table containing mapping greyvalues to bins
 * This function calculates the new greylevel assignments of pixels within a submatrix
 * of the image with size uiXSize and uiYSize. This is done by a bilinear interpolation
 * between four different mappings in order to eliminate boundary artifacts.
 * It uses a division; since division is often an expensive operation, I added code to
 * perform a logical shift instead when feasible.
 */
{
    const unsigned int uiIncr = uiXRes-uiXSize; /* Pointer increment after processing row */
    kz_pixel_t GreyValue; unsigned int uiNum = uiXSize*uiYSize; /* Normalization factor */

    unsigned int uiXCoef, uiYCoef, uiXInvCoef, uiYInvCoef, uiShift = 0;

    if (uiNum & (uiNum - 1))   /* If uiNum is not a power of two, use division */
    for (uiYCoef = 0, uiYInvCoef = uiYSize; uiYCoef < uiYSize;
	 uiYCoef++, uiYInvCoef--,pImage+=uiIncr) {
	for (uiXCoef = 0, uiXInvCoef = uiXSize; uiXCoef < uiXSize;
	     uiXCoef++, uiXInvCoef--) {
	    GreyValue = pLUT[*pImage];		   /* get histogram bin value */
	    *pImage++ = (kz_pixel_t ) ((uiYInvCoef * (uiXInvCoef*pulMapLU[GreyValue]
				      + uiXCoef * pulMapRU[GreyValue])
				+ uiYCoef * (uiXInvCoef * pulMapLB[GreyValue]
				      + uiXCoef * pulMapRB[GreyValue])) / uiNum);
	}
    }
    else {			   /* avoid the division and use a right shift instead */
	while (uiNum >>= 1) uiShift++;		   /* Calculate 2log of uiNum */
	for (uiYCoef = 0, uiYInvCoef = uiYSize; uiYCoef < uiYSize;
	     uiYCoef++, uiYInvCoef--,pImage+=uiIncr) {
	     for (uiXCoef = 0, uiXInvCoef = uiXSize; uiXCoef < uiXSize;
	       uiXCoef++, uiXInvCoef--) {
	       GreyValue = pLUT[*pImage];	  /* get histogram bin value */
	       *pImage++ = (kz_pixel_t)((uiYInvCoef* (uiXInvCoef * pulMapLU[GreyValue]
				      + uiXCoef * pulMapRU[GreyValue])
				+ uiYCoef * (uiXInvCoef * pulMapLB[GreyValue]
				      + uiXCoef * pulMapRB[GreyValue])) >> uiShift);
	    }
	}
    }
}


/************************** main function CLAHE ******************/
int CLAHE (kz_pixel_t* pImage, unsigned int uiXRes, unsigned int uiYRes,
	 kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrX, unsigned int uiNrY,
	      unsigned int uiNrBins, float fCliplimit)
/*   pImage - Pointer to the input/output image
 *   uiXRes - Image resolution in the X direction
 *   uiYRes - Image resolution in the Y direction
 *   Min - Minimum greyvalue of input image (also becomes minimum of output image)
 *   Max - Maximum greyvalue of input image (also becomes maximum of output image)
 *   uiNrX - Number of contextial regions in the X direction (min 2, max uiMAX_REG_X)
 *   uiNrY - Number of contextial regions in the Y direction (min 2, max uiMAX_REG_Y)
 *   uiNrBins - Number of greybins for histogram ("dynamic range")
 *   float fCliplimit - Normalized cliplimit (higher values give more contrast)
 * The number of "effective" greylevels in the output image is set by uiNrBins; selecting
 * a small value (eg. 128) speeds up processing and still produce an output image of
 * good quality. The output image will have the same minimum and maximum value as the input
 * image. A clip limit smaller than 1 results in standard (non-contrast limited) AHE.
 */
{
    unsigned int uiX, uiY;		  /* counters */
    unsigned int uiXSize, uiYSize, uiSubX, uiSubY; /* size of context. reg. and subimages */
    unsigned int uiXL, uiXR, uiYU, uiYB;  /* auxiliary variables interpolation routine */
    unsigned long ulClipLimit, ulNrPixels;/* clip limit and region pixel count */
    kz_pixel_t* pImPointer;		   /* pointer to image */
    kz_pixel_t aLUT[uiNR_OF_GREY];	    /* lookup table used for scaling of input image */
    unsigned long* pulHist, *pulMapArray; /* pointer to histogram and mappings*/
    unsigned long* pulLU, *pulLB, *pulRU, *pulRB; /* auxiliary pointers interpolation */

    if (uiNrX > uiMAX_REG_X) return -1;	   /* # of regions x-direction too large */
    if (uiNrY > uiMAX_REG_Y) return -2;	   /* # of regions y-direction too large */
    if (uiXRes % uiNrX) return -3;	  /* x-resolution no multiple of uiNrX */
    if (uiYRes & uiNrY) return -4;	  /* y-resolution no multiple of uiNrY */
    if (Max >= uiNR_OF_GREY) return -5;	   /* maximum too large */
    if (Min >= Max) return -6;		  /* minimum equal or larger than maximum */
    if (uiNrX < 2 || uiNrY < 2) return -7;/* at least 4 contextual regions required */
    if (fCliplimit == 1.0) return 0;	  /* is OK, immediately returns original image. */
    if (uiNrBins == 0) uiNrBins = 128;	  /* default value when not specified */

    pulMapArray=(unsigned long *)malloc(sizeof(unsigned long)*uiNrX*uiNrY*uiNrBins);
    if (pulMapArray == 0) return -8;	  /* Not enough memory! (try reducing uiNrBins) */

    uiXSize = uiXRes/uiNrX; uiYSize = uiYRes/uiNrY;  /* Actual size of contextual regions */
    ulNrPixels = (unsigned long)uiXSize * (unsigned long)uiYSize;

    if(fCliplimit > 0.0) {		  /* Calculate actual cliplimit	 */
       ulClipLimit = (unsigned long) (fCliplimit * (uiXSize * uiYSize) / uiNrBins);
       ulClipLimit = (ulClipLimit < 1UL) ? 1UL : ulClipLimit;
    }
    else ulClipLimit = 1UL<<14;		  /* Large value, do not clip (AHE) */
    MakeLut(aLUT, Min, Max, uiNrBins);	  /* Make lookup table for mapping of greyvalues */
    /* Calculate greylevel mappings for each contextual region */
    for (uiY = 0, pImPointer = pImage; uiY < uiNrY; uiY++) {
	for (uiX = 0; uiX < uiNrX; uiX++, pImPointer += uiXSize) {
	    pulHist = &pulMapArray[uiNrBins * (uiY * uiNrX + uiX)];
	    MakeHistogram(pImPointer,uiXRes,uiXSize,uiYSize,pulHist,uiNrBins,aLUT);
	    ClipHistogram(pulHist, uiNrBins, ulClipLimit);
	    MapHistogram(pulHist, Min, Max, uiNrBins, ulNrPixels);
	}
	pImPointer += (uiYSize - 1) * uiXRes;		  /* skip lines, set pointer */
    }

    /* Interpolate greylevel mappings to get CLAHE image */
    for (pImPointer = pImage, uiY = 0; uiY <= uiNrY; uiY++) {
	if (uiY == 0) {					  /* special case: top row */
	    uiSubY = uiYSize >> 1;  uiYU = 0; uiYB = 0;
	}
	else {
	    if (uiY == uiNrY) {				  /* special case: bottom row */
		uiSubY = uiYSize >> 1;	uiYU = uiNrY-1;	 uiYB = uiYU;
	    }
	    else {					  /* default values */
		uiSubY = uiYSize; uiYU = uiY - 1; uiYB = uiYU + 1;
	    }
	}
	for (uiX = 0; uiX <= uiNrX; uiX++) {
	    if (uiX == 0) {				  /* special case: left column */
		uiSubX = uiXSize >> 1; uiXL = 0; uiXR = 0;
	    }
	    else {
		if (uiX == uiNrX) {			  /* special case: right column */
		    uiSubX = uiXSize >> 1;  uiXL = uiNrX - 1; uiXR = uiXL;
		}
		else {					  /* default values */
		    uiSubX = uiXSize; uiXL = uiX - 1; uiXR = uiXL + 1;
		}
	    }

	    pulLU = &pulMapArray[uiNrBins * (uiYU * uiNrX + uiXL)];
	    pulRU = &pulMapArray[uiNrBins * (uiYU * uiNrX + uiXR)];
	    pulLB = &pulMapArray[uiNrBins * (uiYB * uiNrX + uiXL)];
	    pulRB = &pulMapArray[uiNrBins * (uiYB * uiNrX + uiXR)];
	    Interpolate(pImPointer,uiXRes,pulLU,pulRU,pulLB,pulRB,uiSubX,uiSubY,aLUT);
	    pImPointer += uiSubX;			  /* set pointer on next matrix */
	    cerr << "In here..\n";
	}
	pImPointer += (uiSubY - 1) * uiXRes;
    }
    free(pulMapArray);					  /* free space for histograms */
    return 0;						  /* return status OK */
}


void HistEq::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "HistEq cannot handle this field type\n";
      return;
    }
    gen=rg->generation;
    
    if (gen!=newgrid->generation){
    //  newgrid=new ScalarFieldRGint;
      // New input
    }
    newgrid=new ScalarFieldRG;

    rg->compute_minmax();
    rg->get_minmax(min,max);

    int ydim = rg->grid.dim1();
    int xdim = rg->grid.dim2();
    
    newgrid->resize(rg->grid.dim1(),rg->grid.dim2(),rg->grid.dim3());

    
    
    np = Thread::numProcessors();    
  
    cerr << "min/max : " << min << " " << max << "\n";
    
//    Task::multiprocess(np, start_HistEq, this);

    unsigned char *theimage = new unsigned char[xdim*ydim];
    int x;
    for (x=0; x<xdim; x++)
      for (int y=0; y<ydim; y++)
	theimage[y*xdim+x]=rg->grid(y,x,0);
    
    int res = CLAHE(theimage,rg->grid.dim2(),rg->grid.dim1(),min,max,
		    conx.get(),cony.get(),bins.get(),clip.get());
    if (res)
      cerr << "Input image must be a multiple of NX and NY.\n";
    for (x=0; x<xdim; x++)
      for (int y=0; y<ydim; y++)
	newgrid->grid(y,x,0)=theimage[y*xdim+x];
    
    outscalarfield->send( newgrid );
}

/*
void HistEq::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}
*/

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.7  2000/03/17 09:29:03  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:08:14  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/09/08 02:26:59  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/31 08:55:32  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:48:55  sparker
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
// Revision 1.2  1999/04/30 01:11:52  dav
// moved TiffReader to SCIRun from PSECore
//
// Revision 1.1  1999/04/29 22:26:32  dav
// Added image files to SCIRun
//
//
