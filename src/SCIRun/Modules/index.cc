#include <PSECore/Dataflow/PackageDB.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>

#define MP(makesuf) \
namespace SCIRun { namespace Modules { using namespace PSECore::Dataflow;\
  Module* make_##makesuf (const clString& id); } }\

// Image
MP(Binop)
MP(Edge)
MP(FFT)
MP(FFTImage)
MP(FilterImage)
MP(Gauss)
MP(Hist)
MP(HistEq)
MP(IFFT)
MP(IFFTImage)
MP(ImageConvolve)
MP(ImageGen)
MP(ImageSel)
MP(ImageToGeom)
MP(Noise)
MP(PMFilterImage)
MP(Radon)
MP(Segment)
MP(Sharpen)
MP(Snakes)
MP(Subsample)
MP(Ted)
MP(Threshold)
      //MP(TiffReader)
MP(Transforms)
MP(Turk)
MP(Unop)
MP(ViewHist)
MP(WhiteNoiseImage)

// Writers
      //MP(TiffWriter)

using namespace PSECore::Dataflow;
using namespace SCIRun::Modules;

#define RM(a,b,c,d) packageDB.registerModule("SCIRun",a,b,c,d)

extern "C" {
void initPackage(const clString& tcl) {

    // Image
  RM("Binop",	     "Binop",	       make_Binop,	    tcl+"/Binop.tcl");
  RM("Edge",	     "Edge",	       make_Edge,	    tcl+"/Edge.tcl");
  RM("FFT",	     "FFT",	       make_FFT,	    tcl+"/FFT.tcl");
  RM("FFTImage",     "FFTImage",       make_FFTImage,	    tcl+"/FFTImage.tcl");
  RM("FilterImage",  "FilterImage",    make_FilterImage,    tcl+"/FilterImage.tcl");
  RM("Gauss",	     "Gauss",	       make_Gauss,	    tcl+"/Gauss.tcl");
  RM("Hist",	     "Hist",	       make_Hist,	    tcl+"/Hist.tcl");
  RM("HistEq",	     "HistEq",	       make_HistEq,	    tcl+"/HistEq.tcl");
  RM("IFFT",	     "IFFT",	       make_IFFT,	    tcl+"/IFFT.tcl");
  RM("IFFTImage",    "IFFTImage",      make_IFFTImage,	    tcl+"/IFFTImage.tcl");
  RM("ImageConvolve","ImageConvolve",  make_ImageConvolve,  tcl+"/ImageConvolve.tcl");
  RM("ImageGen",     "ImageGen",       make_ImageGen,	    tcl+"/ImageGen.tcl");
  RM("ImageSel",     "ImageSel",       make_ImageSel,	    tcl+"/ImageSel.tcl");
  RM("ImageToGeom",  "ImageToGeom",    make_ImageToGeom,    tcl+"/ImageToGeom.tcl");
  RM("Noise",	     "Noise",	       make_Noise,	    tcl+"/Noise.tcl");
  RM("PMFilterImage","PMFilterImage",  make_PMFilterImage,		tcl+"/PMFilterImage.tcl");
  RM("Radon",	     "Radon",	       make_Radon,	    tcl+"/Radon.tcl");
  RM("Segment",	     "Segment",	       make_Segment,	    tcl+"/Segment.tcl");
  RM("Sharpen",	     "Sharpen",	       make_Sharpen,	    tcl+"/Sharpen.tcl");
  RM("Snakes",	     "Snakes",	       make_Snakes,	    tcl+"/Snakes.tcl");
  RM("Subsample",    "Subsample",      make_Subsample,	    tcl+"/Subsample.tcl");
  RM("Ted",	     "Ted",	       make_Ted,	    tcl+"/Ted.tcl");
  RM("Threshold",    "Threshold",      make_Threshold,	    tcl+"/Threshold.tcl");
  //  RM("TiffReader",   "TiffReader",     make_TiffReader,	    tcl+"/TiffReader.tcl");
  RM("Transforms",   "Transforms",     make_Transforms,	    tcl+"/Transforms.tcl");
  RM("Turk",	     "Turk",	       make_Turk,	    tcl+"/Turk.tcl");
  RM("Unop",	     "Unop",	       make_Unop,	    tcl+"/Unop.tcl");
  RM("ViewHist",     "ViewHist",       make_ViewHist,	    tcl+"/ViewHist.tcl");
  RM("WhiteNoiseImage","WhiteNoiseImage",make_WhiteNoiseImage,tcl+"/WhiteNoiseImage.tcl");


    // Writers
  //  RM("TiffWriter",   "TiffWriter",     make_TiffWriter,	    tcl+"/TiffWriter.tcl");

  cerr << "Initfn done -- TCL path was " << tcl << "\n";
}
}
