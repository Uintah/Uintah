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
  RM("Binop",	     "Image", make_Binop,	    tcl+"/Binop.tcl");
  RM("Edge",	     "Image", make_Edge,	    tcl+"/Edge.tcl");
  RM("FFT",	     "Image", make_FFT,	            tcl+"/FFT.tcl");
  RM("FFTImage",     "Image", make_FFTImage,	    tcl+"/FFTImage.tcl");
  RM("FilterImage",  "Image", make_FilterImage,     tcl+"/FilterImage.tcl");
  RM("Gauss",	     "Image", make_Gauss,	    tcl+"/Gauss.tcl");
  RM("Hist",	     "Image", make_Hist,	    tcl+"/Hist.tcl");
  RM("HistEq",	     "Image", make_HistEq,	    tcl+"/HistEq.tcl");
  RM("IFFT",	     "Image", make_IFFT,	    tcl+"/IFFT.tcl");
  RM("IFFTImage",    "Image", make_IFFTImage,	    tcl+"/IFFTImage.tcl");
  RM("ImageConvolve","Image", make_ImageConvolve,   tcl+"/ImageConvolve.tcl");
  RM("ImageGen",     "Image", make_ImageGen,	    tcl+"/ImageGen.tcl");
  RM("ImageSel",     "Image", make_ImageSel,	    tcl+"/ImageSel.tcl");
  RM("ImageToGeom",  "Image", make_ImageToGeom,     tcl+"/ImageToGeom.tcl");
  RM("Noise",	     "Image", make_Noise,	    tcl+"/Noise.tcl");
  RM("PMFilterImage","Image", make_PMFilterImage,   tcl+"/PMFilterImage.tcl");
  RM("Radon",	     "Image", make_Radon,	    tcl+"/Radon.tcl");
  RM("Segment",	     "Image", make_Segment,	    tcl+"/Segment.tcl");
  RM("Sharpen",	     "Image", make_Sharpen,	    tcl+"/Sharpen.tcl");
  RM("Snakes",	     "Image", make_Snakes,	    tcl+"/Snakes.tcl");
  RM("Subsample",    "Image", make_Subsample,	    tcl+"/Subsample.tcl");
  RM("Ted",	     "Image", make_Ted,	            tcl+"/Ted.tcl");
  RM("Threshold",    "Image", make_Threshold,	    tcl+"/Threshold.tcl");
  //  RM("TiffReader",   "Image",     make_TiffReader,	    tcl+"/TiffReader.tcl");
  RM("Transforms",   "Image", make_Transforms,	    tcl+"/Transforms.tcl");
  RM("Turk",	     "Image", make_Turk,	    tcl+"/Turk.tcl");
  RM("Unop",	     "Image", make_Unop,	    tcl+"/Unop.tcl");
  RM("ViewHist",     "Image", make_ViewHist,	    tcl+"/ViewHist.tcl");
  RM("WhiteNoiseImage","WhiteNoiseImage",make_WhiteNoiseImage,tcl+"/WhiteNoiseImage.tcl");


    // Writers
  //  RM("TiffWriter",   "Writers",     make_TiffWriter,	    tcl+"/TiffWriter.tcl");

  cerr << "Initfn done -- TCL path was " << tcl << "\n";
}
}
