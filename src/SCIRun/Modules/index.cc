#include <PSECore/Dataflow/PackageDB.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <iostream.h>

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

// Mesh
MP(Delaunay)
MP(ExtractMesh)
MP(HexMeshToGeom)
MP(InsertDelaunay)
MP(MakeScalarField)
MP(MeshBoundary)
MP(MeshInterpVals)
MP(MeshRender)
MP(MeshToGeom)
MP(MeshView)

// Writers
      //MP(TiffWriter)

using namespace PSECore::Dataflow;
using namespace SCIRun::Modules;

#define RM(a,b,c,d) packageDB.registerModule("SCIRun",a,b,c,d)
#define RMI(b,c,d) RM("Image",b,c,d)
#define RMM(b,c,d) RM("Mesh",b,c,d)

extern "C" {
void initPackage(const clString& tcl) {

    // Image
  RMI("Binop", 		make_Binop,	    tcl+"/Binop.tcl");
  RMI("Edge", 		make_Edge,	    tcl+"/Edge.tcl");
  RMI("FFT", 		make_FFT,	            tcl+"/FFT.tcl");
  RMI("FFTImage", 	make_FFTImage,	    tcl+"/FFTImage.tcl");
  RMI("FilterImage", 	make_FilterImage,     tcl+"/FilterImage.tcl");
  RMI("Gauss", 		make_Gauss,	    tcl+"/Gauss.tcl");
  RMI("Hist", 		make_Hist,	    tcl+"/Hist.tcl");
  RMI("HistEq", 	make_HistEq,	    tcl+"/HistEq.tcl");
  RMI("IFFT", 		make_IFFT,	    tcl+"/IFFT.tcl");
  RMI("IFFTImage", 	make_IFFTImage,	    tcl+"/IFFTImage.tcl");
  RMI("ImageConvolve", 	make_ImageConvolve,   tcl+"/ImageConvolve.tcl");
  RMI("ImageGen", 	make_ImageGen,	    tcl+"/ImageGen.tcl");
  RMI("ImageSel", 	make_ImageSel,	    tcl+"/ImageSel.tcl");
  RMI("ImageToGeom", 	make_ImageToGeom,     tcl+"/ImageToGeom.tcl");
  RMI("Noise",		make_Noise,	    tcl+"/Noise.tcl");
  RMI("PMFilterImage",	make_PMFilterImage,   tcl+"/PMFilterImage.tcl");
  RMI("Radon", 		make_Radon,	    tcl+"/Radon.tcl");
  RMI("Segment", 	make_Segment,	    tcl+"/Segment.tcl");
  RMI("Sharpen", 	make_Sharpen,	    tcl+"/Sharpen.tcl");
  RMI("Snakes", 	make_Snakes,	    tcl+"/Snakes.tcl");
  RMI("Subsample", 	make_Subsample,	    tcl+"/Subsample.tcl");
  RMI("Ted", 		make_Ted,	            tcl+"/Ted.tcl");
  RMI("Threshold", 	make_Threshold,	    tcl+"/Threshold.tcl");
  //  RMI("TiffReader",     make_TiffReader,	    tcl+"/TiffReader.tcl");
  RMI("Transforms", 	make_Transforms,	    tcl+"/Transforms.tcl");
  RMI("Turk", 		make_Turk,	    tcl+"/Turk.tcl");
  RMI("Unop", 		make_Unop,	    tcl+"/Unop.tcl");
  RMI("ViewHist", 	make_ViewHist,	    tcl+"/ViewHist.tcl");
  RMI("WhiteNoiseImage",make_WhiteNoiseImage,tcl+"/WhiteNoiseImage.tcl");

   // Mesh
  RMM("Delaunay",	make_Delaunay,	    tcl+"/Delaunay.tcl");
  RMM("ExtractMesh",	make_ExtractMesh,	    "");
  RMM("HexMeshToGeom",	make_HexMeshToGeom,	    "");
  RMM("InsertDelaunay",	make_InsertDelaunay,	    "");
  RMM("MakeScalarField",	make_MakeScalarField,	"");
  RMM("MeshBoundary",	make_MeshBoundary,	    "");
  RMM("MeshInterpVals",	make_MeshInterpVals,	    tcl+"/MeshInterpVals.tcl");
  RMM("MeshRender",	make_MeshRender,	    "");
  RMM("MeshToGeom",	make_MeshToGeom,	    tcl+"/MeshToGeom.tcl");
  RMM("MeshView",	make_MeshView,	    tcl+"/MeshView.tcl");

    // Writers
  //  RM("Writers", "TiffWriter",     make_TiffWriter,	    tcl+"/TiffWriter.tcl");

  cerr << "Initfn done -- TCL path was " << tcl << "\n";
}
}
