#include <PSECore/Dataflow/PackageDB.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>

#define MP(makesuf) \
namespace DaveW { namespace Modules { using namespace PSECore::Dataflow;\
  Module* make_##makesuf (const clString& id); } }\

// CS684
MP(BldBRDF)
MP(BldScene)
MP(RTrace)
MP(Radiosity)
MP(RayMatrix)
MP(RayTest)
MP(XYZtoRGB)

// EEG
MP(BldEEGMesh)
MP(Coregister)
MP(InvEEGSolve)
MP(RescaleSegFld)
MP(STreeExtractSurf)
MP(SegFldOps)
MP(SegFldToSurfTree)
MP(SelectSurfNodes)
MP(Taubin)
MP(Thermal)

// EGI
MP(Anneal)

// FDM
MP(BuildFDMatrix)
MP(BuildFDField)

// SiRe
MP(SiReAll)
MP(SiReCrunch)
MP(SiReInput)
MP(SiReOutput)

using namespace PSECore::Dataflow;
using namespace DaveW::Modules;

#define RM(a,b,c,d) packageDB.registerModule("PSE Common",a,b,c,d)

extern "C" {
void initPackage(const clString& tcl) {

  // CS684
  RM("CS684",	      "BldBRDF",		 make_BldBRDF,		    tcl+"/BldBRDF.tcl");
  RM("CS684",	      "BldScene",		 make_BldScene,		    tcl+"/BldScene.tcl");
  RM("CS684",	      "RTrace",		 	 make_RTrace,		    tcl+"/RTrace.tcl");
  RM("CS684",	      "Radiosity",		 make_Radiosity,	    tcl+"/Radiosity.tcl");
  RM("CS684",	      "RayMatrix",		 make_RayMatrix,	    tcl+"/RayMatrix.tcl");
  RM("CS684",	      "RayTest",		 make_RayTest,		    tcl+"/RayTest.tcl");
  RM("CS684",	      "XYZtoRGB",		 make_XYZtoRGB,		    tcl+"/XYZtoRGB.tcl");

  // EEG
  RM("EEG",	      "BldEEGMesh",		 make_BldEEGMesh,	    tcl+"/BldEEGMesh.tcl");
  RM("EEG",	      "Coregister",		 make_Coregister,	    tcl+"/Coregister.tcl");
  RM("EEG",	      "InvEEGSolve",	 	 make_InvEEGSolve,	    tcl+"/InvEEGSolve.tcl");
  RM("EEG",	      "RescaleSegFld",		 make_RescaleSegFld,	    tcl+"/RescaleSegFld.tcl");
  RM("EEG",	      "STreeExtractSurf",	 make_STreeExtractSurf,	    tcl+"/STreeExtractSurf.tcl");
  RM("EEG",	      "SegFldOps",		 make_SegFldOps,	    tcl+"/SegFldOps.tcl");
  RM("EEG",	      "SelectSurfNodes",	 make_SelectSurfNodes,	    tcl+"/SelectSurfNodes.tcl");
  RM("EEG",	      "Taubin",			 make_Taubin,		    tcl+"/Taubin.tcl");
  RM("EEG",	      "Thermal",		 make_Thermal,		    tcl+"/Thermal.tcl");

  // EGI
  RM("EGI",	      "Anneal",		 	 make_Anneal,		    tcl+"/Anneal.tcl");

  // FDM
  RM("FDM",	      "BuildFDField",		 make_BuildFDField,	    tcl+"/BuildFDField.tcl");
  RM("FDM",	      "BuildFDMatrix",		 make_BuildFDMatrix,	    tcl+"/BuildFDMatrix.tcl");

  // SiRe
  RM("SiRe",	      "SiReAll",		 make_SiReAll,	    	    tcl+"/SiReAll.tcl");
  RM("SiRe",	      "SiReCrunch",		 make_SiReCrunch,	    tcl+"/SiReCrunch.tcl");
  RM("SiRe",	      "SiReInput",		 make_SiReInput,	    tcl+"/SiReInput.tcl");
  RM("SiRe",	      "SiReOutput",		 make_SiReOutput,	    tcl+"/SiReOutput.tcl");

  cerr << "Initfn done -- TCL path was " << tcl << "\n";
}
}
