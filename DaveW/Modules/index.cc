#include <PSECore/Dataflow/PackageDB.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <iostream>

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
MP(TopoSurfToGeom)

// EGI
MP(Anneal)
MP(DipoleInSphere)

// FDM
MP(BuildFDMatrix)
MP(BuildFDField)

// FEM
MP(DipoleMatToGeom)
MP(DipoleSourceRHS)
MP(ErrorMetric)
MP(FieldFromBasis)
MP(RecipBasis)
MP(RemapVector)
MP(VecSplit)

// ISL
MP(Downhill_Simplex)
MP(OptDip)
MP(SGI_LU)
MP(SGI_Solve)

// MEG
MP(MakeCurrentDensityField)
MP(MagneticScalarField)
MP(SurfToVectGeom)
MP(FieldCurl)

// MEG
MP(EditPath)

// Readers
MP(ContourSetReader)
MP(PathReader)
MP(SegFldReader)
MP(SigmaSetReader)
MP(TensorFieldReader)

// SiRe
MP(SiReAll)
MP(SiReCrunch)
MP(SiReInput)
MP(SiReOutput)

// Tensor
MP(Bundles)
MP(TensorAccessFields)
MP(TensorAnisotropy)

// Writers
MP(ContourSetWriter)
MP(PathWriter)
MP(SegFldWriter)
MP(SigmaSetWriter)
MP(TensorFieldWriter)

using namespace PSECore::Dataflow;
using namespace DaveW::Modules;

#define RM(a,b,c,d) packageDB.registerModule("DaveW",a,b,c,d)

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
  RM("EEG",	      "SegFldToSurfTree",	 make_SegFldToSurfTree,	    "");
  RM("EEG",	      "SelectSurfNodes",	 make_SelectSurfNodes,	    tcl+"/SelectSurfNodes.tcl");
  RM("EEG",	      "Taubin",			 make_Taubin,		    tcl+"/Taubin.tcl");
  RM("EEG",	      "Thermal",		 make_Thermal,		    tcl+"/Thermal.tcl");
  RM("EEG",	      "TopoSurfToGeom",	 	 make_TopoSurfToGeom,   tcl+"/TopoSurfToGeom.tcl");

  // EGI
  RM("EGI",	      "Anneal",		 	 make_Anneal,		    tcl+"/Anneal.tcl");
  RM("EGI",	      "DipoleInSphere",		 make_DipoleInSphere,	    tcl+"/DipoleInSphere.tcl");

  // FDM
  RM("FDM",	      "BuildFDField",		 make_BuildFDField,	    "");
  RM("FDM",	      "BuildFDMatrix",		 make_BuildFDMatrix,	    "");

  // FEM
  RM("FEM",	      "DipoleMatToGeom",	 make_DipoleMatToGeom,	    tcl+"/DipoleMatToGeom.tcl");
  RM("FEM",	      "DipoleSourceRHS",	 make_DipoleSourceRHS,	    "");
  RM("FEM",	      "ErrorMetric",		 make_ErrorMetric,	    tcl+"/ErrorMetric.tcl");
  RM("FEM",	      "FieldFromBasis",		 make_FieldFromBasis,	    "");
  RM("FEM",	      "RecipBasis",		 make_RecipBasis,	    "");
  RM("FEM",	      "RemapVector",		 make_RemapVector,	    tcl+"/RemapVector.tcl");
  RM("FEM",	      "VecSplit",		 make_VecSplit,	            "");
  
  // ISL
  RM("ISL",	      "Downhill_Simplex",	 make_Downhill_Simplex,	    tcl+"/Downhill_Simplex.tcl");
  RM("ISL",	      "OptDip",			 make_OptDip,	   	    tcl+"/OptDip.tcl");
  RM("ISL",	      "SGI_Solve",		 make_SGI_Solve,	    tcl+"/SGI_Solve.tcl");
  RM("ISL",	      "SGI_LU",			 make_SGI_LU,		    tcl+"/SGI_LU.tcl");

  // MEG
  RM("MEG",	      "MakeCurrentDensityField", make_MakeCurrentDensityField, "");
  RM("MEG",	      "MagneticScalarField",	 make_MagneticScalarField,  "");
  RM("MEG",	      "SurfToVectGeom",		 make_SurfToVectGeom,	    tcl+"/SurfToVectGeom.tcl");
  RM("MEG",	      "FieldCurl",		 make_FieldCurl,	    "");

  // Path
  RM("Path",	      "EditPath", 		make_EditPath, 		    "tcl+/EditPath.tcl");

  // Readers
  RM("Readers",	      "ContourSetReader",	 make_ContourSetReader,	    tcl+"/ContourSetReader.tcl");
  RM("Readers",	      "PathReader",	 	 make_PathReader,	    tcl+"/PathReader.tcl");
  RM("Readers",	      "SegFldReader",		 make_SegFldReader,	    tcl+"/SegFldReader.tcl");
  RM("Readers",	      "SigmaSetReader",		 make_SigmaSetReader,	    tcl+"/SigmaSetReader.tcl");
  RM("Readers",	      "TensorFieldReader",	 make_TensorFieldReader,    tcl+"/TensorFieldReader.tcl");

  // SiRe
  RM("SiRe",	      "SiReAll",		 make_SiReAll,	    	    tcl+"/SiReAll.tcl");
  RM("SiRe",	      "SiReCrunch",		 make_SiReCrunch,	    tcl+"/SiReCrunch.tcl");
  RM("SiRe",	      "SiReInput",		 make_SiReInput,	    tcl+"/SiReInput.tcl");
  RM("SiRe",	      "SiReOutput",		 make_SiReOutput,	    tcl+"/SiReOutput.tcl");

  // Tensor
  RM("Tensor",	      "Bundles",	 	 make_Bundles,   	    tcl+"/Bundles.tcl");
  RM("Tensor",	      "TensorAccessFields",	 make_TensorAccessFields,   "");
  RM("Tensor",	      "TensorAnisotropy",	 make_TensorAnisotropy,	    "");

  // Writers
  RM("Writers",	      "ContourSetWriter",	 make_ContourSetWriter,	    tcl+"/ContourSetWriter.tcl");
  RM("Writers",	      "PathWriter",	 	 make_PathWriter,	    tcl+"/PathWriter.tcl");
  RM("Writers",	      "SegFldWriter",		 make_SegFldWriter,	    tcl+"/SegFldWriter.tcl");
  RM("Writers",	      "SigmaSetWriter",		 make_SigmaSetWriter,	    tcl+"/SigmaSetWriter.tcl");
  RM("Writers",	      "TensorFieldWriter",	 make_TensorFieldWriter,    tcl+"/TensorFieldWriter.tcl");

  std::cerr << "Initfn done -- TCL path was " << tcl << "\n";
}
}
