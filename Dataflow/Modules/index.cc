#include <PSECore/Dataflow/PackageDB.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <iostream.h>

#define MP(makesuf) \
namespace PSECommon { namespace Modules { using namespace PSECore::Dataflow;\
  Module* make_##makesuf (const clString& id); } }\

// FEM
MP(ApplyBC)
MP(BuildFEMatrix)
MP(ComposeError)
MP(ErrorInterval)
MP(FEMError)
MP(MeshRefiner)

// Fields
MP(ClipField)
MP(FieldFilter)
MP(FieldGainCorrect)
MP(FieldMedianFilter)
MP(FieldSeed)
MP(Gradient)
MP(GradientMagnitude)
MP(TracePath)
MP(TrainSeg2)
MP(TrainSegment)
MP(TransformField)

// Iterators
MP(MeshIterator)

// Matrix
MP(BldTransform)
MP(EditMatrix)
MP(SolveMatrix)
MP(VisualizeMatrix)
MP(cConjGrad)
MP(cPhase)

// Readers
MP(ColorMapReader)
MP(ColumnMatrixReader)
MP(GeomReader)
MP(GeometryReader)
MP(ImageReader)
MP(MatrixReader)
MP(MeshReader)
MP(PointsReader)
MP(ScalarFieldReader)
MP(SurfaceReader)
MP(VectorFieldReader)

// Salmon
MP(Salmon)

// Surface
MP(GenSurface)
MP(LabelSurface)
MP(SFUGtoSurf)
MP(SurfGen)
MP(SurfNewVals)

// Visualization
MP(AddWells2)
MP(BitVisualize)
MP(BoxClipSField)
MP(ColorMapKey)
MP(CuttingPlane)
MP(CuttingPlaneTex)
MP(FieldCage)
MP(GenAxes)
MP(GenColorMap)
MP(GenTransferFunc)
MP(GenFieldEdges)
MP(GenStandardColorMaps)
MP(Hedgehog)
MP(ImageViewer)
MP(IsoMask)
MP(IsoSurface)
MP(IsoSurfaceDW)
MP(IsoSurfaceMSRG)
MP(RescaleColorMap)
MP(SimpVolVis)
MP(Streamline)
MP(VectorSeg)
MP(VolRendTexSlices)
MP(VolVis)
MP(WidgetTest)

// Writers

MP(ColorMapWriter)
MP(ColumnMatrixWriter)
MP(GeometryWriter)
MP(MatrixWriter)
MP(MeshWriter)
MP(ScalarFieldWriter)
MP(SurfaceWriter)
MP(TetraWriter)
MP(VectorFieldWriter)

using namespace PSECore::Dataflow;
using namespace PSECommon::Modules;

#define RM(a,b,c,d) packageDB.registerModule("PSE Common",a,b,c,d)

extern "C" {
void initPackage(const clString& tcl) {

  // FEM
  RM("FEM",           "Apply BC",                make_ApplyBC,              "");
  RM("FEM",           "Build FE Matrix",         make_BuildFEMatrix,        tcl+"/BuildFEMatrix.tcl");
  RM("FEM",           "Compose Error",           make_ComposeError,         "");
  RM("FEM",           "Error Interval",          make_ErrorInterval,        tcl+"/ErrorInterval.tcl");
  RM("FEM",           "FEM Error",               make_FEMError,             "");
  RM("FEM",           "Mesh Refiner",            make_MeshRefiner,          "");

  // Fields
  RM("Fields",        "Clip Field",              make_ClipField,            tcl+"/ClipField.tcl");
  RM("Fields",        "Field Filter",            make_FieldFilter,          tcl+"/FieldFilter.tcl");
  RM("Fields",        "Field Gain Correct",      make_FieldGainCorrect,     tcl+"/FieldGainCorrect.tcl");
  RM("Fields",        "Field Median Filter",     make_FieldMedianFilter,    tcl+"/FieldMedianFilter.tcl");
  RM("Fields",        "Field Seed",              make_FieldSeed,            tcl+"/FieldSeed.tcl");
  RM("Fields",        "Gradient",                make_Gradient,             "");
  RM("Fields",        "Gradient Magnitude",      make_GradientMagnitude,    "");
  RM("Fields",        "Trace Path",              make_TracePath,            tcl+"/TracePath.tcl");
  RM("Fields",        "Train Seg 2",             make_TrainSeg2,            tcl+"/TrainSeg2.tcl");
  RM("Fields",        "Train Segment",           make_TrainSegment,         tcl+"/TrainSegment.tcl");
  RM("Fields",        "Transform Field",         make_TransformField,       tcl+"/TransformField.tcl");

  // Iterators
  RM("Iterators",     "Mesh Iterator",           make_TransformField,       "");

  // Matrix
  RM("Matrix",        "BldTransform",            make_BldTransform,         tcl+"/BldTransform.tcl");
  RM("Matrix",        "Edit Matrix",             make_EditMatrix,           tcl+"/EditMatrix.tcl");
  RM("Matrix",        "Solve Matrix",            make_SolveMatrix,          "");
  RM("Matrix",        "Visualize Matrix",        make_VisualizeMatrix,      "");
  RM("Matrix",        "cConjGrad",               make_cConjGrad,            tcl+"/cConjGrad.tcl");
  RM("Matrix",        "cPhase",                  make_cPhase,               tcl+"/cPhase.tcl");

  // Readers
  RM("Readers",       "ColorMap Reader",         make_ColorMapReader,       tcl+"/ColorMapReader.tcl");
  RM("Readers",       "Column Matrix Reader",    make_ColumnMatrixReader,   tcl+"/ColumnMatrixReader.tcl");
  RM("Readers",       "Geom Reader",             make_GeomReader,           tcl+"/GeomReader.tcl");
  RM("Readers",       "Geometry Reader",         make_GeometryReader,       tcl+"/GeometryReader.tcl");
  RM("Readers",       "Image Reader",            make_ImageReader,          tcl+"/ImageReader.tcl");
  RM("Readers",       "Matrix Reader",           make_MatrixReader,         tcl+"/MatrixReader.tcl");
  RM("Readers",       "Mesh Reader",             make_MeshReader,           tcl+"/MeshReader.tcl");
  RM("Readers",       "Points Reader",           make_PointsReader,         tcl+"/PointsReader.tcl");
  RM("Readers",       "Scalar Field Reader",     make_ScalarFieldReader,    tcl+"/ScalarFieldReader.tcl");
  RM("Readers",       "Surface Reader",          make_SurfaceReader,        tcl+"/SurfaceReader.tcl");
  RM("Readers",       "Vector Field Reader",     make_VectorFieldReader,    tcl+"/VectorFieldReader.tcl");
 
  // Salmon
  RM("Salmon",        "Salmon",                  make_Salmon,               tcl+"/Salmon.tcl");

  // Surface
  RM("Surface",       "Gen Surface",             make_GenSurface,           tcl+"/GenSurface.tcl");
  RM("Surface",       "Label Surface",           make_LabelSurface,         "");
  RM("Surface",       "SFUG to Surf",            make_SFUGtoSurf,           "");
  RM("Surface",       "Surf Gen",                make_SurfGen,              tcl+"/SurfGen.tcl");
  RM("Surface",       "Surf New Vals",           make_SurfNewVals,          tcl+"/SurfNewVals.tcl");

  // Visualization
  RM("Visualization", "Add Wells 2",             make_AddWells2,            tcl+"/AddWells2.tcl");
  RM("Visualization", "Bit Visualize",           make_BitVisualize,         tcl+"/BitVisualize.tcl");
  RM("Visualization", "Box Clip S Field",        make_BoxClipSField,        tcl+"/BoxClipSField.tcl");
  RM("Visualization", "Color Map Key",           make_ColorMapKey,          "");
  RM("Visualization", "Cutting Plane",           make_CuttingPlane,         tcl+"/CuttingPlane.tcl");
  RM("Visualization", "Cutting Plane Tex",       make_CuttingPlaneTex,      tcl+"/CuttingPlaneTex.tcl");
  RM("Visualization", "FieldCage",               make_FieldCage,            tcl+"/FieldCage.tcl");
  RM("Visualization", "Gen Axes",                make_GenAxes,              tcl+"/GenAxes.tcl");
  RM("Visualization", "Gen Color Map",           make_GenColorMap,          tcl+"/GenColorMap.tcl");
  RM("Visualization", "Gen Transfer Func",       make_GenTransferFunc,      tcl+"/GenTransferFunc.tcl");
  RM("Visualization", "Gen Field Edges",         make_GenFieldEdges,        "");
  RM("Visualization", "Gen Standard Color Maps", make_GenStandardColorMaps, tcl+"/GenStandardColorMaps.tcl");
  RM("Visualization", "Hedgehog",                make_Hedgehog,             tcl+"/Hedgehog.tcl");
  RM("Visualization", "Image Viewer",            make_ImageViewer,          "");
  RM("Visualization", "Iso Mask",                make_IsoMask,              tcl+"/IsoMask.tcl");
  RM("Visualization", "Iso Surface",             make_IsoSurface,           tcl+"/IsoSurface.tcl");
  RM("Visualization", "Iso Surface DW",          make_IsoSurfaceDW,         tcl+"/IsoSurfaceDW.tcl");
  RM("Visualization", "Iso Surface MSRG",        make_IsoSurfaceMSRG,       tcl+"/IsoSurfaceMSRG.tcl");
  RM("Visualization", "Rescale Color Map",       make_RescaleColorMap,      tcl+"/RescaleColorMap.tcl");
  RM("Visualization", "Simp Vol vis",            make_SimpVolVis,           tcl+"/SimpVolVis.tcl");
  RM("Visualization", "Streamline",              make_Streamline,           tcl+"/Streamline.tcl");
  RM("Visualization", "Vector Seg",              make_VectorSeg,            tcl+"/VectorSeg.tcl");
  RM("Visualization", "Vol Rend Tex Slices",     make_VolRendTexSlices,     tcl+"/VolRendTexSlices.tcl");
  RM("Visualization", "Vol Vis",                 make_VolVis,               tcl+"/VolVis.tcl");
  RM("Visualization", "Widget Test",             make_WidgetTest,           tcl+"/WidgetTest.tcl");

  // Writers
  RM("Writers",       "Color Map Writer",        make_ColorMapWriter,       tcl+"/ColorMapWriter.tcl");
  RM("Writers",       "Column Matrix Writer",    make_ColumnMatrixWriter,   tcl+"/ColumnMatrixWriter.tcl");
  RM("Writers",       "Geometry Writer",         make_GeometryWriter,       tcl+"/GeometryWriter.tcl");
  RM("Writers",       "Matrix Writer",           make_MatrixWriter,         tcl+"/MatrixWriter.tcl");
  RM("Writers",       "Mesh Writer",             make_MeshWriter,           tcl+"/MeshWriter.tcl");
  RM("Writers",       "Scalar Field Writer",     make_ScalarFieldWriter,    tcl+"/ScalarFieldWriter.tcl");
  RM("Writers",       "Surface Writer",          make_SurfaceWriter,        tcl+"/SurfaceWriter.tcl");
  RM("Writers",       "Tetra Writer",            make_TetraWriter,          tcl+"/TetraWriter.tcl");
  RM("Writers",       "Vector Field Writer",     make_VectorFieldWriter,    tcl+"/VectorFieldWriter.tcl");

  cerr << "Initfn done -- TCL path was " << tcl << "\n";
}
}
