#include <PSECore/Dataflow/PackageDB.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>

#define MP(makesuf) \
namespace Uintah { namespace Modules { using namespace PSECore::Dataflow;\
  Module* make_##makesuf (const clString& id); } }\

// MPMViz
MP(TecplotFileSelector)
MP(ParticleGridVisControl)
MP(PartToGeom)
MP(RescaleParticleColorMap)
MP(cfdGridLines)
MP(VizControl)

// Readers
MP(ParticleSetReader)
MP(TriangleReader)
MP(MPReader)
// Writeres
MP(MPWriter)

using namespace PSECore::Dataflow;
using namespace Uintah::Modules;

#define RM(a,b,c,d) packageDB.registerModule("Uintah",a,b,c,d)

extern "C" {
void initPackage(const clString& tcl) {

  // FEM
  RM("MPMViz", "Tecplot File Selector", make_TecplotFileSelector, tcl+"/TecplotFileSelector.tcl");
  RM("MPMViz", "Particle Grid Vis Control", make_ParticleGridVisControl, tcl+"/ParticleGridVisControl.tcl");
  RM("MPMViz", "Part To Geom", make_PartToGeom, tcl+"/PartToGeom.tcl");
  RM("MPMViz", "Rescale Particle Color Map", make_RescaleParticleColorMap, tcl+"/RescaleParticleColorMap.tcl");
  RM("MPMViz", "cfd Grid Lines", make_cfdGridLines, tcl+"/cfdGridLines.tcl");
  RM("MPMViz", "Viz Control", make_VizControl, tcl+"/VizControl.tcl");
  // Readers
  RM("Readers", "Particle Set Reader", make_ParticleSetReader, "");
  RM("Readers", "MP Reader", make_MPReader, tcl+"/MPReader.tcl");
  RM("Readers", "Triangle Reader", make_TriangleReader, tcl+"/TriangleReader.tcl");
  // Writers
  RM("Writers", "MP Writer", make_MPWriter, tcl+"/MPWriter.tcl");
}
}
