#include <PSECore/Dataflow/PackageDB.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>

#define MP(makesuf) \
namespace Phil { namespace Modules { using namespace PSECore::Dataflow;\
  Module* make_##makesuf (const clString& id); } }\

MP(Tbon)
MP(TbonCL)
MP(TbonUG)
MP(TbonP)
MP(TbonS)
MP(TbonGC)
MP(TbonOOC1)
MP(TbonOOC2)
MP(ViewGrid)
MP(ViewMesh)
MP(Bono)
MP(BonoP)
MP(BonoCL)
MP(ElapsedTime)
MP(GenScalarField)

using namespace PSECore::Dataflow;
using namespace Phil::Modules;

#define RM(a,b,c,d) packageDB.registerModule("Phil",a,b,c,d)

extern "C" {
void initPackage(const clString& tcl) {

  RM("Tbon", "Tbon", make_Tbon, tcl+"/Tbon.tcl");
  RM("Tbon", "TbonCL", make_TbonCL, tcl+"/TbonCL.tcl");
  RM("Tbon", "TbonUG", make_TbonUG, tcl+"/TbonUG.tcl");
  RM("Tbon", "TbonP", make_TbonP, tcl+"/TbonP.tcl");
  //  RM("Tbon", "GenScalarField", make_GenScalarField, tcl+"/GenScalarField.tcl");
//   RM("Tbon", "TbonS", make_TbonS, tcl+"/TbonS.tcl");
//   RM("Tbon", "TbonGC", make_TbonGC, tcl+"/TbonGC.tcl");
  RM("Tbon", "TbonOOC1", make_TbonOOC1, tcl+"/TbonOOC1.tcl");
  RM("Tbon", "TbonOOC2", make_TbonOOC2, tcl+"/TbonOOC2.tcl");
  RM("Tbon", "ViewGrid", make_ViewGrid, tcl+"/ViewGrid.tcl");
  RM("Tbon", "ViewMesh", make_ViewMesh, tcl+"/ViewMesh.tcl");
  RM("Tbon", "Bono", make_Bono, tcl+"/Bono.tcl");
  RM("Tbon", "BonoP", make_BonoP, tcl+"/BonoP.tcl");
  RM("Tbon", "BonoCL", make_BonoCL, tcl+"/BonoCL.tcl");
  RM("Tbon", "ElapsedTime", make_ElapsedTime, tcl+"/ElapsedTime.tcl");
}
}
