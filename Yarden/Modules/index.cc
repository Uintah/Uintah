#include <PSECore/Dataflow/PackageDB.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <iostream.h>

#define MP(makesuf) \
namespace Yarden { namespace Modules { using namespace PSECore::Dataflow;\
  Module* make_##makesuf (const clString& id); } }\

// BRDF
// MP(BldBRDF)

// EEG

// EGI

// FEM

// Readers

// Visualization
MP(Sage)
MP(Hase)

//MP(SageVFem)

// Writers

using namespace PSECore::Dataflow;
using namespace Yarden::Modules;

#define RM(a,b,c,d) packageDB.registerModule("Yarden",a,b,c,d)

extern "C" {
void initPackage(const clString& tcl) {

  // BRDF

  // EEG

  // EGI

  // FEM

  // Readers

  // Visulaization
  RM("Visualization",	  "Sage",	  make_Sage,   	    tcl+"/Sage.tcl");
  RM("Visualization",	  "Hase",	  make_Hase,   	    tcl+"/Hase.tcl");

  //  RM("Visualization",	      "SageVFem",	 make_SageVFem,   tcl+"/SageVFem.tcl");

  // Writers

  cerr << "Initfn done -- TCL path was " << tcl << "\n";
}
}












