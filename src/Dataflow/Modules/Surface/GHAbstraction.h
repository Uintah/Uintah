#ifndef _GHABSTRACTION_H_
#define _GHABSTRACTION_H_

/*
 *  GHAbstraction.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <PSECommon/Dataflow/Module.h>
#include <SCICore/Datatypes/TriSurface.h>

class Model;

namespace PSECommon {
namespace Modules {

using SCICore::Datatypes::TriSurface;
using PSECommon::Dataflow::Module;

class GHAbstraction {
public:
  GHAbstraction(TriSurface* surf); // init with a surface...

  void Simplify(int nfaces); // reduces model to nfaces...
  void DumpSurface(TriSurface* surf); // dumps reduced surface...

  void RDumpSurface(); // does the real thing...

  void AddPoint(double x, double y, double z);
  void AddFace(int, int, int);

  void SAddPoint(double x, double y, double z);
  void SAddFace(int,int,int); // adds it to TriSurface
  
  void InitAdd(); // inits model...
  void FinishAdd();
  // data for this guy...

  Model *M0;
  TriSurface *orig; // original surface...
  TriSurface *work; // working surface...

  Module *owner;
};

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.3  1999/08/25 03:47:59  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:37:42  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:56  mcq
// Initial commit
//
// Revision 1.4  1999/05/06 20:17:12  dav
// added back PSECommon .h files
//
// Revision 1.2  1999/04/29 03:19:26  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//

#endif
