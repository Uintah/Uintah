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

#include <Dataflow/Dataflow/Module.h>
#include <Core/Datatypes/TriSurface.h>

class Model;

namespace SCIRun {


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

} // End namespace SCIRun


#endif
