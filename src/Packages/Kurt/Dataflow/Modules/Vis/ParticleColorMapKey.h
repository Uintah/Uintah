/*
 *  ParticleColorMapKey.cc: create a key for colormap
 *
 *  Written by:
 *   Packages/Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   June 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Packages/Kurt/Core/Datatypes/VisParticleSetPort.h>

namespace Kurt {
using namespace SCIRun;

class ParticleColorMapKey : public Module {
  ColorMapIPort *imap;
  VisParticleSetIPort* iPort;
  GeometryOPort *ogeom;

  MaterialHandle white;
public:
  ParticleColorMapKey(const clString &id);
  virtual ~ParticleColorMapKey();
  virtual void execute();
};
} // End namespace Kurt

