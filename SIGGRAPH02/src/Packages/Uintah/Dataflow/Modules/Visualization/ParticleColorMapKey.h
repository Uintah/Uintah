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

#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Packages/Uintah/Dataflow/Ports/ScalarParticlesPort.h>
#include <Dataflow/Network/Module.h>

namespace Uintah {
using namespace SCIRun;

class ParticleColorMapKey : public Module {
  ColorMapIPort *imap;
  ScalarParticlesIPort* iPort;
  GeometryOPort *ogeom;

  MaterialHandle white;
public:
  ParticleColorMapKey(GuiContext* ctx);
  virtual ~ParticleColorMapKey();
  virtual void execute();
};
} // End namespace Kurt

