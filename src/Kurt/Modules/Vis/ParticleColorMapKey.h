/*
 *  ParticleColorMapKey.cc: create a key for colormap
 *
 *  Written by:
 *   Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   June 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <Kurt/Datatypes/VisParticleSetPort.h>

namespace PSECommon {
namespace Kurt {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using PSECore::Datatypes::VisParticleSetIPort;
using PSECore::Datatypes::ColorMapOPort;
using PSECore::Datatypes::ColorMapIPort;
using PSECore::Datatypes::GeometryOPort;

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

} // End namespace Modules
} // End namespace Kurt
