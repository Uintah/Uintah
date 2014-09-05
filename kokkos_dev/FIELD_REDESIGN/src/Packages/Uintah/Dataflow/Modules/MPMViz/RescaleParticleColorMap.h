/************************************** 
CLASS 
   RescaleColormap

   A module that can scale the colormap values to fit the data
   or express a fixed data range.

GENERAL INFORMATION 
   RescaleColormap.h
   Written by:
     Kurt Zimmerman
     Department of Computer Science
     University of Utah
     June 1999

   Copyright (C) 1999 SCI Group

KEYWORDS 
   Colormap, Transfer Function

DESCRIPTION 
   This module takes a color map and some data or vector field
   and scales the map to fit that field.

****************************************/ 

#include <SCICore/TclInterface/TCLvar.h>

#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Dataflow/Module.h>

#include <Uintah/Datatypes/Particles/ParticleSet.h>
#include <Uintah/Datatypes/Particles/ParticleSetPort.h>

namespace Uintah {
namespace Modules {


using namespace Uintah::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;


class RescaleParticleColorMap : public Module {
    ColorMapOPort* omap;

  ParticleSetIPort* iPort;
  ColorMapIPort* imap;
public:
  RescaleParticleColorMap(const clString& id);
  virtual ~RescaleParticleColorMap();
  virtual void execute();
protected:
  TCLdouble minVal;
  TCLdouble maxVal;
  TCLstring scaleMode;
};

} // End namespace Module
} // End namespace Uintah
