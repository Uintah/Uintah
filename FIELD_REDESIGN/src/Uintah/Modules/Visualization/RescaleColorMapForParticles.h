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
#include <Uintah/Datatypes/ScalarParticlesPort.h>

namespace Uintah {
namespace Modules {

using PSECore::Datatypes::ScalarParticlesIPort;
using PSECore::Datatypes::ColorMapOPort;
using PSECore::Datatypes::ColorMapIPort;
using PSECore::Dataflow::Module;
using SCICore::Containers::clString;
using namespace SCICore::TclInterface;

class RescaleColorMapForParticles : public Module {
    ColorMapOPort* omap;

  ScalarParticlesIPort* iPort;
  ColorMapIPort* imap;
public:
  RescaleColorMapForParticles(const clString& id);
  virtual ~RescaleColorMapForParticles();
  virtual void execute();
protected:
  TCLdouble minVal;
  TCLdouble maxVal;
  TCLstring scaleMode;
};

} // End namespace Module
} // End namespace Kurt
