/************************************** 
CLASS 
   RescaleColormap

   A module that can scale the colormap values to fit the data
   or express a fixed data range.

GENERAL INFORMATION 
   RescaleColormap.h
   Written by:
     Packages/Kurt Zimmerman
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

#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Network/Module.h>
#include <Packages/Uintah/Dataflow/Ports/ScalarParticlesPort.h>

namespace Uintah {
using namespace SCIRun;

class RescaleColorMapForParticles : public Module {
    ColorMapOPort* omap;

  ScalarParticlesIPort* iPort;
  ColorMapIPort* imap;
public:
  RescaleColorMapForParticles(GuiContext* ctx);
  virtual ~RescaleColorMapForParticles();
  virtual void execute();
protected:
  GuiDouble minVal;
  GuiDouble maxVal;
  GuiString scaleMode;
};
} // End namespace Uintah

