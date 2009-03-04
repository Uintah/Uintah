/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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

#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Dataflow/Network/Module.h>
#include <Uintah/Dataflow/Ports/ScalarParticlesPort.h>

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

