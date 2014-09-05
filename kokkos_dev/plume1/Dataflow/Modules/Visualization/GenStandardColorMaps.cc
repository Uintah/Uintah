/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
   GenStandardColorMaps

   A module that generates fixed Colormaps for visualization purposes.

GENERAL INFORMATION 
   GenStandarColorMaps.h
   Written by: 

     Kurt Zimmerman
     Department of Computer Science 
     University of Utah 
     December 1998

     Copyright (C) 1998 SCI Group

KEYWORDS 
   Colormap, Transfer Function

DESCRIPTION 
     This module is used to create some   
     "standard" non-editable colormaps in Dataflow/Uintah.      
     Non-editable simply means that the colors cannot be      
     interactively manipulated.  The Module does, allow       
     for the the resolution of the colormaps to be changed.
     This class sets up the data structures for Colormaps and 
     creates a module from which the user can choose from several
     popular colormaps.

****************************************/   

#include <Dataflow/Modules/Visualization/GenStandardColorMaps.h> 
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Geom/Material.h>
#include <Core/Malloc/Allocator.h>

#include <math.h>
#include <iostream>
using std::ios;
#include <sstream>
using std::ostringstream;
using std::istringstream;
#include <iomanip>
using std::setw;
#include <stdio.h>

namespace SCIRun {


// ---------------------------------------------------------------------- // 
static bool
genMap(ColorMapHandle &cmap, const string& s, int res, double gamma, bool faux)
{
  unsigned int i;
  istringstream is(s);
  if( is.good() )
  {
    // Slurp in the rgb values.
    unsigned int rgbsize;
    is >> rgbsize;
    vector< Color > rgbs(rgbsize);
    vector< float > rgbT(rgbsize);
    for (i = 0; i < rgbsize; i++)
    {
      int r, g, b;
      is >> r >> g >> b;
      rgbs[i] = Color(r/255.0, g/255.0, b/255.0);
      rgbT[i] = i/float(rgbsize-1);
    }

    // Shift RGB
    const double bp = 1.0 / tan( M_PI_2 * (0.5 + gamma * 0.5));
    for (i=0; i < rgbT.size(); i++)
    {
      rgbT[i] = pow((double)rgbT[i], bp);
    }
    

    // Read in the alpha values.  This is the screen space curve.
    unsigned int asize;
    float w, h;
    is >> asize;
    is >> w >> h;
    vector< float > alphas(asize);
    vector< float > alphaT(asize);

    for (i = 0; i < asize; i++)
    {
      int x, y;
      is >> x >> y;
      alphaT[i] = x / w;
      alphas[i] = 1.0 - y / h;
    }
    
    // The screen space curve may not contain the default endpoints.
    // Add them in here if needed.
    if (alphaT.empty() || alphaT.front() != 0.0)
    {
      alphas.insert(alphas.begin(), 0.5);
      alphaT.insert(alphaT.begin(), 0.0);
    }
    if (alphaT.back() != 1.0)
    {
      alphas.push_back(0.5);
      alphaT.push_back(1.0);
    }
    
    cmap = scinew ColorMap(rgbs, rgbT, alphas, alphaT, res);
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------- // 
  

DECLARE_MAKER(GenStandardColorMaps)
//--------------------------------------------------------------- 
GenStandardColorMaps::GenStandardColorMaps(GuiContext* ctx) 
  : Module("GenStandardColorMaps", ctx, Filter, "Visualization", "SCIRun"),
    positionList(ctx->subVar("positionList")),
    nodeList(ctx->subVar("nodeList")),
    width(ctx->subVar("width")),
    height(ctx->subVar("height")),
    mapName(ctx->subVar("mapName")),
    reverse(ctx->subVar("reverse")),
    minRes(ctx->subVar("minRes")),
    resolution(ctx->subVar("resolution")),
    realres(ctx->subVar("realres")),
    gamma(ctx->subVar("gamma")),
    faux(ctx->subVar("faux"))
{ 
} 


//---------------------------------------------------------- 
GenStandardColorMaps::~GenStandardColorMaps()
{
} 


//-------------------------------------------------------------- 

void GenStandardColorMaps::execute() 
{
   string tclRes;
   gui->eval(id+" getColorMapString", tclRes);

   ColorMapHandle cmap;
   if ( genMap(cmap, tclRes, resolution.get(), gamma.get(), faux.get()) ) 
   {
     ColorMapOPort *outport = (ColorMapOPort *)get_oport("ColorMap");
     outport->send(cmap);
   }
} 
//--------------------------------------------------------------- 
} // End namespace SCIRun
