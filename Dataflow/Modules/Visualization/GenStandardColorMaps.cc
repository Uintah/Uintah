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
genMap(ColorMapHandle &cmap, const string& s, int res)
{
  int r,g,b;
  double a;
  istringstream is(s);
  if( is.good() )
  {
    vector< Color > rgbs(res);
    vector< float > rgbT(res);
    vector< float > alphas(res);
    vector< float > alphaT(res);

    for (int i = 0; i < res; i++)
    {
      is >> r >> g >> b >> a;
      rgbs[i] = Color(r/255.0, g/255.0, b/255.0);
      rgbT[i] = i/float(res);
      alphas[i] = a;
      alphaT[i] = i/float(res);
    }
    if (!res || rgbT[res-1] != 1.0)
    {
      rgbT.push_back(1.0);
    }
    if (!res || alphaT[res-1] != 1.0)
    {
      alphaT.push_back(1.0);
    }
    cmap = scinew ColorMap(rgbs,rgbT,alphas,alphaT);
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
    mapType(ctx->subVar("mapType")),
    minRes(ctx->subVar("minRes")),
    resolution(ctx->subVar("resolution")),
    realres(ctx->subVar("realres")),
    gamma(ctx->subVar("gamma"))
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
   if ( genMap(cmap, tclRes, resolution.get()) ) 
   {
     ColorMapOPort *outport = (ColorMapOPort *)get_oport("ColorMap");
     if (!outport) {
       error("Unable to initialize oport 'ColorMap'.");
       return;
     }
     outport->send(cmap);
   }
} 
//--------------------------------------------------------------- 
} // End namespace SCIRun
