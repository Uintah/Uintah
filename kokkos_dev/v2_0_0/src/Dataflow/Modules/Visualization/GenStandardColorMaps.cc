/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
    tcl_status(ctx->subVar("tcl_status")),
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
   tcl_status.set("Calling GenStandardColorMaps!"); 

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
