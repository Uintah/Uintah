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
bool
GenStandardColorMaps::genMap(const string& s)
{
  int m = resolution.get();
  int r,g,b;
  double a;
  istringstream is(s.c_str());
  // got to check that library function...
  if( is.good() ){
    Array1< Color > rgbs;
    Array1< float > rgbT;
    Array1< float > alphas;
    Array1< float > alphaT;
    rgbs.setsize(m);
    rgbT.setsize(m);
    alphas.setsize(m);
    alphaT.setsize(m);

    for(int i = 0; i < m; i++){
      is >> r >> g >> b >> a;
      rgbs[i] = Color(r/255.0, g/255.0, b/255.0);
      rgbT[i] = i/float(m-1);
      alphas[i] = a;
      alphaT[i] = i/float(m-1);
    }
  
    cmap = scinew ColorMap(rgbs,rgbT,alphas,alphaT,m);
    return true;
  } else {
    return false;
  }
}



// ---------------------------------------------------------------------- // 
  

extern "C" Module* make_GenStandardColorMaps(const string& id) { 
  return new GenStandardColorMaps(id); 
}

//--------------------------------------------------------------- 
GenStandardColorMaps::GenStandardColorMaps(const string& id) 
  : Module("GenStandardColorMaps", id, Filter, "Visualization", "SCIRun"),
    tcl_status("tcl_status",id,this),
    positionList("positionList", id, this),
    nodeList("nodeList", id, this),
    width("width", id, this),
    height("height", id, this),
    mapType("mapType", id, this),
    minRes("minRes", id, this),
    resolution("resolution", id, this),
    realres("realres", id, this),
    gamma("gamma", id, this)
{ 
} 

//---------------------------------------------------------- 
GenStandardColorMaps::~GenStandardColorMaps(){} 

//-------------------------------------------------------------- 

void GenStandardColorMaps::execute() 
{
   outport = (ColorMapOPort *)get_oport("ColorMap");
   if (!outport) {
     postMessage("Unable to initialize "+name+"'s oport\n");
     return;
   }
   static int res = -1;
   tcl_status.set("Calling GenStandardColorMaps!"); 

   if ( res != resolution.get()){
     res = resolution.get();
   }
   
   string tclRes;
   tcl_eval(id+" getColorMapString", tclRes);
   if ( genMap(tclRes) ) 
     outport->send(cmap);
} 
//--------------------------------------------------------------- 
} // End namespace SCIRun
