//static char *id="@(#) $Id$";

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
     "standard" non-editable colormaps in SCIRun/Uintah.      
     Non-editable simply means that the colors cannot be      
     interactively manipulated.  The Module does, allow       
     for the the resolution of the colormaps to be changed.
     This class sets up the data structures for Colormaps and 
     creates a module from which the user can choose from several
     popular colormaps.

****************************************/   

#include <PSECommon/Modules/Visualization/GenStandardColorMaps.h> 
#include <PSECore/Datatypes/ColorMapPort.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Malloc/Allocator.h>

#include <math.h>
#include <iostream>
using std::ios;
#include <sstream>
using std::ostringstream;
using std::istringstream;
#include <iomanip>
using std::setw;
#include <stdio.h>

namespace PSECommon {
namespace Modules {

// ---------------------------------------------------------------------- // 
void
GenStandardColorMaps::genMap(const clString& s)
{
  int m = resolution.get();
  int r,g,b;
  double a;
  istringstream is( s() );
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
}



// ---------------------------------------------------------------------- // 
  

extern "C" Module* make_GenStandardColorMaps(const clString& id) { 
  return new GenStandardColorMaps(id); 
}

//--------------------------------------------------------------- 
GenStandardColorMaps::GenStandardColorMaps(const clString& id) 
  : Module("GenStandardColorMaps", id, Filter),
    mapType("mapType", id, this),
    minRes("minRes", id, this),
    resolution("resolution", id, this),
  tcl_status("tcl_status",id,this) ,
  positionList("positionList", id, this),
  nodeList("nodeList", id, this)
  

{ 
 // Create the output port
  outport = scinew ColorMapOPort(this,"ColorMap",ColorMapIPort::Atomic);
  add_oport(outport);
 // Initialization code goes here 
} 

//---------------------------------------------------------- 
GenStandardColorMaps::~GenStandardColorMaps(){} 

//-------------------------------------------------------------- 
void GenStandardColorMaps::tcl_command( TCLArgs& args, void* userdata)
{

  int i;
  if (args[1] == "setColors") {
    if (args.count() != 3) {
      args.error("GenStandardColorMaps::setColors wrong number of args.");
      return;
    }
    reset_vars();
    int m = resolution.get();
    if (m < minRes.get()) return;
    genMap( args[2] );
    
  } else {
    Module::tcl_command(args, userdata);
  }
}

//-------------------------------------------------------------- 

void GenStandardColorMaps::execute() 
{ 
   static int res = -1;
   tcl_status.set("Calling GenStandardColorMaps!"); 

   if ( res != resolution.get()){
     res = resolution.get();
   }
   
   if( !cmap.get_rep() ) return;

   cmap.detach();
   cmap->ResetScale();
     
   outport->send(cmap);
} 
//--------------------------------------------------------------- 
} // end namespace Modules
} // end namespace PSECommon
