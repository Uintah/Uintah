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
#include <PSECore/CommonDatatypes/ColorMapPort.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Malloc/Allocator.h>

#include <math.h>
#include <iostream.h> 
#include <strstream.h>
#include <iomanip.h>
#include <stdio.h>

namespace PSECommon {
namespace Modules {

GenStandardColorMaps::StandardColorMap::StandardColorMap()
{
  // cerr<<"Error: instanciating the abstract class StandarColorMap\n";
}

// ---------------------------------------------------------------------- // 
ColorMapHandle 
GenStandardColorMaps::StandardColorMap::genMap(const int res)
{
  int n = colors.size();
  int m = res;

  Array1< Color > rgbs;
  Array1< float > rgbT;
  Array1< float > alphas;
  Array1< float > alphaT;
  rgbs.setsize(m);
  rgbT.setsize(m);
  alphas.setsize(m);
  alphaT.setsize(m);

  rgbs[0] = colors[0];
  rgbT[0] = 0;
  alphas[0] = 1.0;
  alphaT[0] = 0;

  float frac = (n-1)/float(m-1);

  for(int i = 1; i < m-1; i++){
    double index = 0;
    double t = modf(i*frac, &index);
    rgbs[i] =
      Color(colors[index][0] + t*(colors[index+1][0] - colors[index][0]),
	    colors[index][1] + t*(colors[index+1][1] - colors[index][1]),
	    colors[index][2] + t*(colors[index+1][2] - colors[index][2]));
    rgbT[i] = i/float(m-1);
    alphas[i] = 1.0;
    alphaT[i] = i/float(m-1);
  }
  
  rgbs[m-1] = colors[n-1];
  rgbT[m-1] = 1.0;
  alphas[m-1] = 1.0;
  alphaT[m-1] = 1.0;

  return scinew ColorMap(rgbs,rgbT,alphas,alphaT,m);
}

// ---------------------------------------------------------------------- // 
GenStandardColorMaps::GrayColorMap::GrayColorMap()
{
  int ncolors =  2;
  colors.setsize(ncolors);
  for(int i = 0; i < ncolors; i++){
        colors[i].r(i);
	colors[i].g(i);
	colors[i].b(i);
  }
} 
// ---------------------------------------------------------------------- // 
  
GenStandardColorMaps::InverseGrayColorMap::InverseGrayColorMap()
{
  int ncolors =   2;
  int i,j;
  colors.setsize(ncolors);
  for(i = 0, j = ncolors-1; i < ncolors; i++, j--){
        colors[i].r(j);
	colors[i].g(j);
	colors[i].b(j);
  }
}
// ---------------------------------------------------------------------- // 

GenStandardColorMaps::RainbowColorMap::RainbowColorMap()
{
  int cols[][3] =  {{ 255, 0, 0},
		   { 255, 102, 0},
		   { 255, 204, 0},
		   { 255, 234, 0},
		   { 204, 255, 0},
		   { 102, 255, 0},
		   { 0, 255, 0},
		   { 0, 255, 102},
		   { 0, 255, 204},
		   { 0, 204, 255},
		   { 0, 102, 255},
		   { 0, 0, 255}};
  int ncolors =  12;
  colors.setsize(ncolors);
  for(int i = 0; i < ncolors; i++){
    colors[i].r(cols[i][0]/255.0);
    colors[i].g(cols[i][1]/255.0);
    colors[i].b(cols[i][2]/255.0);
  }
}
// ---------------------------------------------------------------------- // 
GenStandardColorMaps::InverseRainbow::InverseRainbow()
{
  int cols[][3] =  {{ 0, 0, 255},
		   { 0, 102, 255},
		   { 0, 204, 255},
		   { 0, 255, 204},
		   { 0, 255, 102},
		   { 0, 255, 0},
		   { 102, 255, 0},
		   { 204, 255, 0},
		   { 255, 234, 0},
		   { 255, 204, 0},
		   { 255, 102, 0},
		   { 255, 0, 0} };
  int ncolors =  12;
  colors.setsize(ncolors);
  for(int i = 0; i < ncolors; i++){
    colors[i].r(cols[i][0]/255.0);
    colors[i].g(cols[i][1]/255.0);
    colors[i].b(cols[i][2]/255.0);
  }
}


// ---------------------------------------------------------------------- // 
GenStandardColorMaps::DarkHueColorMap::DarkHueColorMap()
{
  int cols[][3] =  {{ 0,  0,  0 },
		   { 0, 28, 39 },
		   { 0, 30, 55 },
		   { 0, 15, 74 },
		   { 1,  0, 76 },
		   { 28,  0, 84 },
		   { 32,  0, 85 },
		   { 57,  1, 92 },
		   { 108,  0, 114 },
		   { 135,  0, 105 },
		   { 158,  1, 72 },
		   { 177,  1, 39 },
		   { 220,  10, 10 },
		   { 229, 30,  1 },
		   { 246, 72,  1 },
		   { 255, 175, 36 },
		   { 255, 231, 68 },
		   { 251, 255, 121 },
		   { 239, 253, 174 }};
  int ncolors = 19;
  colors.setsize(ncolors);
  for(int i = 0; i < ncolors; i++){
    colors[i].r(cols[i][0]/255.0);
    colors[i].g(cols[i][1]/255.0);
    colors[i].b(cols[i][2]/255.0);
  }

}

// ---------------------------------------------------------------------- // 
GenStandardColorMaps::LightHueColorMap::LightHueColorMap()
{
  int cols[][3] =  {{ 64,  64,  64 },
		   { 64, 80, 84 },
		   { 64, 79, 92 },
		   { 64, 72, 111 },
		   { 64,  64, 102 },
		   { 80, 64, 108 },
		   { 80, 64, 108 },
		   { 92,  64, 110 },
		   { 118,  64, 121 },
		   { 131,  64, 116 },
		   { 133,  64, 100 },
		   { 152,  64, 84 },
		   { 174,  69, 69 },
		   { 179, 79,  64 },
		   { 189, 100,  64 },
		   { 192, 152, 82 },
		   { 192, 179, 98 },
		   { 189, 192, 124 },
		   { 184, 191, 151 }};
  int ncolors = 19;
  colors.setsize(ncolors);
  for(int i = 0; i < ncolors; i++){
    colors[i].r(cols[i][0]/255.0);
    colors[i].g(cols[i][1]/255.0);
    colors[i].b(cols[i][2]/255.0);
  }

}

// ---------------------------------------------------------------------- // 
GenStandardColorMaps::InverseDarkHue::InverseDarkHue()
{
  int cols[][3] =  {{ 239, 253, 174 },
		   { 251, 255, 121 },
		   { 255, 231, 68 },
		   { 255, 175, 36 },
		   { 246, 72,  1 },
		   { 229, 30,  1 },
		   { 220,  10, 10 },
		   { 177,  1, 39 },
		   { 158,  1, 72 },
		   { 135,  0, 105 },
		   { 108,  0, 114 },
		   { 57,  1, 92 },
		   { 32,  0, 85 },
		   { 28,  0, 84 },
		   { 1,  0, 76 },
		   { 0, 15, 74 },
		   { 0, 30, 55 },
		   { 0, 28, 39 },
		   { 0,  0,  0 } };
  int ncolors = 19;
  colors.setsize(ncolors);
  for(int i = 0; i < ncolors; i++){
    colors[i].r(cols[i][0]/255.0);
    colors[i].g(cols[i][1]/255.0);
    colors[i].b(cols[i][2]/255.0);
  }

}

// ---------------------------------------------------------------------- // 

GenStandardColorMaps::BlackBodyColorMap::BlackBodyColorMap()
{
  int  cols[][3] = { {0, 0, 0},
		    {52, 0, 0},
		    {102, 2, 0},
		    {153, 18, 0},
		    {200, 41, 0},
		    {230, 71, 0},
		    {255, 120, 0},
		    {255, 163, 20},
		    {255, 204, 55},
		    {255, 228, 80},
		    {255, 247, 120},
		    {255, 255, 180},
		    {255, 255, 255}};

  int ncolors = 13;
  colors.setsize(ncolors);
  for(int i = 0; i < ncolors; i++){
    colors[i].r(cols[i][0]/255.0);
    colors[i].g(cols[i][1]/255.0);
    colors[i].b(cols[i][2]/255.0);
  }

}

// ---------------------------------------------------------------------- // 
  

Module* make_GenStandardColorMaps(const clString& id) { 
  return new GenStandardColorMaps(id); 
}

//--------------------------------------------------------------- 
GenStandardColorMaps::GenStandardColorMaps(const clString& id) 
  : Module("GenStandardColorMaps", id, Filter),
    mapType("mapType", id, this),
    minRes("minRes", id, this),
    resolution("resolution", id, this),
    tcl_status("tcl_status",id,this) 
  

{ 
 // Create the output port
  outport = scinew ColorMapOPort(this,"ColorMap",ColorMapIPort::Atomic);
  add_oport(outport);
 // Initialization code goes here 
  resolution.set(2);
  minRes.set(2);
  mapType.set(0);
  mapTypes.setsize(8);
  mapTypes[0] = (StandardColorMap *)(scinew GrayColorMap());
  mapTypes[1] = (StandardColorMap *)(scinew InverseGrayColorMap());
  mapTypes[2] = (StandardColorMap *)(scinew RainbowColorMap());
  mapTypes[3] = (StandardColorMap *)(scinew InverseRainbow());
  mapTypes[4] = (StandardColorMap *)(scinew DarkHueColorMap());
  mapTypes[5] = (StandardColorMap *)(scinew InverseDarkHue());
  mapTypes[6] = (StandardColorMap *)(scinew BlackBodyColorMap());
  mapTypes[7] = (StandardColorMap *)(scinew LightHueColorMap());    
} 

//---------------------------------------------------------- 
GenStandardColorMaps::~GenStandardColorMaps(){} 

//-------------------------------------------------------------- 
void GenStandardColorMaps::tcl_command( TCLArgs& args, void* userdata)
{

  int i;
  if (args[1] == "getcolors") {
    if (args.count() != 2) {
      args.error("GenStandardColorMaps::getcolors wrong number of args.");
      return;
    }

    reset_vars();
    const Array1< Color >& colors = mapTypes[mapType.get()]->getColors();
    int n = colors.size();
    int m = resolution.get();
    
    if (m < minRes.get()) return;
    
    ostrstream colorOstr;
    colorOstr.setf(ios::hex, ios::basefield);
    colorOstr.fill('0');
    float frac = (n-1)/float(m-1);
    double index = 0;
    double t;
    Color color;
    
    for(i = 0; i < m; i ++){
      if(i == 0) {
	color = colors[0];
      } else if ( i == m-1 ) {
	color = colors[n - 1];
      } else {
	t = modf(i*frac, &index);
	color =
	  Color(colors[index][0] + t*(colors[index+1][0] - colors[index][0]),
		colors[index][1] + t*(colors[index+1][1] - colors[index][1]),
		colors[index][2] + t*(colors[index+1][2] - colors[index][2]));
      }
      
      colorOstr << "#"<< setw(2) << int(color.r()*255) << setw(2) <<
	int(color.g()*255) << setw(2) << int(color.b()*255) << " ";
    }
    args.result(colorOstr.str());
    
  } else {
    Module::tcl_command(args, userdata);
  }
}

//-------------------------------------------------------------- 

void GenStandardColorMaps::execute() 
{ 
   static int mt = -1;
   static int res = -1;
   static ColorMapHandle cmap = 0;
   tcl_status.set("Calling GenStandardColorMaps!"); 

   if (mt != mapType.get() || res != resolution.get()){
     mt = mapType.get();
     res = resolution.get();
     cmap = mapTypes[mt]->genMap(res);
   }

   cmap.detach();
   cmap->ResetScale();
     
   outport->send(cmap);
} 
//--------------------------------------------------------------- 
} // end namespace Modules
} // end namespace PSECommon
