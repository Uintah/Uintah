  

/**************************************************************** 
 *  GenStandardColormaps:  This module is used to create some   *
 *     "standard" non-editable colormaps in SCIRun/Uintah.      *
 *     Non-editable simply means that the colors cannot be      *
 *     interactively manipulated.  The Module does, allow       *
 *     for the the resolution of the colormaps to be changed.   *
 *                                                              * 
 *  Written by:                                                 * 
 *   Kurt Zimmerman                                             * 
 *   Department of Computer Science                             * 
 *   University of Utah                                         * 
 *   December 1998                                              * 
 *                                                              * 
 *  Copyright (C) 1998 SCI Group/Uintah                         *
 *                                                              * 
 *                                                              * 
 ****************************************************************/ 

#include <Classlib/NotFinished.h> 
#include <Dataflow/Module.h> 
#include <Datatypes/ColorMap.h>
#include <Datatypes/ColorMapPort.h>
#include <Geom/Material.h>
#include <Geom/Color.h>
#include <TCL/TCLvar.h> 
#include <Malloc/Allocator.h>
#include <math.h>
#include <iostream.h> 
#include <strstream.h>
#include <iomanip.h>
#include <stdio.h>


// The abstract class for non-editable Colormaps
class StandardColorMap
{
 public:
  
  StandardColorMap();
  virtual ~StandardColorMap(){}
  ColorMapHandle genMap(const int res);
  const Array1< Color >& getColors() {return colors;}
 protected:
    Array1< Color > colors;
};

StandardColorMap::StandardColorMap()
{
  // cerr<<"Error: instanciating the abstract class StandarColorMap\n";
}


ColorMapHandle StandardColorMap::genMap(const int res)
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

class GrayColorMap : public StandardColorMap
{
 public:
  GrayColorMap();
  virtual ~GrayColorMap(){}
  
};

GrayColorMap::GrayColorMap()
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
class InverseGrayColorMap : public StandardColorMap
{
 public:
  InverseGrayColorMap();
  virtual ~InverseGrayColorMap(){}
  
};
  
InverseGrayColorMap::InverseGrayColorMap()
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
class RainbowColorMap : public StandardColorMap
{
 public:
  RainbowColorMap();
  virtual ~RainbowColorMap(){}

  
};

RainbowColorMap::RainbowColorMap()
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
class DarkHueColorMap : public StandardColorMap
{
 public:
  DarkHueColorMap();
  virtual ~DarkHueColorMap(){}
};

DarkHueColorMap::DarkHueColorMap()
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
 class BlackBodyColorMap : public StandardColorMap
{
 public:
  BlackBodyColorMap();
  virtual ~BlackBodyColorMap(){}
  
};

BlackBodyColorMap::BlackBodyColorMap()
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
class GenStandardColorMaps : public Module { 
  
public: 
  
  TCLstring tcl_status; 
  GenStandardColorMaps(const clString& id); 
  GenStandardColorMaps(const GenStandardColorMaps&, int deep); 
  virtual ~GenStandardColorMaps(); 
  virtual Module* clone(int deep); 
  virtual void execute(); 
  void tcl_command( TCLArgs&, void* );
 private:

  TCLint mapType;
  TCLint minRes;
  TCLint resolution;
  ColorMapOPort  *outport;
  Array1<StandardColorMap *> mapTypes;
  
}; //class 
  

extern "C" { 
  
  Module* make_GenStandardColorMaps(const clString& id) 
  { 
    return new GenStandardColorMaps(id); 
  } 
  
}//extern 
  

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
  mapTypes.setsize(5);
  mapTypes[0] = (StandardColorMap *)(scinew GrayColorMap());
  mapTypes[1] = (StandardColorMap *)(scinew InverseGrayColorMap());
  mapTypes[2] = (StandardColorMap *)(scinew RainbowColorMap());
  mapTypes[3] = (StandardColorMap *)(scinew DarkHueColorMap());
  mapTypes[4] = (StandardColorMap *)(scinew BlackBodyColorMap());
    
} 

//---------------------------------------------------------- 
GenStandardColorMaps::GenStandardColorMaps(const GenStandardColorMaps& copy, int deep) 
  : Module(copy, deep), 
    mapType("mapType", id, this),
    minRes("minRes", id, this),
    resolution("resolution", id, this),
    tcl_status("tcl_status",id,this) 
{} 

//------------------------------------------------------------ 
GenStandardColorMaps::~GenStandardColorMaps(){} 

//------------------------------------------------------------- 
Module* GenStandardColorMaps::clone(int deep) 
{ 
  return new GenStandardColorMaps(*this, deep); 
} 

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

   outport->send(cmap);
  
} 
//--------------------------------------------------------------- 
