#ifndef GENSTANDARDCOLORMAPS_H
#define GENSTANDARDCOLORMAPS_H 1

#include <SCICore/TclInterface/TCLvar.h> 
#include <SCICore/Geom/Color.h>
#include <SCICore/Datatypes/ColorMap.h>

#include <PSECore/Dataflow/Module.h> 
#include <PSECore/Datatypes/ColorMapPort.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

/**************************************
CLASS
   GenStandardColorMaps

   A module that generates fixed Colormaps for visualation purposes.

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

class GenStandardColorMaps : public Module { 
  class StandardColorMap;
  
public: 
  

  // GROUP: Constructors: 
  ////////// 
  // Contructor taking 
  // [in] string as an identifier 
  GenStandardColorMaps(const clString& id); 

  // GROUP: Destructors: 
  ////////// 
  // Destructor
  virtual ~GenStandardColorMaps(); 

  ////////// 
  // execution scheduled by scheduler   
  virtual void execute(); 

  //////////
  // overides tcl_command in base class Module
  void tcl_command( TCLArgs&, void* );

private:

  TCLstring tcl_status; 
  TCLint mapType;
  TCLint minRes;
  TCLint resolution;
  ColorMapOPort  *outport;
  Array1<StandardColorMap *> mapTypes;


  // The following classes are nested.  They have no scope outside
  // of GenStandardColormaps.

  //----------------------------------------------------------------------
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
  }; // class StandardColorMap

  //----------------------------------------------------------------------
  class GrayColorMap : public StandardColorMap
  {
  public:
    GrayColorMap();
    virtual ~GrayColorMap(){}
  
  }; // class GrayColorMap

  //----------------------------------------------------------------------
  class InverseGrayColorMap : public StandardColorMap
  {
  public:
    InverseGrayColorMap();
    virtual ~InverseGrayColorMap(){}
  
  }; // class InverseGrayColorMap

  //----------------------------------------------------------------------
  class RainbowColorMap : public StandardColorMap
  {
  public:
    RainbowColorMap();
    virtual ~RainbowColorMap(){}
  
  }; // class RainbowColorMap

  //----------------------------------------------------------------------
  class InverseRainbow : public StandardColorMap
  {
  public:
    InverseRainbow();
    virtual ~InverseRainbow(){}
  }; // class InverseRainbow

  //----------------------------------------------------------------------
  class DarkHueColorMap : public StandardColorMap
  {
  public:
    DarkHueColorMap();
    virtual ~DarkHueColorMap(){}
  }; // class DarkHueColorMap

  //----------------------------------------------------------------------
  class LightHueColorMap : public StandardColorMap
  {
  public:
    LightHueColorMap();
    virtual ~LightHueColorMap(){}
  }; // class LightHueColorMap

  //----------------------------------------------------------------------
  class InverseDarkHue : public StandardColorMap
  {
  public:
    InverseDarkHue();
    virtual ~InverseDarkHue(){}
  }; // class InverseDarkHue

  //----------------------------------------------------------------------
  class BlackBodyColorMap : public StandardColorMap
  {
  public:
    BlackBodyColorMap();
    virtual ~BlackBodyColorMap(){}
  
  }; // class BlackBodyColorMap

  //----------------------------------------------------------------------
  class DonColorMap : public StandardColorMap
  {
  public:
    DonColorMap();
    virtual ~DonColorMap(){}
  
  }; // class DonColorMap

}; //class GenStandardColorMaps

} // end namespace PSECommon
} // end namespace Modules
#endif
