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
using namespace SCICore::Containers;

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
  // void tcl_command( TCLArgs&, void* );

private:

  TCLstring tcl_status; 
  TCLstring positionList;
  TCLstring nodeList;
  TCLint width;
  TCLint height;
  TCLint mapType;
  TCLint minRes;
  TCLint resolution;
  ColorMapOPort  *outport;
  ColorMapHandle cmap;
  Array1< Color > colors;
  void genMap(const clString& s);

}; //class GenStandardColorMaps

} // end namespace PSECommon
} // end namespace Modules
#endif
