#ifndef GENSTANDARDCOLORMAPS_H
#define GENSTANDARDCOLORMAPS_H 1

#include <Core/GuiInterface/GuiVar.h> 
#include <Core/Geom/Color.h>
#include <Dataflow/Network/Module.h> 
#include <Dataflow/Ports/ColorMapPort.h>
// Note that the inclusion of Dataflow/Ports/ColorMapPort.h breaks
// the coding convention. Normally, since we only have a ColorMapPort*,
// we would not include the header but have instead a class Prototype 
// "class ColorMapOPort". But ColorMapOPort is a typedef not a class.  
// We would have to re-create the prototype here, which is uglier than 
// just including the header file.

namespace SCIRun {


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
     "standard" non-editable colormaps in Dataflow/Uintah.
     Non-editable simply means that the colors cannot be
     interactively manipulated.  The Module does, allow
     for the the resolution of the colormaps to be changed.
     This class sets up the data structures for Colormaps and
     creates a module from which the user can choose from several
     popular colormaps.  By clicking in the Color band the user 
     manipulate the transparency of the color.  This is useful,
     for example, when volume rendering, though many visualization
     tool ignore the transparency data.

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


private:

  GuiString tcl_status; 
  GuiString positionList;
  GuiString nodeList;
  GuiInt width;
  GuiInt height;
  GuiInt mapType;
  GuiInt minRes;
  GuiInt resolution;
  ColorMapOPort  *outport;
  ColorMapHandle cmap;
  Array1< Color > colors;
  bool genMap(const clString& s);

}; //class GenStandardColorMaps

} // End namespace SCIRun
#endif
