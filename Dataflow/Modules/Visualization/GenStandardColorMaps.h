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


#ifndef GENSTANDARDCOLORMAPS_H
#define GENSTANDARDCOLORMAPS_H 1

#include <Core/GuiInterface/GuiVar.h> 
#include <Core/Datatypes/Color.h>
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
  GenStandardColorMaps(GuiContext* ctx); 

  // GROUP: Destructors: 
  ////////// 
  // Destructor
  virtual ~GenStandardColorMaps(); 

  ////////// 
  // execution scheduled by scheduler   
  virtual void execute(); 


private:

  GuiString positionList;
  GuiString nodeList;
  GuiInt width;
  GuiInt height;
  GuiString mapName;
  GuiInt reverse;
  GuiInt minRes;
  GuiInt resolution;
  GuiInt realres;
  GuiDouble gamma;
  GuiInt faux;
  
}; //class GenStandardColorMaps

} // End namespace SCIRun
#endif
