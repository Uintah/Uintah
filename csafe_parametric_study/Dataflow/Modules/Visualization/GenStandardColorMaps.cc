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

#include <Dataflow/Network/Module.h> 
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Core/Malloc/Allocator.h>

#include <math.h>
#include <vector>
#include <sstream>

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
  ColorMapHandle genMap(const string& s, int res, double gamma, bool faux);

  ColorMapHandle cmap_out_handle_;

  GuiString gui_map_name_;
  GuiDouble gui_gamma_;
  GuiInt gui_resolution_;
  GuiInt gui_reverse_;
  GuiInt gui_faux_;
  GuiString gui_position_list_;
  GuiString gui_node_list_;
  GuiInt gui_width_;
  GuiInt gui_height_;
  
}; //!class GenStandardColorMaps


DECLARE_MAKER(GenStandardColorMaps)
//!--------------------------------------------------------------- 
GenStandardColorMaps::GenStandardColorMaps(GuiContext* ctx) 
  : Module("GenStandardColorMaps", ctx, Filter, "Visualization", "SCIRun"),
    gui_map_name_(ctx->subVar("mapName"), "Rainbow"),
    gui_gamma_(ctx->subVar("gamma"), 0),
    gui_resolution_(ctx->subVar("resolution"), 256),
    gui_reverse_(ctx->subVar("reverse"), 0),
    gui_faux_(ctx->subVar("faux"), 0),
    gui_position_list_(ctx->subVar("positionList"), ""),
    gui_node_list_(ctx->subVar("nodeList"), ""),
    gui_width_(ctx->subVar("width"), 1),
    gui_height_(ctx->subVar("height"), 1)
{ 
} 


GenStandardColorMaps::~GenStandardColorMaps()
{
} 


void GenStandardColorMaps::execute() 
{
  string tclRes;
  get_gui()->eval(get_id()+" getColorMapString", tclRes);
  
  if( !cmap_out_handle_.get_rep() ||

      gui_map_name_.changed( true ) ||
      gui_gamma_.changed( true ) ||
      gui_resolution_.changed( true ) ||
      gui_reverse_.changed( true ) ||
      gui_faux_.changed( true ) ||
      gui_position_list_.changed( true ) ||
      gui_node_list_.changed( true ) ) {
    
    update_state(Executing);

    cmap_out_handle_ =
      genMap(tclRes, gui_resolution_.get(), gui_gamma_.get(), gui_faux_.get());
  }

  send_output_handle( "ColorMap", cmap_out_handle_, true );
}


ColorMapHandle
GenStandardColorMaps::genMap(const string& s, int res, double gamma, bool faux)
{
  std::istringstream is(s);

  if( !is.good() ) {
    return 0;

  } else {
    unsigned int i;

    //! Slurp in the rgb values.
    unsigned int rgbsize;
    is >> rgbsize;
    vector< Color > rgbs(rgbsize);
    vector< float > rgbT(rgbsize);
    for (i = 0; i < rgbsize; i++) {
      int r, g, b;
      is >> r >> g >> b;
      rgbs[i] = Color(r/255.0, g/255.0, b/255.0);
      rgbT[i] = i/float(rgbsize-1);
    }

    //! Shift RGB
    const double bp = 1.0 / tan( M_PI_2 * (0.5 + gamma * 0.5));
    for (i=0; i < rgbT.size(); i++) {
      rgbT[i] = pow((double)rgbT[i], bp);
    }
    

    //! Read in the alpha values.  This is the screen space curve.
    unsigned int asize;
    float w, h;
    is >> asize;
    is >> w >> h;
    vector< float > alphas(asize);
    vector< float > alphaT(asize);

    for (i = 0; i < asize; i++) {
      int x, y;
      is >> x >> y;
      alphaT[i] = x / w;
      alphas[i] = 1.0 - y / h;
    }
    
    //! The screen space curve may not contain the default endpoints.
    //! Add them in here if needed.
    if (alphaT.empty() || alphaT.front() != 0.0) {
      alphas.insert(alphas.begin(), 0.5);
      alphaT.insert(alphaT.begin(), 0.0);
    }
    if (alphaT.back() != 1.0) {
      alphas.push_back(0.5);
      alphaT.push_back(1.0);
    }
    
    return scinew ColorMap(rgbs, rgbT, alphas, alphaT, res);
  }
}

} // End namespace SCIRun
