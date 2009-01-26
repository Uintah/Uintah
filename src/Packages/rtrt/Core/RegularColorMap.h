/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#ifndef RTRT_REGULAR_COLOR_MAP_H__
#define RTRT_REGULAR_COLOR_MAP_H__ 1

/**************************************************************
 *
 * RegularColorMap
 *
 * This class provides utilities for regular color maps such as
 * loading from a file, preclassified colormaps, and switching
 * colormaps from within the rtrt framework.
 *
 * Author: James Bigler
 * Original Date: Feb. 2, 2004
 *
 */

#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Core/Thread/Mutex.h>

namespace rtrt {

  template<class T> class Array1;
  
  class RegularColorMap {
  protected:
    // This is used to prevent race conditions
    SCIRun::Mutex lock;
    
    // These are the colors used in the colormap.
    Array1<Color> color_list;

    // This is true whenever you need to copy over the colormap from
    // new_blended_colors to blended_colors.
    bool update_blended_colors;
    Array1<Color> new_blended_colors;

    void fillColor(int type, Array1<Color> &colors);
    bool fillColor(const char* filename, Array1<Color> &colors);

    // This uses color_list_lookup to fill blended with interpolated
    // colors.  blended should already be resized to accommodate the
    // input.
    void fillBlendedColors(Array1<Color> &blended);
  public:
    // We need a way to create one of these things.
    RegularColorMap(const char* filename, int size = 256);
    RegularColorMap(Array1<Color> &input_color_list, int size = 256);

    // This can be used to generate default values
    RegularColorMap(int type = RegCMap_InvRainbow, int size = 256);

    virtual ~RegularColorMap();

    // This return the enum of the type based on the input string
    // passed in.
    static int parseType(const char* type);
    
    enum {
      RegCMap_Unknown,
      RegCMap_InvRainbow,
      RegCMap_Rainbow,
      RegCMap_InvBlackBody,
      RegCMap_BlackBody,
      RegCMap_InvGrayScale,
      RegCMap_GrayScale,
      RegCMap_InvRIsoLum
    };

    // These are used to change the colors in the colormap.
    void changeColorMap(int type);
    // True is returned if everything went OK.
    bool changeColorMap(const char* filename);

    // This allows one to change the number of bins used for the blended_colors
    bool changeSize(int new_size);

    // This allows a material to be animated at the frame change
    virtual void animate(double t, bool& changed);

    //////////////////////////////////////////////////////////////
    // Data
    //////////////////////////////////////////////////////////////
    
    // The blended result which has been padded out to the size
    // specified.
    ScalarTransform1D<float, Color> blended_colors_lookup;
    // These are what are used by blended_colors_lookup to do the lookup
    Array1<Color> blended_colors;
  };

} // end namespace rtrt

#endif
