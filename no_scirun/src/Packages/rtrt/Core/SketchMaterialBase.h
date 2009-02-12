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



#ifndef SKETCHMATERIALBASE_H
#define SKETCHMATERIALBASE_H 1

#include <Packages/rtrt/Core/Color.h>

namespace rtrt {

  class SketchMaterialBase {
  public:
    //  The thickness of the silhouette edges.  The gui version is the
    //  one controlled by the gui, while the other is the one used for
    //  rendering (and is the inverse).
    float sil_thickness, inv_sil_thickness, gui_sil_thickness;
    // This is the color the silhouette edge
    Color sil_color;
    float gui_sil_color_r, gui_sil_color_g, gui_sil_color_b;
    // This is the color the ridge
    Color ridge_color;
    float gui_ridge_color_r, gui_ridge_color_g, gui_ridge_color_b;
    // This is the color the valley
    Color valley_color;
    float gui_valley_color_r, gui_valley_color_g, gui_valley_color_b;

    int normal_method, gui_normal_method;
    int show_silhouettes, gui_show_silhouettes;
    int use_cool2warm, gui_use_cool2warm;

    SketchMaterialBase(float sil_thickness):
      sil_thickness(sil_thickness), inv_sil_thickness(1/sil_thickness),
      gui_sil_thickness(sil_thickness),
      sil_color(0,0,0),
      gui_sil_color_r(0), gui_sil_color_g(0), gui_sil_color_b(0),
      ridge_color(1,1,1),
      gui_ridge_color_r(1), gui_ridge_color_g(1), gui_ridge_color_b(1),
      valley_color(0,0,0),
      gui_valley_color_r(0), gui_valley_color_g(0), gui_valley_color_b(0),
      normal_method(1), gui_normal_method(1),
      show_silhouettes(1), gui_show_silhouettes(1),
      use_cool2warm(1), gui_use_cool2warm(1)
    {}
    virtual ~SketchMaterialBase() {}
  };

} // end namespace rtrt

#endif
