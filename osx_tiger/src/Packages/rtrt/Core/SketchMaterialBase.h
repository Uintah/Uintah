
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
