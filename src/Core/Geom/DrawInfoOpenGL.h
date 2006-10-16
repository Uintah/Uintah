/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

  
/*
 *  DrawInfoOpenGL.h: Displayable Geometry
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Core_Geom_DrawInfoOpenGL_h
#define SCI_Core_Geom_DrawInfoOpenGL_h 1

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Plane.h>

#include <Core/Geom/share.h>

#include <map>
#include <list>

class GLUquadric;

namespace SCIRun {

class ViewWindow;
class Material;
class GeomDL;

const int     GEOM_FONT_COUNT = 5;

struct SCISHARE DrawInfoOpenGL {
public:
  DrawInfoOpenGL();
  ~DrawInfoOpenGL();

  enum DrawType {
    WireFrame,
    Flat,
    Gouraud
  };

  typedef std::map<GeomDL *, std::pair<unsigned int, unsigned int> > dl_map_t;
  typedef std::list<unsigned int> dl_list_t;

  void          set_drawtype(DrawType dt);
  DrawType      get_drawtype();
  bool          dl_lookup(GeomDL *obj, unsigned int &state, unsigned int &dl);
  bool          dl_update(GeomDL *obj, unsigned int state);
  bool          dl_addnew(GeomDL *obj, unsigned int state, unsigned int &dl);
  bool          dl_remove(GeomDL *obj);
  void          init_lighting(int use_light);
  void          init_clip(void);
  void          init_view(double znear, double zfar, Point& eyep, Point& lookat);
  bool          init_font(int a);
  void          set_material(Material*);
  void          reset();

  int           polycount_;
  int           lighting_;
  int           show_bbox_;
  int           currently_lit_;
  int           pickmode_;
  int           pickchild_;
  int           npicks_;
  int           fog_;
  int           cull_;
  int           display_list_p_;
  bool          mouse_action_;
  int           check_clip_; // see if you should ignore clipping planes    
  int           clip_planes_; // clipping planes that are on
  Plane         planes_[6]; // clipping plane equations
  Material*     current_matl_;
  int           ignore_matl_;
  GLUquadric*   qobj_;
  Vector        view_;  // view vector...
  int           axis_;     // which axis you are working with...
  int           dir_;      // direction +/- -> depends on the view...
  double        ambient_scale_;
  double        diffuse_scale_;
  double        specular_scale_;
  double        emission_scale_;
  double        shininess_scale_;
  double        point_size_; // so points can be thicker than 1 pixel
  double        line_width_; // so lines can be thicker than 1 pixel
  double        polygon_offset_factor_; // so lines & points are offset from faces
  double        polygon_offset_units_; // so lines & points are offset from faces
  int           fontstatus_[GEOM_FONT_COUNT];
  unsigned int  fontbase_[GEOM_FONT_COUNT];
  int           using_cmtexture_;
  unsigned int  cmtexture_;  // GeomColorMap texture, 1D
  dl_map_t      dl_map_;
  dl_list_t     dl_freelist_;
private:
  DrawType      drawtype_;
};

} // End namespace SCIRun


#endif /* SCI_Geom_DrawInfoOpenGL_h */

