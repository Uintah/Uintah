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



/*
 *  Freetype.h: Interface to Freetype 2 library
 *
 *  Written by:
 *   McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   October 2004
 *
 *  Copyright (C) 2004 Scientific Computing and Imaging Institute
 */

#ifndef SCIRun_src_Core_Geom_Freetype_h
#define SCIRun_src_Core_Geom_Freetype_h 1

#include <sci_defs/ogl_defs.h>
#ifdef HAVE_FREETYPE
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
#endif
#include <Core/Geometry/Point.h>
#include <string>
#include <vector>
using std::string;
using std::vector;

namespace SCIRun {

class FreeTypeFace;
class BBox;

class FreeTypeLibrary {
public:
  FreeTypeLibrary();
  virtual ~FreeTypeLibrary();
  FreeTypeFace *	load_face(string);
#ifdef HAVE_FREETYPE
  FT_Library		library_;
#endif
};


class FreeTypeFace {
public:
  FreeTypeFace(FreeTypeLibrary *, string);
  virtual ~FreeTypeFace();
  bool			has_kerning_p();
  bool			scalable_p();
  void			set_dpi(unsigned int, unsigned int);
  void			set_points(double points);
  string		get_family_name();
  string		get_style_name();
  string		get_filename();
#ifdef HAVE_FREETYPE
  FT_Face		ft_face_;
#endif
private:
  double		points_;
  unsigned int		x_dpi_;
  unsigned int		y_dpi_;
  FreeTypeLibrary *	library_;
  string		filename_;
};


class FreeTypeGlyph {
public:
  FreeTypeGlyph();
  virtual ~FreeTypeGlyph();
#ifdef HAVE_FREETYPE
  FT_Glyph		glyph_;
  FT_UInt		index_;
  Point			position_;
  FT_Vector		ft_position();
#endif
};

typedef vector<FreeTypeGlyph *> FreeTypeGlyphs;

class FreeTypeText {
public:
  FreeTypeText(string text, FreeTypeFace *face, Point *pos = 0);
  virtual ~FreeTypeText();
  enum anchor_e { n, e, s, w, ne, se, sw, nw, c};
  
private:
  string		text_;
  FreeTypeFace *	face_;
  FreeTypeGlyphs	glyphs_;
  Point			position_;
  anchor_e		anchor_;
  
public:
  FreeTypeFace *	get_face();
  void			layout();
  void			render(int, int, unsigned char *);
  void			get_bounds(BBox &);
  void			set_anchor(anchor_e anchor) { anchor_ = anchor; }
  void			set_position(const Point &pt);
#ifdef HAVE_FREETYPE
  FT_Vector		ft_position(); 
#endif
};
  
 
 
} // End namespace SCIRun


#endif 

