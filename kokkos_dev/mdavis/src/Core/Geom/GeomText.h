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
 *  GeomText.h:  Texts of GeomObj's
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Mar 1998
 *
 *  Copyright (C) 1998 SCI Text
 */

#ifndef SCI_Geom_Text_h
#define SCI_Geom_Text_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Transform.h>

#include <Core/Datatypes/Color.h>


class FTGLTextureFont;

namespace SCIRun {

class GeomText : public GeomObj {
public:
  string text;
  string fontsize;
  Point at;
  Color c;
public:
  GeomText();
  GeomText(const GeomText&);

  // note: possible strings for fontsize are "*", "0", "6", "8", 
  // "10", "11", "12", "13", "14", "15", "16", "17" otherwise it will
  // be set to "*"---Kurt Zimmerman
  GeomText( const string &, const Point &, const Color &c = Color(1,1,1),
	    const string& fontsize = "*");
  virtual ~GeomText();
  virtual GeomObj* clone();

  virtual void reset_bbox();
  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


class GeomTexts : public GeomObj {
protected:
  int fontindex_;
  vector<string> text_;
  vector<Point>  location_;
  vector<Color>  color_;
  vector<float>  index_;

public:
  GeomTexts();
  GeomTexts(const GeomTexts &);
  virtual ~GeomTexts();
  virtual GeomObj* clone();

  void set_font_index(int);

  void add (const string &text, const Point &loc);
  void add (const string &text, const Point &loc, const Color &c);
  void add (const string &text, const Point &loc, float index);

  virtual void reset_bbox();
  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class GeomTextsCulled : public GeomTexts {
protected:
  vector<Vector> normal_;

public:
  GeomTextsCulled();
  GeomTextsCulled(const GeomTextsCulled &);
  virtual ~GeomTextsCulled();
  virtual GeomObj* clone();

  void add (const string &text, const Point &loc, const Vector &vec);
  void add (const string &text, const Point &loc, const Vector &vec,
	    const Color &c);
  void add (const string &text, const Point &loc, const Vector &vec,
	    float index);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

#ifdef HAVE_FTGL
class GeomFTGLFontRenderer : public GeomObj{
private:
  FTGLTextureFont *font_;
  int ptRez_,screenRez_;
  string filename_;
public:
  GeomFTGLFontRenderer(const string &filename, int ptRez=72, int screenRez=72);
  GeomFTGLFontRenderer(const GeomFTGLFontRenderer &);
  virtual ~GeomFTGLFontRenderer();
  virtual GeomObj *clone();
  static PersistentTypeID type_id;
  virtual void io(Piostream&);
  virtual void get_bounds(BBox&);
  virtual void get_bounds(BBox&,const string &);
  virtual void draw(DrawInfoOpenGL*, Material*, double time){}
  const string filename() { return filename_; }
  void render(const string &text);
  void set_resolution(int ptRez=72, int screenRez=72);
  int get_ptRez() { return ptRez_; }
  int get_screenRez() { return screenRez_; }
};
  
typedef LockingHandle<GeomFTGLFontRenderer> GeomFTGLFontRendererHandle;

#endif


class GeomTextTexture : public GeomObj {
public:
  enum anchor_e { n, e, s, w, ne, se, sw, nw, c};
private:

  void		build_transform(Transform &);
  void		render();
  Transform	transform_;
  string	text_;
  Point		origin_;
  Vector	up_;
  Vector	left_;
  Color		color_;
  anchor_e	anchor_;
  bool		own_font_;
#ifdef HAVE_FTGL
  GeomFTGLFontRendererHandle font_;
#endif
public:
  GeomTextTexture(const string &fontfile);
  GeomTextTexture(const GeomTextTexture&);
  GeomTextTexture(const string &fontfile,
		  const string &text, 
		  const Point &at,
		  const Vector &up,
		  const Vector &left,
		  const Color &c = Color(1.,1.,1.));
#ifdef HAVE_FTGL
  GeomTextTexture(GeomFTGLFontRendererHandle font,
		  const string &text, 
		  const Point &at,
		  const Vector &up,
		  const Vector &left,
		  const Color &c = Color(1.,1.,1.));
#endif

  virtual ~GeomTextTexture();
  virtual GeomObj* clone();
  
  virtual void get_bounds(BBox&);

  void	set_anchor(anchor_e anchor) { anchor_ = anchor; }
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  virtual void io(Piostream&);
   static PersistentTypeID type_id;
  bool		up_hack_;
};



} // End namespace SCIRun


#endif /* SCI_Geom_Text_h */

