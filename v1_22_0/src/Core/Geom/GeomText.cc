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
 *  GeomText.cc:  Texts of GeomObj's
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Mar 1998
 *
 *  Copyright (C) 1998 SCI Text
 */

#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geom/GeomText.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Thread/MutexPool.h>
#include <iostream>

#ifdef HAVE_FTGL
#include <FTGL/FTGLTextureFont.h>
#endif


namespace SCIRun {

using std::ostream;

static Persistent* make_GeomText()
{
    return scinew GeomText;
}

PersistentTypeID GeomText::type_id("GeomText", "GeomObj", make_GeomText);

GeomText::GeomText()
  : GeomObj()
{
}

GeomText::GeomText( const string &text, const Point &at, const Color &c,
		    const string &fontsize)
  : GeomObj(), text(text), fontsize(fontsize), at(at), c(c)
{
}


GeomText::GeomText(const GeomText& copy)
: GeomObj(copy)
{
  text = copy.text;
  at = copy.at;
  c = copy.c;
}


GeomObj* GeomText::clone()
{
    return scinew GeomText(*this);
}

void GeomText::get_bounds(BBox& in_bb)
{
  in_bb.extend( at );
}

GeomText::~GeomText()
{
}

void GeomText::reset_bbox()
{
}

#define GEOMTEXT_VERSION 2

void
GeomText::io(Piostream& stream)
{

    const int version = stream.begin_class("GeomText", GEOMTEXT_VERSION);
    // Do the base class first...
    GeomObj::io(stream);
    Pio(stream, at);
    Pio(stream, text);
    if (version > 1)
    {
      Pio(stream, c);
    }
    stream.end_class();
}

static Persistent* make_GeomTexts()
{
    return scinew GeomTexts;
}

PersistentTypeID GeomTexts::type_id("GeomTexts", "GeomObj", make_GeomTexts);

GeomTexts::GeomTexts()
  : GeomObj(),
    fontindex_(2)
{
}


GeomTexts::GeomTexts(const GeomTexts& copy) :
  GeomObj(copy),
  fontindex_(copy.fontindex_),
  text_(copy.text_),
  location_(copy.location_),
  color_(copy.color_)
{
}


GeomTexts::~GeomTexts()
{
}


GeomObj* GeomTexts::clone()
{
    return scinew GeomTexts(*this);
}


void
GeomTexts::get_bounds(BBox& in_bb)
{
  for (unsigned int i = 0; i < location_.size(); i++)
  {
    in_bb.extend( location_[i] );
  }
}

void
GeomTexts::reset_bbox()
{
}


void
GeomTexts::set_font_index(int a)
{
  if (a >= 0 && a < 5)
  {
    fontindex_ = a;
  }
}


void
GeomTexts::add(const string &t, const Point &p)
{
  text_.push_back(t);
  location_.push_back(p);
}

void
GeomTexts::add(const string &t, const Point &p, const Color &c)
{
  add(t, p);
  color_.push_back(c);
}

void
GeomTexts::add(const string &t, const Point &p, float index)
{
  add(t, p);
  index_.push_back(index);
}


#define GEOMTEXTS_VERSION 2

void
GeomTexts::io(Piostream& stream)
{
    const int version = stream.begin_class("GeomTexts", GEOMTEXTS_VERSION);
    // Do the base class first.
    GeomObj::io(stream);
    Pio(stream, fontindex_);
    Pio(stream, text_);
    Pio(stream, location_);
    Pio(stream, color_);
    if (version > 1) { Pio(stream, index_); }
    stream.end_class();
}

static Persistent* make_GeomTextsCulled()
{
    return scinew GeomTextsCulled;
}

PersistentTypeID GeomTextsCulled::type_id("GeomTextsCulled", "GeomTexts", make_GeomTextsCulled);


GeomTextsCulled::GeomTextsCulled()
  : GeomTexts()
{
}


GeomTextsCulled::GeomTextsCulled(const GeomTextsCulled& copy) :
  GeomTexts(copy),
  normal_(copy.normal_)
{
}


GeomTextsCulled::~GeomTextsCulled()
{
}


GeomObj* GeomTextsCulled::clone()
{
    return scinew GeomTextsCulled(*this);
}


void
GeomTextsCulled::add(const string &t, const Point &p, const Vector &n)
{
  GeomTexts::add(t, p);
  normal_.push_back(n);
}


void
GeomTextsCulled::add(const string &t, const Point &p,
		     const Vector &n, const Color &c)
{
  GeomTexts::add(t, p, c);
  normal_.push_back(n);
}


void
GeomTextsCulled::add(const string &t, const Point &p,
		     const Vector &n, float index)
{
  GeomTexts::add(t, p, index);
  normal_.push_back(n);
}


#define GEOMCULLEDTEXTS_VERSION 1

void
GeomTextsCulled::io(Piostream& stream)
{
    stream.begin_class("GeomTextsCulled", GEOMCULLEDTEXTS_VERSION);
    // Do the base class first...
    GeomTexts::io(stream);
    Pio(stream, normal_);
    stream.end_class();
}


static Persistent* make_GeomTextTexture()
{
  return scinew GeomTextTexture(string("invalid"));
}

PersistentTypeID GeomTextTexture::type_id("GeomTextTexture", "GeomObj", make_GeomTextTexture);

GeomTextTexture::GeomTextTexture(const string &fontfile)
  : GeomObj(),
    text_(),
    origin_(0.,0.,0.),
    up_(0.,1.,0.),
    left_(0.,1.,0.),
    color_(Color(1,1,1)),
    anchor_(sw)
#ifdef HAVE_FTGL
    ,font_(scinew GeomFTGLFontRenderer(fontfile.c_str()))
#endif
    ,up_hack_(false)
{
}

GeomTextTexture::GeomTextTexture( const string &fontfile,
				  const string &text, 
				  const Point &at,
				  const Vector &up,
				  const Vector &left,
				  const Color &c)
  : GeomObj(), 
    text_(text),
    origin_(at),
    up_(up),
    left_(left),
    color_(c),
    anchor_(sw),
    own_font_(true)
#ifdef HAVE_FTGL
    ,font_(scinew GeomFTGLFontRenderer(fontfile.c_str()))
#endif
    ,up_hack_(false)
{
}

#ifdef HAVE_FTGL
GeomTextTexture::GeomTextTexture( GeomFTGLFontRendererHandle font,
				  const string &text, 
				  const Point &at,
				  const Vector &up,
				  const Vector &left,
				  const Color &c)
  : GeomObj(), 
    text_(text),
    origin_(at),
    up_(up),
    left_(left),
    color_(c),
    anchor_(sw),
    own_font_(false),
    font_(font),
    up_hack_(false)
{
}
#endif


GeomTextTexture::GeomTextTexture(const GeomTextTexture& copy)
: GeomObj(copy)
{
}


GeomObj* GeomTextTexture::clone()
{
    return scinew GeomTextTexture(*this);
}

void GeomTextTexture::get_bounds(BBox& in_bb)
{
#ifdef HAVE_FTGL
  if (!font_.get_rep()) return;
  BBox bbox;
  font_->get_bounds(bbox);
  Transform identity;
  build_transform(identity);
  Point ll,ur;
  transform_.project(bbox.min(),ll);
  transform_.project(bbox.max(),ur);
  in_bb.extend(ll);
  in_bb.extend(ur);
#else
  in_bb.extend(origin_);
#endif
}

GeomTextTexture::~GeomTextTexture()
{
}

void GeomTextTexture::render() 
{
#ifdef HAVE_FTGL
  font_->render(text_.c_str());
#endif
}


void
GeomTextTexture::build_transform(Transform &modelview) {
#ifdef HAVE_FTGL
  if (!font_.get_rep()) return;
  float llx, lly, llz,urx,ury,urz;
  BBox bbox;
  font_->get_bounds(bbox,text_);
  llx = bbox.min().x();
  lly = bbox.min().y();
  llz = bbox.min().z();

  urx = bbox.max().x();
  ury = bbox.max().y();
  urz = bbox.max().z();

  Vector tempup, templeft;
  modelview.project_normal(up_, tempup);
  if (tempup.y() < 0.0) {
    up_ *= -1.;
  }

  modelview.project_normal(left_, templeft);
  if (templeft.x() < 0.0) {
    left_ *= -1.;
  }

  Vector transnormal(Cross(left_,up_));
  modelview.project_inplace(transnormal);
  if (transnormal.z() < 0.0) {
    left_ *= -1.;
  }

  Transform xform_to_origin;
  Vector xlat2(-(llx+urx)/2., -(lly+ury)/2., -(llz+urz)/2.);
  xform_to_origin.post_translate(xlat2);

  Transform xform_rotate;
  Vector normal = Cross(left_,up_).normal(), left = left_.normal(), up = up_.normal();
  Vector x(1.,0.,0), y(0.,1.,0.), z(0.,0.,1.);
  double dx, dy, dz;
  bool done = false;
  while (!done) {
    done = true;
    dx = fabs(Dot(left, x.normal())), dy = fabs(Dot(up,y.normal())), dz = fabs(Dot(normal, z.normal()));
    if (dx < 0.999 && dx <= dy && dx <= dz) {
      xform_rotate.rotate(x, left);
      done = false;
    } else if (dy < 0.999 && dy <= dx && dy <= dz) {
      xform_rotate.rotate(y, up);
      done = false;
    } else if (dz < 0.999) {
      xform_rotate.rotate(z, normal);
      done = false;
    }
    xform_rotate.project(Vector(1.,0.,0.),x);
    xform_rotate.project(Vector(0.,1.,0.),y);
    xform_rotate.project(Vector(0.,0.,1.),z);
  }


  //std::cerr << text_ << ": origin: " << origin_ << "  left: " << left_ << "  up: " << up_ << " normal: " << normal << " x: " << x << " y: " << y << " z: " << z << " dx: " << dx << " dy: " << dy << " dz: " << dz << std::endl;
  Transform xform_scale;
  Vector scales (Dot(left, x.normal()), Dot(up,y.normal()), Dot(normal, z.normal()));
  scales /= ury-lly;
  scales *= up_.length();
  xform_scale.post_scale(scales);
  
  Transform xform_to_position;
  Vector delta = origin_.asVector();
  xform_to_position.post_translate(delta);

  transform_.load_identity();
  transform_.pre_trans(xform_to_origin);
  transform_.pre_trans(xform_scale);
  transform_.pre_trans(xform_rotate);  
  transform_.pre_trans(xform_to_position);
#endif
}
    
void
GeomTextTexture::io(Piostream& stream)
{
  stream.begin_class("GeomTextTexture", 1);
  // Do the base class first...
  GeomObj::io(stream);
  stream.end_class();
}



#ifdef HAVE_FTGL
static Persistent* make_GeomFTGLFontRenderer()
{
  return scinew GeomFTGLFontRenderer(string(""));
}

PersistentTypeID GeomFTGLFontRenderer::type_id("GeomFTGLFontRenderer", "GeomObj", make_GeomFTGLFontRenderer);

GeomFTGLFontRenderer::GeomFTGLFontRenderer(const string &fontfile, 
					   int ptRez,
					   int screenRez) :
  filename_(fontfile)
{
  font_ = scinew FTGLTextureFont(fontfile.c_str());
  font_->CharMap(ft_encoding_unicode);
  set_resolution(ptRez,screenRez);
}

GeomFTGLFontRenderer::GeomFTGLFontRenderer(const GeomFTGLFontRenderer& copy)
  : GeomObj(copy), font_(copy.font_)
{
}

GeomObj* GeomFTGLFontRenderer::clone()
{
    return scinew GeomFTGLFontRenderer(*this);
}

void
GeomFTGLFontRenderer::set_resolution(int ptRez, int screenRez)
{
  //  int ptRez = int(ptSize*screenRez);
  if (screenRez_ != screenRez || ptRez_ != ptRez) {
    ptRez_ = ptRez;
    screenRez_ = screenRez;
    font_->FaceSize(ptRez_, screenRez_);
  }
}

void GeomFTGLFontRenderer::get_bounds(BBox& in_bb)
{
  if (!font_) return;
  float llx, lly, llz,urx,ury,urz;
  font_->BBox("X",llx,lly,llz,urx,ury,urz);
  in_bb.extend(Point(llx,lly,llz));
  in_bb.extend(Point(urx,ury,urz));
}


void GeomFTGLFontRenderer::get_bounds(BBox& in_bb, const string &text)
{
  if (!font_) return;
  float llx, lly, llz,urx,ury,urz;
  font_->BBox(text.c_str(),llx,lly,llz,urx,ury,urz);
  in_bb.extend(Point(llx,lly,llz));
  in_bb.extend(Point(urx,ury,urz));
}

GeomFTGLFontRenderer::~GeomFTGLFontRenderer()
{
  //  delete font_;
}

void
GeomFTGLFontRenderer::render(const string &text) 
{
  font_->Render(text.c_str());
}

void
GeomFTGLFontRenderer::io(Piostream& stream)
{
  stream.begin_class("GeomFTGLFontRendere", 1);
  // Do the base class first...
  GeomObj::io(stream);
  stream.end_class();
}

#endif


} // End namespace SCIRun
