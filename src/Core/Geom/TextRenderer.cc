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
 *  TextRenderer.cc
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   November, 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <Core/Geom/TextRenderer.h>
#include <Core/Geom/TextureObj.h>
#include <Core/Math/MiscMath.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geometry/BBox.h>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Environment.h>
#include <Core/Util/SimpleProfiler.h>

#ifdef HAVE_FREETYPE
#include FT_MODULE_ERRORS_H
#endif

namespace SCIRun {  

static SimpleProfiler profiler("TextRenderer");

TextRenderer::TextRenderer(FreeTypeFace *face) :
  face_(face),
  glyphs_(),
  shadow_offset_(make_pair(1,-1)),
  textures_(),
  texture_(0),
  x_(0),
  y_(0),
  height_(0)
{
  profiler.disable();
  set_color(1.0, 1.0, 1.0, 1.0);
  set_shadow_color(0.0, 0.0, 0.0, 1.0);
}

TextRenderer::~TextRenderer() 
{
  TextureSet::iterator titer = textures_.begin(), tlast = textures_.end();
  while (titer != tlast) {
    delete (*titer);
    ++titer;
  }

  GlyphMap::iterator giter = glyphs_.begin(), glast = glyphs_.end();
  while (giter != glast) {
    delete (giter->second);
    ++giter;
  }
}

int
TextRenderer::width(const string &text, int flags) {
  render_string_glyphs_to_texture(text);
#ifdef HAVE_FREETYPE
  int wid;
  for (unsigned int c = 0; c < text.size(); ++c)
    wid += glyphs_[text[c]]->ft_metrics_.horiAdvance;
  return wid / 64;
#else
  return 0;
#endif
}

int
TextRenderer::height(const string &text, int flags) {
#ifdef HAVE_FREETYPE
  render_string_glyphs_to_texture(text);
  int hei = 0;
  for (unsigned int c = 0; c < text.size(); ++c)
    hei = Max(hei, (int)glyphs_[text[c]]->ft_metrics_.horiBearingY);
  return hei / 64;
#else
  return 0;
#endif
}


void
TextRenderer::set_color(float r, float g, float b, float a)
{
  color_[0] = r;
  color_[1] = g;
  color_[2] = b;
  color_[3] = a;
}


void
TextRenderer::set_color(float rgba[4]) 
{
  set_color(rgba[0], rgba[1], rgba[2], rgba[3]);
}


void
TextRenderer::set_shadow_color(float r, float g, float b, float a)
{
  shadow_color_[0] = r;
  shadow_color_[1] = g;
  shadow_color_[2] = b;
  shadow_color_[3] = a;
}


void
TextRenderer::set_shadow_color(float rgba[4]) 
{
  set_shadow_color(rgba[0], rgba[1], rgba[2], rgba[3]);
}

void
TextRenderer::set_shadow_offset(int x, int y)
{
  shadow_offset_.first = x;
  shadow_offset_.second = y;
}



unsigned int
TextRenderer::layout_text(const string &text, 
                          const Point &anchor, 
                          const Vector &left, 
                          const Vector &up, 
                          int flags) 
{
#ifdef HAVE_FREETYPE
  profiler.enter("layout_text");
  //  render_string_glyphs_to_texture("X");
  //  GlyphInfo *standard = glyphs_['X'];
  const bool kern = face_->has_kerning_p();
  bool shadow = flags & SHADOW;
  const bool vert = flags & VERTICAL;
  unsigned int passes = shadow ? 2 : 1;
  unsigned num = text.size()*passes;
  if (layout_.size() < num)
    layout_.resize(num);
  profiler("resize");
  int width_pt = 0;
  int height_pt = 0;
  Vector left_pt = left / 64.0;
  Vector up_pt = up / 64.0;
  Point ll;
  int linegap = (face_->ft_face_->height - 
                 face_->ft_face_->ascender + 
                 face_->ft_face_->descender);
  Point position;
  profiler("start");
  for (unsigned int p = 0; p < passes; ++p) {
    int linenum = 0;
    width_pt = 0;
    height_pt = 0;
    position = anchor;
    if (shadow)
      position = (position + 
                  shadow_offset_.first * left + 
                  shadow_offset_.second * up);
    for (unsigned int c = 0; c < text.size(); ++c) {
      ASSERT(p*text.size()+c < layout_.size());
      LayoutInfo &layout = layout_[p*text.size()+c];

      if (!glyphs_[text[c]]) { 
        layout.glyph_info_ = glyphs_[' '];
      } else {
        layout.glyph_info_ = glyphs_[text[c]];
      }
      ASSERT(layout.glyph_info_);

      if (flags & REVERSE) 
        layout.color_ = shadow ? color_ : shadow_color_;
      else
        layout.color_ = shadow ? shadow_color_ : color_;
      FT_Glyph_Metrics &metrics = layout.glyph_info_->ft_metrics_;

      if (!vert) {

        if (text[c] == '\n') {
          int advance = (++linenum)*(linegap + face_->ft_face_->height);
          position = anchor - up_pt*advance;
          layout.glyph_info_ = 0;
        } else {
          if (c && kern && !vert) {
            FT_Vector kerning; 
            FT_Get_Kerning(face_->ft_face_, 
                           layout_[c-1].glyph_info_->index_,
                           layout.glyph_info_->index_, 
                           FT_KERNING_DEFAULT, &kerning); 
            position += left_pt * kerning.x;
            position += up_pt * kerning.y;
          }
          
          height_pt = Max(height_pt, (int)metrics.horiBearingY);
          width_pt = width_pt + metrics.horiAdvance;
          
          ll = (position + 
                up_pt * (metrics.horiBearingY - metrics.height) +
                left_pt * metrics.horiBearingX);
          layout.vertices_[0] = ll;
          layout.vertices_[1] = ll + left_pt * metrics.width;
          layout.vertices_[2] = (ll + left_pt * metrics.width + 
                                 up_pt * metrics.height);
          layout.vertices_[3] = ll + up_pt * metrics.height;
          position = position + left_pt * metrics.horiAdvance;
        }
      } else {

        int halfwidth = (metrics.width) >> 1;
        //        halfwidth = halfwidth - halfwidth%64;
        //        halfwidth = halfwidth & ~64;
        width_pt = Max(width_pt, (int)metrics.width);
        int advance = (linegap + 64 + (int)metrics.height);//face_->ft_face_->height);
        height_pt = height_pt + advance;

        ll = (position - up_pt * advance - left_pt*halfwidth);
        layout.vertices_[0] = ll;
        layout.vertices_[1] = ll + left_pt * metrics.width;
        layout.vertices_[2] = (ll + left_pt * metrics.width + 
                               up_pt * metrics.height);
        layout.vertices_[3] = ll + up_pt * metrics.height;
        position = position - up_pt * advance;
      }
    }
    shadow = false;
    profiler("pass");
  }

  Vector width = width_pt * left_pt;
  Vector half_width = (width_pt / 2.0) * left_pt;

  Vector height = height_pt * up_pt;
  Vector half_height = (height_pt / 2.0) * up_pt;
  
  Vector offset(0,0,0);
  if (vert)
    offset = height + half_width; ;// - left_pt * ((width_pt / 2) & ~64);
  switch (flags & ANCHOR_MASK) {
  case N:  offset = offset - half_width - height; break;
  case E:  offset = offset - width - half_height; break;
  case S:  offset = offset - half_width; break;
  case W:  offset = offset - half_height; break;
  case NE: offset = offset - width - height; break;
  case SE: offset = offset - width; break;
  case SW: break;
  case NW: offset = offset - height; break;
  case C:  offset = offset - half_width - half_height; break;
  }
  profiler("compute_offset");
  offset_ = offset;
  for (unsigned int c = 0; c < num; ++c) 
    for (int v = 0; v < 4; ++v)
      for (int o = 0; o < 3; ++o)
        layout_[c].vertices_[v](o) = Floor(layout_[c].vertices_[v](o));
        
  profiler("offset");
  profiler.leave();
  
  return num;
#else
  return 0;
#endif
}



void
TextRenderer::render(const string &text, float x, float y, int flags)
{
#ifdef HAVE_FREETYPE
  profiler.enter("TextRenderer::render");
  CHECK_OPENGL_ERROR();
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  CHECK_OPENGL_ERROR();
  glLoadIdentity();
  glScaled(2.0, 2.0, 2.0);
  glTranslated(-.5, -.5, -.5);
  glColor4d(1.0, 0.0, 0.0, 1.0);
  
  profiler("gl1");
  GLint gl_viewport[4];
  glGetIntegerv(GL_VIEWPORT, gl_viewport);
  float vw = gl_viewport[2];
  float vh = gl_viewport[3];
  glScaled(1/vw, 1/vh, 1.0);

  profiler("gl2");
  glDisable(GL_CULL_FACE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glShadeModel(GL_FLAT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  CHECK_OPENGL_ERROR();
  profiler("gl3");
  render_string_glyphs_to_texture(text);
  profiler("render_string");
  Point anchor(x,y,0.0);
  Vector left(1,0,0);
  Vector up(0,1,0);
  unsigned int num = layout_text(text, anchor, left, up, flags);
  glTranslated(offset_.x(), offset_.y(), offset_.z());
  profiler("layout_text");
  texture_->bind();
  for (unsigned int c = 0; c < num; ++c) {
    LayoutInfo &layout = layout_[c];
    GlyphInfo *glyph = layout.glyph_info_;
    ASSERT(glyph);
    //    if (!glyph) continue;
    glyph->texture_->set_color(layout.color_);
    glyph->texture_->draw(4, layout.vertices_, glyph->tex_coords_);
  }
  profiler("texture draw");
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  CHECK_OPENGL_ERROR();
  profiler.leave();
  profiler.print();
#endif
}


void 
TextRenderer::render_string_glyphs_to_texture(const string &text) {
  for (unsigned int c = 0; c < text.size(); ++c)
    if (glyphs_.find(text[c]) == glyphs_.end()) 
      glyphs_[text[c]] = render_glyph_to_texture(text[c]);
}



TextRenderer::GlyphInfo *
TextRenderer::render_glyph_to_texture(const wchar_t &character) {
#ifdef HAVE_FREETYPE
  string str = " ";
  str[0] = character;
  GlyphInfo *glyph_info = new GlyphInfo();
  //  glyph_info->glyph_ = new FreeTypeGlyph();
  glyph_info->index_ = FT_Get_Char_Index(face_->ft_face_, character);
  if (glyph_info->index_ == 0) {
    cerr << "0 glyph, returning\n";
    return 0;
  }


  FT_Error err;
  err = FT_Load_Glyph(face_->ft_face_, glyph_info->index_, FT_LOAD_DEFAULT);

  if (err) {
    cerr << "Freetype error: " << err << " \n";
    cerr << "FreeType Unable to Load Glyph: ";
    cerr << character;
    cerr <<"  " << to_string(int(character));
    cerr <<"  " <<  string(__FILE__) << to_string(__LINE__) << std::endl;
    return 0;
    //    throw ("FreeType Unable to Load Glyph: "+character+
    //           string(__FILE__)+to_string(__LINE__));
  }

  FT_Glyph glyph;
  err = FT_Get_Glyph(face_->ft_face_->glyph, &glyph);
  if (err) {
    throw ("FreeType Unable to Get Glyph: "+character+
           string(__FILE__)+to_string(__LINE__));
  }

  glyph_info->ft_metrics_ = face_->ft_face_->glyph->metrics;
  err = FT_Glyph_To_Bitmap(&glyph, FT_RENDER_MODE_NORMAL, 0, 1);
  if (err) {
    FT_Done_Glyph(glyph);
    cerr << "Freetype error: " << err << std::endl;
    //    throw "/n/nFreeType Unable to Render Glyph to Bitmap: "+to_string(int(character))+"\n\n";//string(__FILE__)+to_string(__LINE__)
    return 0;
  }

  FT_BitmapGlyph bitmap_glyph = (FT_BitmapGlyph)(glyph);
  
  ASSERT(bitmap_glyph->bitmap.num_grays == 256);
  ASSERT(bitmap_glyph->bitmap.pixel_mode == FT_PIXEL_MODE_GRAY);

  int width = bitmap_glyph->bitmap.width;
  int height = bitmap_glyph->bitmap.rows;

  if (!texture_ || x_ + width > texture_->width()) {
    x_ = 0;
    y_ += height_;
    height_ = 0;
  }

  if (!texture_ || y_ + height > texture_->height()) {
    texture_ = new TextureObj(1, 256, 256);
    textures_.insert(texture_);
    y_ = 0;
    x_ = 0;
    height_ = 0;
  }

  int tex_width = texture_->width();
  int tex_height = texture_->height();

  height_ = Max(height_, height);
  glyph_info->texture_ = texture_;

  int v = 0;
  glyph_info->tex_coords_[v++] = x_/float(tex_width);
  glyph_info->tex_coords_[v++] = (y_+height)/float(tex_height);
  glyph_info->tex_coords_[v++] = (x_+width)/float(tex_width);
  glyph_info->tex_coords_[v++] = (y_+height)/float(tex_height);
  glyph_info->tex_coords_[v++] = (x_+width)/float(tex_width);
  glyph_info->tex_coords_[v++] = y_/float(tex_height);
  glyph_info->tex_coords_[v++] = x_/float(tex_width);
  glyph_info->tex_coords_[v++] = y_/float(tex_height);

  Nrrd *nrrd = texture_->nrrd_handle_->nrrd_;
  unsigned char *data = (unsigned char *)nrrd->data;

  // render glyph to texture data
  int pos;
  for (int y = 0; y < height; ++y) {
    int Y = y_+y;
    if (Y < 0 || Y >= tex_height) continue;
    for (int x = 0; x < width; ++x) {
      int X = x_+x;
      if (X < 0 || X >= tex_width) continue;
      if (!(X>=0 && X < nrrd->axis[1].size &&
           Y>=0 && Y < nrrd->axis[2].size)) {
        cerr << "X: " << X
             << "Y: " << Y
             << "A1: " << nrrd->axis[1].size 
             << "A2: " << nrrd->axis[2].size;
      }
          
      data[Y*tex_width+X] =
        bitmap_glyph->bitmap.buffer[y*Abs(bitmap_glyph->bitmap.pitch)+x];
    }
  }

  x_ += width;
  FT_Done_Glyph(glyph);
  texture_->set_dirty();
  return glyph_info;
#else
  return 0;
#endif
}

}
