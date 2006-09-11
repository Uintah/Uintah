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

#ifdef HAVE_FREETYPE
#include FT_MODULE_ERRORS_H
#endif

#ifdef min
#undef min
#undef max
#endif


namespace SCIRun {  


TextRenderer::TextRenderer(FreeTypeFace *face) :
  face_(face),
  glyphs_(),
  shadow_offset_(make_pair(1,-1)),
  cursor_position_(0),
  textures_(),
  texture_(0),
  x_(0),
  y_(0),
  height_(0)
{
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
    if (glyphs_[text[c]]) 
      hei = Max(hei, (int)(glyphs_[text[c]]->ft_metrics_.horiBearingY));
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
                          int &flags) 
{
#ifdef HAVE_FREETYPE
  unsigned int passes = 1;

  if (flags & SHADOW) {
    passes = 2;
  }

  if (flags & EXTRUDED) {
    passes = Max(Abs(shadow_offset_.first), 
                 Abs(shadow_offset_.second))+1;
  }

  unsigned total_glyphs = text.size()*passes ; 
  if (layout_.size() < total_glyphs+1) {
    layout_.resize(total_glyphs+1);
  }

  Vector left_pt = left / 64.0;
  Vector up_pt = up / 64.0;
  Vector nextline = (face_->ft_face_->size->metrics.height + passes*64)*up_pt;

  Vector cursor_up = (face_->ft_face_->size->metrics.ascender)*up_pt;
  Vector cursor_down = (face_->ft_face_->size->metrics.descender)*up_pt;

  float delta_color[4] = {0.0, 0.0, 0.0, 0.0};
  Vector delta_position(0.0, 0.0, 0.0);
  if (passes > 1) {
    delta_position = (shadow_offset_.first * left + 
                      shadow_offset_.second * up) / (passes - 1);
    for (int rgba = 0; rgba < 4; ++rgba) {
      delta_color[rgba] = (color_[rgba] - shadow_color_[rgba]) / (passes - 1);
    }
  }

  BBox bbox;
  bool disable_cursor_flag = true;
  bool do_cursor_layout = false;
  Point cursor_ll;
  
  for (unsigned int pass = 0; pass < passes; ++pass) {
    Point position = (pass * delta_position).asPoint();
    int linenum = 0;
    for (unsigned int c = 0; c < text.size(); ++c) {
      ASSERT(pass*text.size()+c < layout_.size());
      LayoutInfo &layout = layout_[pass*text.size()+c];

      layout.glyph_info_ = 0;
      if (!(flags & VERTICAL) && text[c] == '\n') {
        position = (pass * delta_position - nextline * (++linenum)).asPoint();
        continue;
      } 

      if (!glyphs_[text[c]]) { 
        layout.glyph_info_ = glyphs_[' '];
      } else {
        layout.glyph_info_ = glyphs_[text[c]];
      }

      if (!layout.glyph_info_) {
        continue;
      }

      for (int rgba = 0; rgba < 4; ++rgba) {
        if (bool(flags & REVERSE) != bool(passes == 1)) {
          layout.color_[rgba] = color_[rgba] - pass * delta_color[rgba];
        } else {
          layout.color_[rgba] = shadow_color_[rgba] + pass * delta_color[rgba];
        }
      }

      FT_Glyph_Metrics &metrics = layout.glyph_info_->ft_metrics_;

      Point ll;
      if (!(flags & VERTICAL)) {
        if (c && face_->has_kerning_p()) {
          FT_Vector kerning; 
          FT_Get_Kerning(face_->ft_face_, 
                         layout_[c-1].glyph_info_->index_,
                         layout.glyph_info_->index_, 
                         FT_KERNING_DEFAULT, &kerning); 
          position += left_pt * kerning.x;
          // position += up_pt * kerning.y;
        }


        
        ll = (position + 
              up_pt * (metrics.horiBearingY - metrics.height) +
              left_pt * metrics.horiBearingX);


        if (pass == (passes - 1)) {
          // expand bbox by a a phantom cursor 
          // This fixes the layout from jumping around when 
          // using different strings and anchor combinations
          // ie, the string 'gpq' with its descenders
          // will have a different bounding box than 'ABC'... 
          // creating a phantom cursor that has a descender and ascender
          // at least as large as any glyph in the font fixes this.
          
          if (c == (text.size()-1)) {
            Point temp = position + 
              left_pt * (metrics.horiAdvance);
            //              left_pt * (metrics.horiBearingX + metrics.width);
            bbox.extend(temp + cursor_down);
            bbox.extend(temp + 2*left + cursor_down);
            bbox.extend(temp + 2*left + cursor_up);
            bbox.extend(temp + cursor_up);
          }
            
              
            
          // Cursor is in current layout position
          if (c == cursor_position_) {
            do_cursor_layout = true;
            cursor_ll = position + left_pt * metrics.horiBearingX;
            if (c) 
              cursor_ll = cursor_ll - left;
          }

          // Cursor is at end of string
          if (cursor_position_ == text.size() && (c == text.size()-1)) {
            do_cursor_layout = true;
            cursor_ll = position + 
              left_pt * (metrics.horiAdvance);
            //              left_pt * (metrics.horiBearingX + metrics.width);
          }
                         
        }


        position = position + left_pt * metrics.horiAdvance;
      } else {
        int halfwidth = (metrics.width) >> 1;
        int advance = (64 + (int)metrics.height);
        ll = (position - up_pt * advance - left_pt * halfwidth);
        position = position - up_pt * advance;
      }

      layout.vertices_[0] = ll;
      layout.vertices_[1] = ll + left_pt * metrics.width;
      layout.vertices_[2] = (ll + left_pt * metrics.width + 
                             up_pt * metrics.height);
      layout.vertices_[3] = ll + up_pt * metrics.height;

      for (int v = 0; v < 4; ++v) {
        bbox.extend(layout.vertices_[v]);
      }
    }
  }

  if (!(flags & VERTICAL) && 
      !do_cursor_layout && !cursor_position_) {

    cursor_ll = Point(0.0, 0.0, 0.0);
    do_cursor_layout = true;
  }

  if (do_cursor_layout) {
    // Cursor layout is now valid, dont turn off the cursor flag
    disable_cursor_flag = false; 
    layout_.back().vertices_[0] = cursor_ll + cursor_down;
    layout_.back().vertices_[1] = cursor_ll + 2 * left + cursor_down ;
    layout_.back().vertices_[2] = cursor_ll + 2 * left + cursor_up;
    layout_.back().vertices_[3] = cursor_ll + cursor_up;
    for (int v = 0; v < 4; ++v) {
      layout_.back().color_[v] = color_[v];
      bbox.extend(layout_.back().vertices_[v]);
    }
  }


  if (disable_cursor_flag) {
    flags &= ~CURSOR;
  }

  if (!bbox.valid()) {
    return 0;
  }
  Vector width = bbox.diagonal()*left;
  Vector height = bbox.diagonal()*up;

  offset_ = Vector(0,0,0);
  switch (flags & ANCHOR_MASK) {
  case N:  offset_ = - width/2.0;                break;
  case E:  offset_ = - width       + height/2.0; break;
  case S:  offset_ = - width/2.0   + height;     break;
  case W:  offset_ =   height/2.0;               break;
  case NE: offset_ = - width;                    break;
  case SE: offset_ = - width       + height;     break;
  case SW: offset_ =   height;                   break;
  case C:  offset_ = - width/2.0   + height/2.0; break;
  default:
  case NW: break;
  }

  Vector fix = bbox.min().x()*left + bbox.max().y()*up;
  offset_ = offset_ + anchor.asVector() - fix;

  for (unsigned int c = 0; c < total_glyphs; ++c) 
    for (int v = 0; v < 4; ++v)
      for (int o = 0; o < 3; ++o)
        layout_[c].vertices_[v](o) = Floor(layout_[c].vertices_[v](o));
        
  return total_glyphs;
#else
  return 0;
#endif
}



void
TextRenderer::render(const string &text, float x, float y, int flags)
{
#ifdef HAVE_FREETYPE
  render_string_glyphs_to_texture(text);
  CHECK_OPENGL_ERROR();
  if (!texture_) return;


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
  
  GLint gl_viewport[4];
  glGetIntegerv(GL_VIEWPORT, gl_viewport);
  float vw = gl_viewport[2];
  float vh = gl_viewport[3];
  glScaled(1/vw, 1/vh, 1.0);

  glDisable(GL_CULL_FACE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glShadeModel(GL_FLAT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  CHECK_OPENGL_ERROR();

  Point anchor(x,y,0.0);
  Vector left(1,0,0);
  Vector up(0,1,0);
  unsigned int num = layout_text(text, anchor, left, up, flags);
  glTranslated(Round(offset_.x()), Round(offset_.y()), Round(offset_.z()));
  texture_->bind();
  for (unsigned int c = 0; c < num; ++c) {
    LayoutInfo &layout = layout_[c];
    GlyphInfo *glyph = layout.glyph_info_;
    //    ASSERT(glyph);
    if (!glyph) {
      continue;
    }
    glyph->texture_->set_color(layout.color_);
    glyph->texture_->draw(4, layout.vertices_, glyph->tex_coords_);
  }

  glDisable(GL_TEXTURE_2D);

  if ((flags & CURSOR) && layout_.size()) {
    glColor4fv(layout_.back().color_);
    //    glColor4f(1.0, 0.0, 0.0, 1.0);
    glBegin(GL_QUADS);
    for (int v = 0; v < 4; ++v) {
      glVertex3d(layout_.back().vertices_[v](0),
                 layout_.back().vertices_[v](1),
                 layout_.back().vertices_[v](2));
    }
    glEnd();
  }
               
               
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  CHECK_OPENGL_ERROR();
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
  //  int pos;
  for (int y = 0; y < height; ++y) {
    int Y = y_+y;
    if (Y < 0 || Y >= tex_height) continue;
    for (int x = 0; x < width; ++x) {
      int X = x_+x;
      if (X < 0 || X >= tex_width) continue;
      if (!(X>=0 && X < int(nrrd->axis[1].size) &&
           Y>=0 && Y < int(nrrd->axis[2].size))) {
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


void
TextRenderer::set_cursor_position(unsigned int pos) {
  cursor_position_ = pos;
}

}
