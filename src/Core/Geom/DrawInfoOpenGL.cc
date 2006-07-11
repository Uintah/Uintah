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
 *  DrawInfoOpenGL.cc: OpenGL State Machine
 *
 *  Written by:
 *   McKay Davis
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   November, 2005
 *
 *  Copyright (C) 2005 Scientific Computing and Imaging Institute
 */

#include <sci_defs/ogl_defs.h>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <Core/Geom/DrawInfoOpenGL.h>
#include <Core/Geom/GeomDL.h>
#include <Core/Geom/Material.h>
#include <Core/Math/MiscMath.h>
#include <iostream>

#ifndef _WIN32
#include <X11/X.h>
#include <X11/Xlib.h>
#else
#include <windows.h>
#endif


using std::map;
using std::cerr;
using std::endl;
using std::pair;

namespace SCIRun {

static void
quad_error(GLenum code)
{
  cerr << "WARNING: Quadric Error (" << (char*)gluErrorString(code) << ")" << endl;
}


// May need to do this for really old GCC compilers?
//typedef void (*gluQuadricCallbackType)(...);
typedef GLvoid (*gluQuadricCallbackType)(...);

DrawInfoOpenGL::DrawInfoOpenGL() :
  polycount_(0),
  lighting_(1),
  show_bbox_(0),
  currently_lit_(1),
  pickmode_(1),
  pickchild_(0),
  npicks_(0),
  fog_(0),
  cull_(0),
  display_list_p_(0),
  mouse_action_(0),
  check_clip_(1),
  clip_planes_(0),
  current_matl_(0),
  ignore_matl_(0),
  qobj_(0),
  axis_(0),
  dir_(0),
  ambient_scale_(1.0),
  diffuse_scale_(1.0),
  specular_scale_(1.0),
  emission_scale_(1.0),
  shininess_scale_(1.0),
  point_size_(1.0),
  line_width_(1.0),
  polygon_offset_factor_(0.0),
  polygon_offset_units_(0.0),
  using_cmtexture_(0),
  cmtexture_(0),
  drawtype_(Gouraud)
{
  for (int i=0; i < GEOM_FONT_COUNT; i++)
  {
    fontstatus_[i] = 0;
    fontbase_[i] = 0;
  }

  qobj_=gluNewQuadric();

  if ( !qobj_ )
  {
    printf( "Error in GeomOpenGL.cc: DrawInfoOpenGL(): gluNewQuadric()\n" );
  }

#ifdef _WIN32
  gluQuadricCallback(qobj_, /* FIX (GLenum)GLU_ERROR*/ 0, (void (__stdcall*)())quad_error);
#else
  gluQuadricCallback(qobj_, (GLenum)GLU_ERROR, (gluQuadricCallbackType)quad_error);
#endif
}


void
DrawInfoOpenGL::reset()
{
  polycount_ = 0;
  current_matl_ = 0;
  ignore_matl_ = 0;
  fog_ = 0;
  cull_ = 0;
  check_clip_ = 0;
  pickmode_ = 0;
  pickchild_ = 0;
  npicks_ = 0;
}


DrawInfoOpenGL::~DrawInfoOpenGL()
{
  map<GeomDL *, pair<unsigned int, unsigned int> >::iterator loc;
  loc = dl_map_.begin();
  while (loc != dl_map_.end())
  {
    (*loc).first->dl_unregister(this);
    ++loc;
  }
}


DrawInfoOpenGL::DrawType
DrawInfoOpenGL::get_drawtype()
{
  return drawtype_;
}

void
DrawInfoOpenGL::set_drawtype(DrawType dt)
{
  drawtype_ = dt;
  switch(drawtype_)
  {
  case DrawInfoOpenGL::WireFrame:
    gluQuadricDrawStyle(qobj_, (GLenum)GLU_LINE);
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    break;
  case DrawInfoOpenGL::Flat:
    gluQuadricDrawStyle(qobj_, (GLenum)GLU_FILL);
    glShadeModel(GL_FLAT);
    glPolygonMode(GL_FRONT_AND_BACK,(GLenum)GL_FILL);
    break;
  case DrawInfoOpenGL::Gouraud:
    gluQuadricDrawStyle(qobj_, (GLenum)GLU_FILL);
    glShadeModel(GL_SMOOTH);
    glPolygonMode(GL_FRONT_AND_BACK,(GLenum)GL_FILL);
    break;
  }
}


void
DrawInfoOpenGL::init_lighting(int use_light)
{
  if (use_light)
  {
    glEnable(GL_LIGHTING);
    switch(drawtype_)
    {
    case DrawInfoOpenGL::WireFrame:
      gluQuadricNormals(qobj_, (GLenum)GLU_SMOOTH);
      break;
    case DrawInfoOpenGL::Flat:
      gluQuadricNormals(qobj_, (GLenum)GLU_FLAT);
      break;
    case DrawInfoOpenGL::Gouraud:
      gluQuadricNormals(qobj_, (GLenum)GLU_SMOOTH);
      break;
    }
  }
  else
  {
    glDisable(GL_LIGHTING);
    gluQuadricNormals(qobj_,(GLenum)GLU_NONE);
  }
  if (fog_)
    glEnable(GL_FOG);
  else
    glDisable(GL_FOG);
  if (cull_)
    glEnable(GL_CULL_FACE);
  else
    glDisable(GL_CULL_FACE);
}



void
DrawInfoOpenGL::init_clip(void)
{
  for (int num = 0; num < 6; num++) {
    GLdouble plane[4];
    planes_[num].get(plane); 
    glClipPlane((GLenum)(GL_CLIP_PLANE0+num),plane); 
    if (check_clip_ && clip_planes_ & (1 << num))
      glEnable((GLenum)(GL_CLIP_PLANE0+num));
    else 
      glDisable((GLenum)(GL_CLIP_PLANE0+num));
  }
}


void
DrawInfoOpenGL::set_material(Material* matl)
{
  if (matl==current_matl_ || ignore_matl_)
  {
    return;     
  }
  float color[4];
  (matl->ambient*ambient_scale_).get_color(color);
  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
  (matl->diffuse*diffuse_scale_).get_color(color);
  if (matl->transparency < 1.0)
  {
    color[3] = matl->transparency * matl->transparency;
    color[3] *= color[3];
  }
  glColor4fv(color);
  (matl->specular*specular_scale_).get_color(color);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);
  (matl->emission*emission_scale_).get_color(color);
  glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, color);
  if (!current_matl_ || matl->shininess != current_matl_->shininess)
  {
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, matl->shininess*shininess_scale_);
  }     
  current_matl_=matl;
}


bool
DrawInfoOpenGL::dl_lookup(GeomDL *obj, unsigned int &state, unsigned int &dl)
{
  map<GeomDL *, pair<unsigned int, unsigned int> >::iterator loc;
  loc = dl_map_.find(obj);
  if (loc != dl_map_.end())
  {
    dl = (*loc).second.first;
    state = (*loc).second.second;
    return true;
  }
  return false;
}


bool
DrawInfoOpenGL::dl_addnew(GeomDL *obj, unsigned int state, unsigned int &dl)
{
  if (!dl_freelist_.empty())
  {
    dl = dl_freelist_.front();
    dl_freelist_.pop_front();
  }
  else
  {
    dl = glGenLists(1);
  }
  if (dl)
  {
    dl_map_[obj] = pair<unsigned int, unsigned int>(dl, state);
    obj->dl_register(this);
    return true;
  }
  return false;
}


bool
DrawInfoOpenGL::dl_update(GeomDL *obj, unsigned int state)
{
  map<GeomDL *, pair<unsigned int, unsigned int> >::iterator loc;
  loc = dl_map_.find(obj);
  if (loc != dl_map_.end())
  {
    (*loc).second.second = state;
    return true;
  }
  return false;
}


bool
DrawInfoOpenGL::dl_remove(GeomDL *obj)
{
  map<GeomDL *, pair<unsigned int, unsigned int> >::iterator loc;
  loc = dl_map_.find(obj);
  if (loc != dl_map_.end())
  {
    dl_freelist_.push_front((*loc).second.first);
    dl_map_.erase(loc);
    return true;
  }
  return false;
}


// this is for transparent rendering stuff.

void
DrawInfoOpenGL::init_view( double /*znear*/, double /*zfar*/,
                           Point& /*eyep*/, Point& /*lookat*/ )
{
  double model_mat[16]; // this is the modelview matrix

  glGetDoublev(GL_MODELVIEW_MATRIX,model_mat);

  // this is what you rip the view vector from
  // just use the "Z" axis, normalized
  view_ = Vector(model_mat[0*4+2],model_mat[1*4+2],model_mat[2*4+2]);

  view_.normalize();

  // 0 is X, 1 is Y, 2 is Z
  dir_ = 1;

  if (Abs(view_.x()) > Abs(view_.y()))
  {
    if (Abs(view_.x()) > Abs(view_.z()))
    { // use x dir
      axis_=0;
      if (view_.x() < 0)
      {
        dir_=-1;
      }
    }
    else
    { // use z dir
      axis_=2;
      if (view_.z() < 0)
      {
        dir_=-1;
      }
    }
  }
  else if (Abs(view_.y()) > Abs(view_.z()))
  { // y greates
    axis_=1;
    if (view_.y() < 0)
    {
      dir_=-1;
    }
  }
  else
  { // z is the one
    axis_=2;
    if (view_.z() < 0)
    {
      dir_=-1;
    }
  }
}


bool
DrawInfoOpenGL::init_font(int a)
{
#ifndef _WIN32
  if (a > GEOM_FONT_COUNT || a < 0) { return false; }

  if ( fontstatus_[a] == 0 )
  {
    Display *dpy = XOpenDisplay( NULL );

    static const char *fontname[GEOM_FONT_COUNT] = {
      "-schumacher-clean-medium-r-normal-*-*-60-*-*-*-*-*",
      "-schumacher-clean-bold-r-normal-*-*-100-*-*-*-*-*",
      "-schumacher-clean-bold-r-normal-*-*-140-*-*-*-*-*",
      "-*-courier-bold-r-normal-*-*-180-*-*-*-*-*",
      "-*-courier-bold-r-normal-*-*-240-*-*-*-*-*"
    };

    XFontStruct* fontInfo = XLoadQueryFont(dpy, fontname[a]);
    if (fontInfo == NULL)
    {
      cerr << "DrawInfoOpenGL::init_font: font '" << fontname[a]
           << "' not found.\n";
      fontstatus_[a] = 2;
      return false;
    }
    Font id = fontInfo->fid;
    unsigned int first = fontInfo->min_char_or_byte2;
    unsigned int last = fontInfo->max_char_or_byte2;

    fontbase_[a] = glGenLists((GLuint) last+1);

    if (fontbase_[a] == 0)
    {
      cerr << "DrawInfoOpenGL::init_font: Out of display lists.\n";
      fontstatus_[a] = 2;
      return false;
    }
    glXUseXFont(id, first, last-first+1, fontbase_[a]+first);
    fontstatus_[a] = 1;
  }
  if (fontstatus_[a] == 1)
  {
    return true;
  }
  return false;

#else // WIN32
  if (a > GEOM_FONT_COUNT || a < 0) { return false; }

  if ( fontstatus_[a] == 0 )
  {
    HDC hDC = wglGetCurrentDC();
    if (!hDC)
      return false;

    DWORD first, count;

    // for now, just use the system font
    SelectObject(hDC,GetStockObject(SYSTEM_FONT));

    // rasterize the standard character set.
    first = 0;
    count = 256;
    fontbase_[a] =  glGenLists(count);

    if (fontbase_[a] == 0)
    {
      cerr << "DrawInfoOpenGL::init_font: Out of display lists.\n";
      fontstatus_[a] = 2;
      return false;
    }
    wglUseFontBitmaps( hDC, first, count, fontbase_[a]+first );
    fontstatus_[a] = 1;
  }
  if (fontstatus_[a] == 1)
  {
    return true;
  }
  return false;
#endif
}

}
