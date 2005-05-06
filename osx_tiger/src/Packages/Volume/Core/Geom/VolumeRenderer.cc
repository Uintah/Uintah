//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : VolumeRenderer.cc
//    Author : Milan Ikits
//    Date   : Thu Jul  8 00:04:15 2004

#include <string>
#include <iostream>
#include <Core/Geom/GeomOpenGL.h>
#include <Packages/Volume/Core/Geom/VolumeRenderer.h>
#include <Packages/Volume/Core/Geom/VolShader.h>
#include <Packages/Volume/Core/Util/ShaderProgramARB.h>
#include <Packages/Volume/Core/Util/Pbuffer.h>
#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Core/Util/DebugStream.h>

using std::cerr;
using std::endl;
using std::string;
using SCIRun::DrawInfoOpenGL;

namespace Volume {

//static SCIRun::DebugStream dbg("VolumeRenderer", false);

VolumeRenderer::VolumeRenderer(TextureHandle tex,
                               ColorMapHandle cmap1, Colormap2Handle cmap2,
                               int tex_mem):
  TextureRenderer(tex, cmap1, cmap2, tex_mem),
  shading_(false),
  ambient_(0.5),
  diffuse_(0.5),
  specular_(0.0),
  shine_(30.0),
  light_(0),
  adaptive_(true)
{
  mode_ = MODE_OVER;
}

VolumeRenderer::VolumeRenderer(const VolumeRenderer& copy):
  TextureRenderer(copy),
  shading_(copy.shading_),
  ambient_(copy.ambient_),
  diffuse_(copy.diffuse_),
  specular_(copy.specular_),
  shine_(copy.shine_),
  light_(copy.light_),
  adaptive_(copy.adaptive_)
{}

VolumeRenderer::~VolumeRenderer()
{}

GeomObj*
VolumeRenderer::clone()
{
  return scinew VolumeRenderer(*this);
}

void
VolumeRenderer::set_mode(RenderMode mode)
{
  if(mode_ != mode) {
    mode_ = mode;
    alpha_dirty_ = true;
  }
}

void
VolumeRenderer::set_sampling_rate(double rate)
{
  if(sampling_rate_ != rate) {
    sampling_rate_ = rate;
    alpha_dirty_ = true;
  }
}

void
VolumeRenderer::set_interactive_rate(double rate)
{
  if(irate_ != rate) {
    irate_ = rate;
    alpha_dirty_ = true;
  }
}

void
VolumeRenderer::set_interactive_mode(bool mode)
{
  if(imode_ != mode) {
    imode_ = mode;
    alpha_dirty_ = true;
  }
}

void
VolumeRenderer::set_adaptive(bool b)
{
  adaptive_ = b;
}
    
#ifdef SCI_OPENGL
void
VolumeRenderer::draw(DrawInfoOpenGL* di, Material* mat, double)
{
  if(!pre_draw(di, mat, shading_)) return;
  mutex_.lock();
  di_ = di;
  if(di->get_drawtype() == DrawInfoOpenGL::WireFrame ) {
    draw_wireframe();
  } else {
    draw();
  }
  di_ = 0;
  mutex_.unlock();
}

void
VolumeRenderer::draw()
{
  tex_->lock_bricks();
  
  Ray view_ray = compute_view();
  vector<Brick*> bricks;
  tex_->get_sorted_bricks(bricks, view_ray);
  if(bricks.size() == 0) return;

  if(adaptive_ && ((cmap2_.get_rep() && cmap2_->updating()) || di_->mouse_action))
    set_interactive_mode(true);
  else
    set_interactive_mode(false);

  double rate = imode_ ? irate_ : sampling_rate_;
  Vector diag = tex_->bbox().diagonal();
  Vector cell_diag(diag.x()/tex_->nx(), diag.y()/tex_->ny(), diag.z()/tex_->nz());
  double dt = cell_diag.length()/rate;
  int num_slices = (int)(diag.length()/dt);
  
  Array1<float> vertex(0, 100, num_slices*6);
  Array1<float> texcoord(0, 100, num_slices*6);
  Array1<int> size(0, 100, num_slices*6);
  
  //--------------------------------------------------------------------------

  int nc = bricks[0]->nc();
  int nb0 = bricks[0]->nb(0);
  bool use_cmap2 = cmap2_.get_rep() && nc == 2;
  bool use_shading = shading_ && nb0 == 4;
  GLboolean use_fog = glIsEnabled(GL_FOG);
  // glGetBooleanv(GL_FOG, &use_fog);
  GLfloat light_pos[4];
  glGetLightfv(GL_LIGHT0+light_, GL_POSITION, light_pos);
  GLfloat clear_color[4];
  glGetFloatv(GL_COLOR_CLEAR_VALUE, clear_color);
  int vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);

  //--------------------------------------------------------------------------
  // set up blending

  int psize[2];
  psize[0] = NextPowerOf2(vp[2]);
  psize[1] = NextPowerOf2(vp[3]);
    
  if(blend_num_bits_ != 8) {
    if(!blend_buffer_ || blend_num_bits_ != blend_buffer_->num_color_bits()
       || psize[0] != blend_buffer_->width()
       || psize[1] != blend_buffer_->height()) {
      blend_buffer_ = new Pbuffer(psize[0], psize[1], GL_FLOAT, blend_num_bits_, true,
                                  GL_FALSE, GL_DONT_CARE, 24);
      if(blend_buffer_->create()) {
        blend_buffer_->destroy();
        delete blend_buffer_;
        blend_buffer_ = 0;
        blend_num_bits_ = 8;
        use_blend_buffer_ = false;
      } else {
        blend_buffer_->set_use_default_shader(false);
        blend_buffer_->set_use_texture_matrix(false);
      }
    }
  }
  
  if(blend_num_bits_ == 8) {
    glEnable(GL_BLEND);
    switch(mode_) {
    case MODE_OVER:
      glBlendEquation(GL_FUNC_ADD);
      glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
      break;
    case MODE_MIP:
      glBlendEquation(GL_MAX);
      glBlendFunc(GL_ONE, GL_ONE);
      break;
    default:
      break;
    }
  } else {
    double mv[16], pr[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, mv);
    glGetDoublev(GL_PROJECTION_MATRIX, pr);

    blend_buffer_->activate();
    glDrawBuffer(GL_FRONT);
    float* cc = clear_color;
    glClearColor(cc[0], cc[1], cc[2], cc[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    blend_buffer_->swapBuffers();

    glViewport(vp[0], vp[1], vp[2], vp[3]);
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd(pr);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(mv);
  }
  
  glColor4f(1.0, 1.0, 1.0, 1.0);
  glDepthMask(GL_FALSE);

  //--------------------------------------------------------------------------
  // load colormap texture
  if(use_cmap2) {
    // rebuild if needed
    build_colormap2();
    bind_colormap2();
  } else {
    // rebuild if needed
    build_colormap1();
    bind_colormap1();
  }
  
  //--------------------------------------------------------------------------
  // enable data texture unit 0
  glActiveTexture(GL_TEXTURE0_ARB);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glEnable(GL_TEXTURE_3D);

  //--------------------------------------------------------------------------
  // set up shaders
  FragmentProgramARB* shader = 0;
  int blend_mode = 0;
  if(blend_num_bits_ != 8) {
    if(mode_ == MODE_OVER) {
      if(blend_buffer_->need_shader()) {
        blend_mode = 1;
      } else {
        blend_mode = 3;
      }
    } else {
      if(blend_buffer_->need_shader()) {
        blend_mode = 2;
      } else {
        blend_mode = 4;
      }
    }
  }
  shader = vol_shader_factory_->shader(use_cmap2 ? 2 : 1, nb0, use_shading, false,
                                       use_fog, blend_mode);
  if(shader) {
    if(!shader->valid()) {
      shader->create();
    }
    shader->bind();
  }
  
  if(use_shading) {
    // set shader parameters
    Vector l(light_pos[0], light_pos[1], light_pos[2]);
    //cerr << "LIGHTING: " << pos << endl;
    double m[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, m);
    Transform mv;
    mv.set_trans(m);
    Transform t = tex_->transform();
    l = mv.unproject(l);
    l = t.unproject(l);
    shader->setLocalParam(0, l.x(), l.y(), l.z(), 1.0);
    shader->setLocalParam(1, ambient_, diffuse_, specular_, shine_);
  }
  
  //--------------------------------------------------------------------------
  // render bricks

  Transform tform = tex_->transform();
  double mvmat[16];
  tform.get_trans(mvmat);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);
  
  for(unsigned int i=0; i<bricks.size(); i++) {
    Brick* b = bricks[i];
    load_brick(b);
    vertex.resize(0);
    texcoord.resize(0);
    size.resize(0);
    b->compute_polygons(view_ray, dt, vertex, texcoord, size);
    draw_polygons(vertex, texcoord, size, false, use_fog,
                  blend_num_bits_ > 8 ? blend_buffer_ : 0);
  }

  glPopMatrix();
  
  glDepthMask(GL_TRUE);

  //--------------------------------------------------------------------------
  // release shader

  if(shader && shader->valid())
    shader->release();
  
  //--------------------------------------------------------------------------
  // release textures
  if(use_cmap2) {
    release_colormap2();
  } else {
    release_colormap1();
  }
  glActiveTexture(GL_TEXTURE0_ARB);
  glDisable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, 0);

  //--------------------------------------------------------------------------

  if(blend_num_bits_ == 8) {
    glDisable(GL_BLEND);
  } else {
    blend_buffer_->deactivate();
    blend_buffer_->set_use_default_shader(true);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glTranslatef(-1.0, -1.0, 0.0);
    glScalef(2.0, 2.0, 2.0);

    GLboolean depth_test = glIsEnabled(GL_DEPTH_TEST);
    GLboolean lighting = glIsEnabled(GL_LIGHTING);
    GLboolean cull_face = glIsEnabled(GL_CULL_FACE);
    
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    
    glActiveTexture(GL_TEXTURE0);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    blend_buffer_->bind(GL_FRONT);

    glBegin(GL_QUADS);
    {
      glTexCoord2f(0.0,  0.0);
      glVertex2f( 0.0,  0.0);
      glTexCoord2f(vp[2],  0.0);
      glVertex2f( 1.0,  0.0);
      glTexCoord2f(vp[2],  vp[3]);
      glVertex2f( 1.0,  1.0);
      glTexCoord2f( 0.0,  vp[3]);
      glVertex2f( 0.0,  1.0);
    }
    glEnd();
    
    blend_buffer_->release(GL_FRONT);

    if(depth_test) glEnable(GL_DEPTH_TEST);
    if(lighting) glEnable(GL_LIGHTING);
    if(cull_face) glEnable(GL_CULL_FACE);
    
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    blend_buffer_->set_use_default_shader(false);
  }
  // Look for errors
  GLenum errcode;
  if((errcode=glGetError()) != GL_NO_ERROR) {
    cerr << "VolumeRenderer::end | "
         << (char*)gluErrorString(errcode) << "\n";
  }
  tex_->unlock_bricks();
}

void
VolumeRenderer::draw_wireframe()
{
  tex_->lock_bricks();
  Ray view_ray = compute_view();
  Transform tform = tex_->transform();
  double mvmat[16];
  tform.get_trans(mvmat);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);
  glEnable(GL_DEPTH_TEST);
  GLboolean lighting = glIsEnabled(GL_LIGHTING);
  glDisable(GL_LIGHTING);
  glColor4f(0.8, 0.8, 0.8, 1.0);
  vector<Brick*> bricks;
  tex_->get_sorted_bricks(bricks, view_ray);
  for(unsigned int i=0; i<bricks.size(); i++) {
    Brick& brick = *bricks[i];
    glBegin(GL_LINES);
    {
      for(int i=0; i<4; i++) {
        glVertex3d(brick[i].x(), brick[i].y(), brick[i].z());
        glVertex3d(brick[i+4].x(), brick[i+4].y(), brick[i+4].z());
      }
    }
    glEnd();
    glBegin(GL_LINE_LOOP);
    {
      glVertex3d(brick[0].x(), brick[0].y(), brick[0].z());
      glVertex3d(brick[1].x(), brick[1].y(), brick[1].z());
      glVertex3d(brick[3].x(), brick[3].y(), brick[3].z());
      glVertex3d(brick[2].x(), brick[2].y(), brick[2].z());
    }
    glEnd();
    glBegin(GL_LINE_LOOP);
    {
      glVertex3d(brick[4].x(), brick[4].y(), brick[4].z());
      glVertex3d(brick[5].x(), brick[5].y(), brick[5].z());
      glVertex3d(brick[7].x(), brick[7].y(), brick[7].z());
      glVertex3d(brick[6].x(), brick[6].y(), brick[6].z());
    }
    glEnd();
  }
  if(lighting) glEnable(GL_LIGHTING);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  tex_->unlock_bricks();
}

#endif // SCI_OPENGL

} // namespace Volume
