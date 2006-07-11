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

#include <sci_defs/ogl_defs.h>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <Core/Geom/DrawInfoOpenGL.h>
#include <Core/Volume/VolumeRenderer.h>
#include <Core/Volume/VolShader.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Geom/Pbuffer.h>
#include <Core/Volume/TextureBrick.h>
#include <Core/Util/DebugStream.h>
#include <Core/Math/MiscMath.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using std::endl;
using std::string;

namespace SCIRun {

#ifdef _WIN32
#undef min
#undef max
#endif


struct cmap_data {
public:
  cmap_data() : tex_id_(0), dirty_(true), alpha_dirty_(true) {}
  Array2<float>  data_;
  unsigned int tex_id_;
  bool dirty_;
  bool alpha_dirty_;
};

//static SCIRun::DebugStream dbg("VolumeRenderer", false);

VolumeRenderer::VolumeRenderer(TextureHandle tex,
                               ColorMapHandle cmap1,
                               vector<ColorMap2Handle> &cmap2,
                               vector<Plane *> &planes,
                               int tex_mem):
  TextureRenderer(tex, cmap1, cmap2, tex_mem),
  grange_(1.0),
  goffset_(0.0),
  shading_(false),
  ambient_(0.5),
  diffuse_(0.5),
  specular_(0.0),
  shine_(30.0),
  light_(0),
  adaptive_(true),
  draw_level_(20),
  level_alpha_(20),
  planes_(planes)
{
  mode_ = MODE_OVER;
  vector<bool>::iterator it = draw_level_.begin();
  for(;it != draw_level_.end(); ++it){
    (*it) = false;
  }
  vector< double >::iterator it2 = level_alpha_.begin();
  for(;it2 != level_alpha_.end(); ++it2){
    (*it2) = 0;
  }
}

VolumeRenderer::VolumeRenderer(const VolumeRenderer& copy):
  TextureRenderer(copy),
  shading_(copy.shading_),
  ambient_(copy.ambient_),
  diffuse_(copy.diffuse_),
  specular_(copy.specular_),
  shine_(copy.shine_),
  light_(copy.light_),
  adaptive_(copy.adaptive_),
  draw_level_(copy.draw_level_),
  level_alpha_(copy.level_alpha_),
  planes_(copy.planes_)
{
}


VolumeRenderer::~VolumeRenderer()
{
}


GeomObj*
VolumeRenderer::clone()
{
  return new VolumeRenderer(*this);
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
  if(di->get_drawtype() == DrawInfoOpenGL::WireFrame )
  {
    draw_wireframe();
  }
  else
  {
    draw_volume();
  }
  di_ = 0;
  mutex_.unlock();
}


void
VolumeRenderer::draw_volume()
{
  tex_->lock_bricks();

  const int levels = tex_->nlevels();

  Ray view_ray = compute_view();
  vector<vector<TextureBrickHandle> > bricks(levels);
  int total_brick_size = 0;
  int firstlevel = -1;
  for (int i = 0; i < levels; i++)
  {
    tex_->get_sorted_bricks(bricks[i], view_ray, levels - i - 1);
    total_brick_size += bricks[i].size();
    if (firstlevel < 0 && bricks[i].size()) { firstlevel = i; }
  }
  if (total_brick_size == 0) {
    tex_->unlock_bricks();
    return;
  }

  bool cmap2_updating = false;
  for (unsigned int c = 0; c < cmap2_.size(); ++c)
    if (cmap2_[c]->updating()) {
      cmap2_updating = true;
      break;
    }

  set_interactive_mode(adaptive_ && (cmap2_updating || di_->mouse_action_));

  // Set sampling rate based on interaction.
  const double rate = imode_ ? irate_ : sampling_rate_;
  const Vector diag = tex_->bbox().diagonal();
  const Vector cell_diag(diag.x() / (tex_->nx() * pow(2.0, levels-1)),
                         diag.y() / (tex_->ny() * pow(2.0, levels-1)),
                         diag.z() / (tex_->nz() * pow(2.0, levels-1)));
  const double dt = cell_diag.length()/rate;
  const int num_slices = (int)(diag.length()/dt);

  vector<float> vertex;
  vector<float> texcoord;
  vector<int> size;
  vector<int> mask;
  vertex.reserve(num_slices*6);
  texcoord.reserve(num_slices*6);
  size.reserve(num_slices*6);
  mask.reserve(num_slices*6);

  //--------------------------------------------------------------------------

  const int nb0 = bricks[firstlevel][0]->nb(0);
  const bool use_cmap1 = cmap1_.get_rep();
  const bool use_cmap2 =
    cmap2_.size() && ShaderProgramARB::shaders_supported();
  if(!use_cmap1 && !use_cmap2)
  {
    tex_->unlock_bricks();
    return;
  }

  const bool use_shading = true;
  const GLboolean use_fog = glIsEnabled(GL_FOG);
  GLfloat light_pos[4];
  glGetLightfv(GL_LIGHT0+light_, GL_POSITION, light_pos);
  GLfloat clear_color[4];
  glGetFloatv(GL_COLOR_CLEAR_VALUE, clear_color);
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);

  //--------------------------------------------------------------------------
  // set up blending
  if(blend_num_bits_ == 8) {
    glEnable(GL_BLEND);
    switch(mode_) {
    case MODE_OVER:
#ifndef _WIN32
      if(gluCheckExtension((GLubyte*)"GL_ARB_imaging",glGetString(GL_EXTENSIONS)))
#else
      if (glBlendEquation)
#endif
        glBlendEquation(GL_FUNC_ADD);

      glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
      break;
    case MODE_MIP:
#ifndef _WIN32
      if(gluCheckExtension((GLubyte*)"GL_ARB_imaging",glGetString(GL_EXTENSIONS)))
#else
      if (glBlendEquation)
#endif
        glBlendEquation(GL_MAX);
      glBlendFunc(GL_ONE, GL_ONE);
      break;
    default:
      break;
    }
  } else {
    int psize[2];
    psize[0] = Pow2(vp[2]);
    psize[1] = Pow2(vp[3]);
    if(!blend_buffer_ || blend_num_bits_ != blend_buffer_->num_color_bits()
       || psize[0] != blend_buffer_->width()
       || psize[1] != blend_buffer_->height())
    {
      blend_buffer_ = new Pbuffer(psize[0], psize[1],
                                  GL_FLOAT, blend_num_bits_, true,
                                  GL_FALSE, GL_DONT_CARE, 24);
      if (!blend_buffer_->create())
      {
        blend_buffer_->destroy();
        delete blend_buffer_;
        blend_buffer_ = 0;
        blend_num_bits_ = 8;
        use_blend_buffer_ = false;
      }
      else
      {
        blend_buffer_->set_use_default_shader(false);
        blend_buffer_->set_use_texture_matrix(false);
      }
    }

    double mv[16], pr[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, mv);
    glGetDoublev(GL_PROJECTION_MATRIX, pr);

    GLfloat fstart, fend, fcolor[4];
    // Copy the fog state to the new context.
    if (use_fog)
    {
      glGetFloatv(GL_FOG_START, &fstart);
      glGetFloatv(GL_FOG_END, &fend);
      glGetFloatv(GL_FOG_COLOR, fcolor);
    }

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

    if (use_fog)
    {
      glFogi(GL_FOG_MODE, GL_LINEAR);
      glFogf(GL_FOG_START, fstart);
      glFogf(GL_FOG_END, fend);
      glFogfv(GL_FOG_COLOR, fcolor);
    }
  }

  glColor4f(1.0, 1.0, 1.0, 1.0);
  glDepthMask(GL_FALSE);

  //--------------------------------------------------------------------------
  // Load colormap texture into texture unit 2, rebuild if needed.
  vector< cmap_data * > cmaps;
  if (use_cmap2)
  {
    build_colormap2();
    bind_colormap2();
  }
  else
  {
    if (levels > 2 && mode_ == MODE_OVER)
    {
      int i;
      for (i = 0; i < levels; i++ ) {
        cmaps.push_back( new cmap_data );
      }
      for (int i = 0; i < levels; ++i )
      {
        cmaps.push_back( new cmap_data );
        build_colormap1( cmaps[levels - i - 1]->data_,
                         cmaps[levels - i - 1]->tex_id_,
                         cmaps[levels - i - 1]->dirty_,
                         cmaps[levels - i - 1]->alpha_dirty_,
                         double( invert_opacity_  ?
                                 tan(1.570796327 *
                                     (0.5 - level_alpha_[levels - i - 1])*
                                     0.49999) : i ));;
      }
    }
    else
    {
      build_colormap1(cmap1_array_, cmap1_tex_, cmap1_dirty_, alpha_dirty_);
      bind_colormap1(cmap1_array_, cmap1_tex_);
    }
  }

  //--------------------------------------------------------------------------
  // enable data texture unit 0
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#ifdef _WIN32
  if (glActiveTexture)
#endif
  glActiveTexture(GL_TEXTURE0_ARB);
#endif
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
  shader = vol_shader_factory_->shader(use_cmap2 ? 2 : 1, nb0,
                                       use_shading, false,
                                       use_fog, blend_mode, cmap2_.size());
  if (shader)
  {
    if (!shader->valid())
    {
      shader->create();
    }
    shader->bind();
  }

  if (use_shading)
  {
    // set shader parameters
    double mat[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, mat);
    Transform mv;
    mv.set_trans(mat);
    const Transform &t = tex_->transform();
    Vector light(light_pos[0], light_pos[1], light_pos[2]);
    light = mv.unproject(light);
    light = t.unproject(light);
    light.safe_normalize();
    shader->setLocalParam(0, light.x(), light.y(), light.z(), 1.0);
    shader->setLocalParam(1, ambient_, diffuse_, specular_, shine_);
  }

  //-------------------------------------------------------------------------
  // set up stenciling
  if (use_stencil_)
  {
    glStencilMask(1);
    glStencilFunc(GL_EQUAL, 0, 1);
    glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);
    glEnable(GL_STENCIL_TEST);
  }

  //--------------------------------------------------------------------------
  // render bricks

  // set up transform
  const Transform &tform = tex_->transform();
  double mvmat[16];
  tform.get_trans(mvmat);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);

  if (use_cmap2)
  {
    const float cm2scale = 1.0 / Pow2(cmap2_.size());
    shader->setLocalParam(2, grange_, goffset_, cm2scale, cm2scale);
  }
  else
  {
    shader->setLocalParam(2, 1, 0, 0, 0);
  }

  if (levels == 1)
  {
#ifdef __APPLE__
    // Blend mode for no texture palette support.
    if (!ShaderProgramARB::shaders_supported() && mode_ == MODE_OVER)
    {
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      // Scale slice opacity (from build_colormap1)
      const double level_exponent = 0.0;  // used for multi-layer code
      const double bp = tan(1.570796327 * (0.5 - slice_alpha_ * 0.49999));
      double alpha = pow(0.5, bp); // 0.5 as default global cmap alpha
      alpha = 1.0 - pow((1.0 - alpha), imode_ ?
                        1.0/irate_/pow(2.0, level_exponent) :
                        1.0/sampling_rate_/pow(2.0, level_exponent) );
      glColor4f(1.0, 1.0, 1.0, alpha);
    }
#endif

    vector<TextureBrickHandle>& bs  = bricks[0];
    for (unsigned int i=0; i < bs.size(); i++) {
      TextureBrickHandle b = bs[i];
      vertex.clear();
      texcoord.clear();
      mask.clear();
      size.clear();
      b->compute_polygons(view_ray, dt, vertex, texcoord, size);
      b->mask_polygons(size, vertex, texcoord, mask, planes_);
      if (vertex.size() == 0) { continue; }
      load_brick(bs, i, use_cmap2);
      shader->setLocalParam(4, 1.0/b->nx(), 1.0/b->ny(), 1.0/b->nz(), 0.0);
      draw_polygons(vertex, texcoord, size, false, use_fog,
                    blend_num_bits_ > 8 ? blend_buffer_ : 0,
                    &mask, shader);
    }
  }
  else
  {
    Point corner[8];
    BBox bbox = tex_->bbox();
    corner[0] = bbox.min();
    corner[1] = Point(bbox.min().x(), bbox.min().y(), bbox.max().z());
    corner[2] = Point(bbox.min().x(), bbox.max().y(), bbox.min().z());
    corner[3] = Point(bbox.min().x(), bbox.max().y(), bbox.max().z());
    corner[4] = Point(bbox.max().x(), bbox.min().y(), bbox.min().z());
    corner[5] = Point(bbox.max().x(), bbox.min().y(), bbox.max().z());
    corner[6] = Point(bbox.max().x(), bbox.max().y(), bbox.min().z());
    corner[7] = bbox.max();
    double ts[8];
    for(unsigned int i = 0; i < 8; i++) {
      ts[i] = Dot(corner[i]-view_ray.origin(), view_ray.direction());
    }
    Sort(ts, 8);
    double tmin = (floor(ts[0]/dt) + 1)*dt;
    double tmax = floor(ts[7]/dt)*dt;
    int count = 1;
    int reset_val = (int)(pow(2.0, levels - 1));

    for (double t = tmax ; t >= tmin; t -= dt)
    {
      if (use_stencil_)
      {
        glStencilMask(~0);
        glClear(GL_STENCIL_BUFFER_BIT);
        glStencilMask(1);
      }
    
      for (int j = 0; j < levels; ++j )
      {
        if ( !draw_level_[j] ) continue;
        if ( j > 0 ){
          bool go_on = false;
          int k = j;
          while( k < levels ){
            int draw_level = int(pow(2.0, k));
            if ( count < draw_level )
            {
              break;
            }
            else if( count == draw_level )
            {
              go_on = true;
              break;
            }
            else
            {
              ++k;
            }
          }

          if ( !go_on )
          {
            break;
          }
        }

        if (levels > 2 && mode_ == MODE_OVER)
        {
          bind_colormap1( cmaps[j]->data_, cmaps[j]->tex_id_ );
        }

#ifdef __APPLE__
        // Blend mode for no texture palette support.
        if (!ShaderProgramARB::shaders_supported() && mode_ == MODE_OVER)
        {
          glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

          // Scale slice opacity (from build_colormap1)
          const double level_exponent = invert_opacity_  ?
            tan(1.570796327 * (0.5 - level_alpha_[levels - j - 1]) * 0.49999) :
            (double)j;
          const double bp = tan(1.570796327 * (0.5 - slice_alpha_ * 0.49999));
          double alpha = pow(0.5, bp); // 0.5 as default global cmap alpha
          alpha = 1.0 - pow((1.0 - alpha), imode_ ?
                            1.0/irate_/pow(2.0, level_exponent) :
                            1.0/sampling_rate_/pow(2.0, level_exponent) );
          glColor4f(1.0, 1.0, 1.0, alpha);
        }
#endif

        vector<TextureBrickHandle>& bs  = bricks[j];
        for(unsigned int i = 0; i < bs.size(); i++) {
          TextureBrickHandle b = bs[i];
          vertex.clear();
          texcoord.clear();
          mask.clear();
          size.clear();
          b->compute_polygon( view_ray, t, vertex, texcoord, size);
          b->mask_polygons(size, vertex, texcoord, mask, planes_);
          if (vertex.size() == 0) { continue; }
          load_brick(bs, i, use_cmap2);
          shader->setLocalParam(4, 1.0/b->nx(), 1.0/b->ny(), 1.0/b->nz(), 0.0);
          draw_polygons(vertex, texcoord, size, false, use_fog,
                        blend_num_bits_ > 8 ? blend_buffer_ : 0,
                        &mask, shader);
        }

        if (levels > 2 && mode_ == MODE_OVER)
        {
          release_colormap1();
        }
      }
      if( count == reset_val ) count = 0;
      ++count;
    }
  }

  // Undo transform.
  glPopMatrix();

  // Turn off stenciling.
  if (use_stencil_)
  {
    glDisable(GL_STENCIL_TEST);
  }

  glDepthMask(GL_TRUE);

  // Release shader.
  if (shader && shader->valid())
  {
    shader->release();
  }

  // Release textures.
  if (use_cmap2)
  {
    release_colormap2();
  }
  else
  {
    if (!(levels > 2 && mode_ == MODE_OVER))
    {
      release_colormap1();
    }
  }
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#ifdef _WIN32
  if (glActiveTexture)
#endif
  glActiveTexture(GL_TEXTURE0_ARB);
#endif
  glDisable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, 0);

  //--------------------------------------------------------------------------
  if (blend_num_bits_ == 8)
  {
    glDisable(GL_BLEND);
  }
  else
  {
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

#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#ifdef _WIN32
    if (glActiveTexture)
#endif
      glActiveTexture(GL_TEXTURE0);
#endif
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

    if (depth_test) glEnable(GL_DEPTH_TEST);
    if (lighting) glEnable(GL_LIGHTING);
    if (cull_face) glEnable(GL_CULL_FACE);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    blend_buffer_->set_use_default_shader(false);
  }

  for (unsigned int i = 0; i < cmaps.size(); i++)
  {
    delete cmaps[i];
  }

  CHECK_OPENGL_ERROR("VolumeRenderer::draw_volume end");

  tex_->unlock_bricks();
}


void
VolumeRenderer::draw_wireframe()
{
  tex_->lock_bricks();
  Ray view_ray = compute_view();
  const Transform &tform = tex_->transform();
  double mvmat[16];
  tform.get_trans(mvmat);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);
  glEnable(GL_DEPTH_TEST);
  GLboolean lighting = glIsEnabled(GL_LIGHTING);
  glDisable(GL_LIGHTING);
  vector<TextureBrickHandle> bricks;
  tex_->get_sorted_bricks(bricks, view_ray);

  const double rate = imode_ ? irate_ : sampling_rate_;
  const Vector diag = tex_->bbox().diagonal();
  const Vector cell_diag(diag.x()/tex_->nx(),
                         diag.y()/tex_->ny(),
                         diag.z()/tex_->nz());
  const double dt = cell_diag.length()/rate;
  const int num_slices = (int)(diag.length()/dt);

  vector<float> vertex;
  vector<float> texcoord;
  vector<int> size;
  vertex.reserve(num_slices*6);
  texcoord.reserve(num_slices*6);
  size.reserve(num_slices*6);

  for (unsigned int i=0; i<bricks.size(); i++)
  {
    glColor4f(0.8*(i+1.0)/bricks.size(), 0.8*(i+1.0)/bricks.size(), 0.8, 1.0);

    TextureBrickHandle b = bricks[i];
    const Point &pmin(b->bbox().min());
    const Point &pmax(b->bbox().max());
    Point corner[8];
    corner[0] = pmin;
    corner[1] = Point(pmin.x(), pmin.y(), pmax.z());
    corner[2] = Point(pmin.x(), pmax.y(), pmin.z());
    corner[3] = Point(pmin.x(), pmax.y(), pmax.z());
    corner[4] = Point(pmax.x(), pmin.y(), pmin.z());
    corner[5] = Point(pmax.x(), pmin.y(), pmax.z());
    corner[6] = Point(pmax.x(), pmax.y(), pmin.z());
    corner[7] = pmax;

    glBegin(GL_LINES);
    {
      for(int i=0; i<4; i++) {
        glVertex3d(corner[i].x(), corner[i].y(), corner[i].z());
        glVertex3d(corner[i+4].x(), corner[i+4].y(), corner[i+4].z());
      }
    }
    glEnd();
    glBegin(GL_LINE_LOOP);
    {
      glVertex3d(corner[0].x(), corner[0].y(), corner[0].z());
      glVertex3d(corner[1].x(), corner[1].y(), corner[1].z());
      glVertex3d(corner[3].x(), corner[3].y(), corner[3].z());
      glVertex3d(corner[2].x(), corner[2].y(), corner[2].z());
    }
    glEnd();
    glBegin(GL_LINE_LOOP);
    {
      glVertex3d(corner[4].x(), corner[4].y(), corner[4].z());
      glVertex3d(corner[5].x(), corner[5].y(), corner[5].z());
      glVertex3d(corner[7].x(), corner[7].y(), corner[7].z());
      glVertex3d(corner[6].x(), corner[6].y(), corner[6].z());
    }
    glEnd();

    glColor4f(0.4, 0.4, 0.4, 1.0);

    vertex.clear();
    texcoord.clear();
    size.clear();

    // Scale out dt such that the slices are artificially further apart.
    b->compute_polygons(view_ray, dt * 10, vertex, texcoord, size);
    vector<int> mask;
    b->mask_polygons(size, vertex, texcoord, mask, planes_);

    draw_polygons_wireframe(vertex, texcoord, size, false, false, 0, &mask);
  }
  if(lighting) glEnable(GL_LIGHTING);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  tex_->unlock_bricks();
}

#endif // SCI_OPENGL


double
VolumeRenderer::num_slices_to_rate(int num_slices)
{
  const Vector diag = tex_->bbox().diagonal();
  const Vector cell_diag(diag.x()/tex_->nx(),
                         diag.y()/tex_->ny(),
                         diag.z()/tex_->nz());
  const double dt = diag.length() / num_slices;
  const double rate = cell_diag.length() / dt;

  return rate;
}


void
VolumeRenderer::set_gradient_range(double min, double max)
{
  double range = max-min;
  if (fabs(range) < 0.001) {
    grange_ = 1.0;
    goffset_ = 1.0;
  } else {
    grange_ = 1/(max-min);
    goffset_ = -min/(max-min);
  }
}



} // namespace SCIRun
