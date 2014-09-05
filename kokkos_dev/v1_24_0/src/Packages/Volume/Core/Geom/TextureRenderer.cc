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
//    File   : TextureRenderer.cc
//    Author : Milan Ikits
//    Date   : Wed Jul  7 23:34:20 2004

#include <Core/Geom/GeomOpenGL.h>
#include <Packages/Volume/Core/Geom/TextureRenderer.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Color.h>

#include <Packages/Volume/Core/Util/Utils.h>
#include <Packages/Volume/Core/Util/Pbuffer.h>
#include <Packages/Volume/Core/Util/ShaderProgramARB.h>
#include <Packages/Volume/Core/Geom/VolShader.h>
#include <Packages/Volume/Core/Datatypes/CM2Shader.h>

#include <iostream>
using std::cerr;
using std::string;

using namespace SCIRun;

namespace Volume {

static const string Cmap2ShaderStringNV =
"!!ARBfp1.0 \n"
"TEMP c, z; \n"
"ATTRIB t = fragment.texcoord[0]; \n"
"PARAM s = program.local[0]; # {bp, sliceRatio, 0.0, 0.0} \n"
"TEX c, t, texture[0], RECT; \n"
"POW z.w, c.w, s.x; # alpha1 = pow(alpha, bp); \n"
"SUB z.w, 1.0, z.w; # alpha2 = 1.0-pow(1.0-alpha1, sliceRatio); \n"
"POW c.w, z.w, s.y; \n"
"SUB c.w, 1.0, c.w; \n"
"MUL c.xyz, c.xyzz, c.w; # c *= alpha2; \n"
"MOV result.color, c; \n"
"END";

static const string Cmap2ShaderStringATI =
"!!ARBfp1.0 \n"
"TEMP c, z; \n"
"ATTRIB t = fragment.texcoord[0]; \n"
"PARAM s = program.local[0]; # {bp, sliceRatio, 0.0, 0.0} \n"
"TEX c, t, texture[0], 2D; \n"
"POW z.w, c.w, s.x; # alpha1 = pow(alpha, bp); \n"
"SUB z.w, 1.0, z.w; # alpha2 = 1.0-pow(1.0-alpha1, sliceRatio); \n"
"POW c.w, z.w, s.y; \n"
"SUB c.w, 1.0, c.w; \n"
"MUL c.xyz, c.xyzz, c.w; # c *= alpha2; \n"
"MOV result.color, c; \n"
"END";

TextureRenderer::TextureRenderer(TextureHandle tex,
                                 ColorMapHandle cmap1, Colormap2Handle cmap2,
                                 int tex_mem) :
  GeomObj(),
  tex_(tex),
  mutex_("TextureRenderer Mutex"),
  cmap1_(cmap1),
  cmap2_(cmap2),
  cmap1_dirty_(true),
  cmap2_dirty_(true),
  mode_(MODE_NONE),
  interp_(true),
  lighting_(0),
  sampling_rate_(1.0),
  irate_(0.5),
  imode_(false),
  slice_alpha_(0.5),
  sw_raster_(false),
  cmap_size_(128),
  cmap1_tex_(0),
  cmap2_tex_(0),
  use_pbuffer_(true),
  raster_buffer_(0),
  shader_factory_(0),
  cmap2_buffer_(0),
  cmap2_shader_nv_(new FragmentProgramARB(Cmap2ShaderStringNV)),
  cmap2_shader_ati_(new FragmentProgramARB(Cmap2ShaderStringATI)),
  vol_shader_factory_(new VolShaderFactory()),
  blend_buffer_(0),
  blend_num_bits_(8),
  use_blend_buffer_(true),
  free_tex_mem_(tex_mem)
{}

TextureRenderer::TextureRenderer(const TextureRenderer& copy) :
  GeomObj(copy),
  tex_(copy.tex_),
  mutex_("TextureRenderer Mutex"),
  cmap1_(copy.cmap1_),
  cmap2_(copy.cmap2_),
  cmap1_dirty_(copy.cmap1_dirty_),
  cmap2_dirty_(copy.cmap2_dirty_),
  mode_(copy.mode_),
  interp_(copy.interp_),
  lighting_(copy.lighting_),
  sampling_rate_(copy.sampling_rate_),
  irate_(copy.irate_),
  imode_(copy.imode_),
  slice_alpha_(copy.slice_alpha_),
  sw_raster_(copy.sw_raster_),
  cmap_size_(copy.cmap_size_),
  cmap1_tex_(copy.cmap1_tex_),
  cmap2_tex_(copy.cmap2_tex_),
  use_pbuffer_(copy.use_pbuffer_),
  raster_buffer_(copy.raster_buffer_),
  shader_factory_(copy.shader_factory_),
  cmap2_buffer_(copy.cmap2_buffer_),
  cmap2_shader_nv_(copy.cmap2_shader_nv_),
  cmap2_shader_ati_(copy.cmap2_shader_ati_),
  vol_shader_factory_(copy.vol_shader_factory_),
  blend_buffer_(copy.blend_buffer_),
  blend_num_bits_(copy.blend_num_bits_),
  use_blend_buffer_(copy.use_blend_buffer_),
  free_tex_mem_(copy.free_tex_mem_)
{}

TextureRenderer::~TextureRenderer()
{
  delete cmap2_shader_nv_;
  delete cmap2_shader_ati_;
  delete vol_shader_factory_;
}

void
TextureRenderer::set_texture(TextureHandle tex)
{
  mutex_.lock();
  tex_ = tex;
  mutex_.unlock();
}

void
TextureRenderer::set_colormap1(ColorMapHandle cmap1)
{
  mutex_.lock();
  cmap1_ = cmap1;
  cmap1_dirty_ = true;
  mutex_.unlock();
}

void
TextureRenderer::set_colormap2(Colormap2Handle cmap2)
{
  mutex_.lock();
  cmap2_ = cmap2;
  cmap2_dirty_ = true;
  mutex_.unlock();
}

void
TextureRenderer::set_colormap_size(int size)
{
  if(cmap_size_ != size) {
    mutex_.lock();
    cmap_size_ = size;
    cmap1_dirty_ = true;
    cmap2_dirty_ = true;
    mutex_.unlock();
  }
}

void
TextureRenderer::set_slice_alpha(double alpha)
{
  if(slice_alpha_ != alpha) {
    mutex_.lock();
    slice_alpha_ = alpha;
    alpha_dirty_ = true;
    mutex_.unlock();
  }
}

void
TextureRenderer::set_sw_raster(bool b)
{
  if(sw_raster_ != b) {
    mutex_.lock();
    sw_raster_ = b;
    cmap2_dirty_ = true;
    mutex_.unlock();
  }
}


void
TextureRenderer::set_blend_num_bits(int b)
{
  mutex_.lock();
  blend_num_bits_ = b;
  mutex_.unlock();
}

bool
TextureRenderer::use_blend_buffer()
{
  return use_blend_buffer_;
}

#define TEXTURERENDERER_VERSION 1

void 
TextureRenderer::io(Piostream&)
{
  // nothing for now...
  NOT_FINISHED("TextureRenderer::io");
}

bool
TextureRenderer::saveobj(std::ostream&, const string&, GeomSave*)
{
  NOT_FINISHED("TextureRenderer::saveobj");
  return false;
}

Ray
TextureRenderer::compute_view()
{
  Transform field_trans = tex_->transform();
  double mvmat[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, mvmat);
  // index space view direction
  Vector v = field_trans.unproject(Vector(-mvmat[2], -mvmat[6], -mvmat[10]));
  Transform mv;
  mv.set_trans(mvmat);
  Point p = field_trans.unproject(mv.unproject(Point(0.0, 0.0, 0.0)));
  return Ray(p, v);
}

void
TextureRenderer::load_brick(Brick* brick)
{
  int nc = brick->nc();
  int idx[2];
  for(int c=0; c<nc; c++) {
    glActiveTexture(GL_TEXTURE0+c);
    int nb = brick->nb(c);
    int nx = brick->nx();
    int ny = brick->ny();
    int nz = brick->nz();
    idx[c] = -1;
    for(unsigned int i=0; i<tex_pool_.size() && idx[c]<0; i++) {
      if(tex_pool_[i].id != 0 && tex_pool_[i].brick == brick
         && !brick->dirty() && tex_pool_[i].comp == c
         && nx == tex_pool_[i].nx && ny == tex_pool_[i].ny
         && nz == tex_pool_[i].nz && nb == tex_pool_[i].nb
         && glIsTexture(tex_pool_[i].id)) {
        idx[c] = i;
      }
    }
    if(idx[c] != -1) {
      // bind texture object
      glBindTexture(GL_TEXTURE_3D, tex_pool_[idx[c]].id);
      // set interpolation method
      if(interp_) {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      } else {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      }
    } else {
      // find matching texture object
      for(unsigned int i=0; i<tex_pool_.size() && idx[c]<0; i++) {
        if(tex_pool_[i].id != 0
           && nx == tex_pool_[i].nx && ny == tex_pool_[i].ny
           && nz == tex_pool_[i].nz && nb == tex_pool_[i].nb) {
          idx[c] = i;
        }
      }
      bool reuse = (idx[c] >= 0);
      // 
      if(!reuse) {
        // if cannot reuse existing object allocate new object
        int new_size = nx*ny*nz*nb;
        if(new_size > free_tex_mem_) {
          // if there's no space, find object to replace
          int free_idx = -1;
          int size, size_max = -1;
          // find smallest available objects to delete
          // TODO: this is pretty dumb, optimize it later
          for(unsigned int i=0; i<tex_pool_.size(); i++) {
            for(int j=0; j<c; j++) {
              if(idx[j] == (int)i) continue;
            }
            size = tex_pool_[i].nx*tex_pool_[i].ny*tex_pool_[i].nz*tex_pool_[i].nb;
            if(new_size < free_tex_mem_+size && (size_max < 0 || size < size_max)) {
              free_idx = i;
              size_max = size;
            }
          }
          // delete found object
          if(glIsTexture(tex_pool_[free_idx].id))
            glDeleteTextures(1, &tex_pool_[free_idx].id);
          tex_pool_[free_idx].id = 0;
          free_tex_mem_ += size_max;
        }
        // find tex table entry to reuse
        for(unsigned int i=0; i<tex_pool_.size() && idx[c]<0; i++) {
          if(tex_pool_[i].id == 0)
            idx[c] = i;
        }
        // allocate new object
        unsigned int tex_id;
        glGenTextures(1, &tex_id);
        if(idx[c] < 0) {
          // create new entry
          tex_pool_.push_back(TexParam(nx, ny, nz, nb, tex_id));
          idx[c] = tex_pool_.size()-1;
        } else {
          // reuse existing entry
          tex_pool_[idx[c]].nx = nx; tex_pool_[idx[c]].ny = ny;
          tex_pool_[idx[c]].nz = nz; tex_pool_[idx[c]].nb = nb;
          tex_pool_[idx[c]].id = tex_id;
        }
        free_tex_mem_ -= new_size;
      }
      tex_pool_[idx[c]].brick = brick;
      tex_pool_[idx[c]].comp = c;
      // bind texture object
      glBindTexture(GL_TEXTURE_3D, tex_pool_[idx[c]].id);
      // set border behavior
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
      // set interpolation method
      if(interp_) {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      } else {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      }
      // download texture data
      unsigned int format = (nb == 1 ? GL_LUMINANCE : GL_RGBA);
      if(reuse) {
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, nx, ny, nz, format,
                        brick->tex_type(), brick->tex_data(c));
      } else {
        glTexImage3D(GL_TEXTURE_3D, 0, format, nx, ny, nz, 0, format,
                     brick->tex_type(), brick->tex_data(c));
      }
    }
  }
  brick->set_dirty(false);
  glActiveTexture(GL_TEXTURE0);
  int errcode = glGetError();
  if(errcode != GL_NO_ERROR) {
    cerr << "VolumeRenderer::load_texture | "
         << (char*)gluErrorString(errcode) << "\n";
  }
}

void
TextureRenderer::draw_polygons(Array1<float>& vertex, Array1<float>& texcoord,
                               Array1<int>& poly, bool normal, bool fog, Pbuffer* buffer)
{
  di_->polycount += poly.size();
  float mvmat[16];
  if(fog) {
    glGetFloatv(GL_MODELVIEW_MATRIX, mvmat);
  }
  if(buffer) {
    glActiveTexture(GL_TEXTURE3);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  }
  for(int i=0, k=0; i<poly.size(); i++) {
    if(buffer) {
      buffer->bind(GL_FRONT);
    }
    glBegin(GL_POLYGON);
    {
      if(normal) {
        float* v0 = &vertex[(k+0)*3];
        float* v1 = &vertex[(k+1)*3];
        float* v2 = &vertex[(k+2)*3];
        Vector dv1(v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]);
        Vector dv2(v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]);
        Vector n = Cross(dv1, dv2);
        n.normalize();
        glNormal3f(n.x(), n.y(), n.z());
      }
      for(int j=0; j<poly[i]; j++) {
        float* t = &texcoord[(k+j)*3];
        glMultiTexCoord3f(GL_TEXTURE0, t[0], t[1], t[2]);
        float* v = &vertex[(k+j)*3];
        if(fog) {
          float vz = mvmat[2]*v[0] + mvmat[6]*v[1] + mvmat[10]*v[2] + mvmat[14];
          glMultiTexCoord3f(GL_TEXTURE1, -vz, 0.0, 0.0);
        }
        glVertex3f(v[0], v[1], v[2]);
      }
    }
    glEnd();
    if(buffer) {
      buffer->release(GL_FRONT);
      buffer->swapBuffers();
    }
    k += poly[i];
  }
  if(buffer) {
    glActiveTexture(GL_TEXTURE0);
  }
}

void
TextureRenderer::build_colormap1()
{
  if(cmap1_dirty_ || alpha_dirty_) {
    bool size_dirty = false;
    if(cmap_size_ != cmap1_array_.dim1()) {
      cmap1_array_.resize(cmap_size_, 4);
      size_dirty = true;
    }
    // rebuild texture
    double dv = 1.0/(cmap1_array_.dim1() - 1);
    switch(mode_) {
    case MODE_SLICE: {
      for(int j=0; j<cmap1_array_.dim1(); j++) {
        // interpolate from colormap
        Color c = cmap1_->getColor(j*dv);
        double alpha = cmap1_->getAlpha(j*dv);
        // pre-multiply and quantize
        cmap1_array_(j,0) = (unsigned char)(c.r()*alpha*255);
        cmap1_array_(j,1) = (unsigned char)(c.g()*alpha*255);
        cmap1_array_(j,2) = (unsigned char)(c.b()*alpha*255);
        cmap1_array_(j,3) = (unsigned char)(alpha*255);
      }
    } break;
    case MODE_MIP: {
      for(int j=0; j<cmap1_array_.dim1(); j++) {
        // interpolate from colormap
        Color c = cmap1_->getColor(j*dv);
        double alpha = cmap1_->getAlpha(j*dv);
        // pre-multiply and quantize
        cmap1_array_(j,0) = (unsigned char)(c.r()*alpha*255);
        cmap1_array_(j,1) = (unsigned char)(c.g()*alpha*255);
        cmap1_array_(j,2) = (unsigned char)(c.b()*alpha*255);
        cmap1_array_(j,3) = (unsigned char)(alpha*255);
      }
    } break;
    case MODE_OVER: {
      double bp = tan(1.570796327 * (0.5 - slice_alpha_*0.49999));
      for(int j=0; j<cmap1_array_.dim1(); j++) {
        // interpolate from colormap
        Color c = cmap1_->getColor(j*dv);
        double alpha = cmap1_->getAlpha(j*dv);
        // scale slice opacity
        alpha = pow(alpha, bp);
        // opacity correction
        alpha = 1.0 - pow((1.0 - alpha), imode_ ? 1.0/irate_ : 1.0/sampling_rate_);
        // pre-multiply and quantize
        cmap1_array_(j,0) = (unsigned char)(c.r()*alpha*255);
        cmap1_array_(j,1) = (unsigned char)(c.g()*alpha*255);
        cmap1_array_(j,2) = (unsigned char)(c.b()*alpha*255);
        cmap1_array_(j,3) = (unsigned char)(alpha*255);
      }
    } break;
    default:
      break;
    }
    // update texture
    if(cmap1_tex_ == 0 || size_dirty) {
      glDeleteTextures(1, &cmap1_tex_);
      glGenTextures(1, &cmap1_tex_);
      glBindTexture(GL_TEXTURE_1D, cmap1_tex_);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, cmap1_array_.dim1(), 0,
                   GL_RGBA, GL_UNSIGNED_BYTE, &cmap1_array_(0,0));
    } else {
      glBindTexture(GL_TEXTURE_1D, cmap1_tex_);
      glTexSubImage1D(GL_TEXTURE_1D, 0, 0, cmap1_array_.dim1(),
                      GL_RGBA, GL_UNSIGNED_BYTE, &cmap1_array_(0,0));
    }
    cmap1_dirty_ = false;
    alpha_dirty_ = false;
  }
}

void
TextureRenderer::build_colormap2()
{
  if(cmap2_dirty_ || alpha_dirty_) {

    if(!sw_raster_ && use_pbuffer_ && !raster_buffer_) {
      raster_buffer_ = new Pbuffer(256, 64, GL_FLOAT, 32, true, GL_FALSE);
      cmap2_buffer_ = new Pbuffer(256, 64, GL_INT, 8, true, GL_FALSE);
      shader_factory_ = new CM2ShaderFactory();
      if(raster_buffer_->create() || cmap2_buffer_->create()
         || cmap2_shader_nv_->create() || cmap2_shader_ati_->create()) {
        raster_buffer_->destroy();
        cmap2_buffer_->destroy();
        cmap2_shader_nv_->destroy();
        cmap2_shader_ati_->destroy();
        delete raster_buffer_;
        delete cmap2_buffer_;
        delete shader_factory_;
        raster_buffer_ = 0;
        cmap2_buffer_ = 0;
        shader_factory_ = 0;
        use_pbuffer_ = false;
      } else {
        raster_buffer_->set_use_default_shader(false);
        cmap2_buffer_->set_use_default_shader(false);
      }
    }

    if(!sw_raster_ && use_pbuffer_) {
      //--------------------------------------------------------------
      // hardware rasterization
      if(cmap2_dirty_) {
        raster_buffer_->activate();
        raster_buffer_->set_use_texture_matrix(false);
        glDrawBuffer(GL_FRONT);
        glViewport(0, 0, raster_buffer_->width(), raster_buffer_->height());
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);
        raster_buffer_->swapBuffers();
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(-1.0, -1.0, 0.0);
        glScalef(2.0, 2.0, 2.0);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glDisable(GL_CULL_FACE);
        glDisable(GL_BLEND);
        //glEnable(GL_BLEND);
        //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glActiveTexture(GL_TEXTURE0);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        // rasterize widgets
        vector<CM2Widget*> widgets = cmap2_->widgets();
        for (unsigned int i=0; i<widgets.size(); i++) {
          raster_buffer_->bind(GL_FRONT);
          widgets[i]->rasterize(*shader_factory_, cmap2_->faux(), raster_buffer_);
          raster_buffer_->release(GL_FRONT);
          raster_buffer_->swapBuffers();
        }
        //glDisable(GL_BLEND);
        raster_buffer_->deactivate();
        raster_buffer_->set_use_texture_matrix(true);
      }
      //--------------------------------------------------------------
      // opacity correction and quantization
      cmap2_buffer_->activate();
      glDrawBuffer(GL_FRONT);
      glViewport(0, 0, cmap2_buffer_->width(), cmap2_buffer_->height());
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glTranslatef(-1.0, -1.0, 0.0);
      glScalef(2.0, 2.0, 2.0);
      glDisable(GL_DEPTH_TEST);
      glDisable(GL_LIGHTING);
      glDisable(GL_CULL_FACE);
      FragmentProgramARB* shader =
        raster_buffer_->need_shader() ? cmap2_shader_nv_ : cmap2_shader_ati_;
      shader->bind();
      double bp = mode_ == MODE_MIP ? 1.0 : 
        tan(1.570796327 * (0.5 - slice_alpha_*0.49999));
      shader->setLocalParam(0, bp, imode_ ? 1.0/irate_ : 1.0/sampling_rate_,
                            0.0, 0.0);
      glActiveTexture(GL_TEXTURE0);
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
      raster_buffer_->bind(GL_FRONT);
      glBegin(GL_QUADS);
      {
        glTexCoord2f( 0.0,  0.0);
        glVertex2f( 0.0,  0.0);
        glTexCoord2f(1.0,  0.0);
        glVertex2f( 1.0,  0.0);
        glTexCoord2f(1.0,  1.0);
        glVertex2f( 1.0,  1.0);
        glTexCoord2f( 0.0,  1.0);
        glVertex2f( 0.0,  1.0);
      }
      glEnd();
      raster_buffer_->release(GL_FRONT);
      shader->release();
      cmap2_buffer_->swapBuffers();
      cmap2_buffer_->deactivate();
    } else {
      //--------------------------------------------------------------
      // software rasterization
      bool size_dirty =
        cmap_size_ != raster_array_.dim2()
        || cmap_size_/4 != raster_array_.dim1();
      if(cmap2_dirty_ || size_dirty) {
        if(size_dirty) {
          raster_array_.resize(cmap_size_/4, cmap_size_, 4);
          cmap2_array_.resize(cmap_size_/4, cmap_size_, 4);
        }
        // clear cmap
        for(int i=0; i<raster_array_.dim1(); i++) {
          for(int j=0; j<raster_array_.dim2(); j++) {
            raster_array_(i,j,0) = 0.0;
            raster_array_(i,j,1) = 0.0;
            raster_array_(i,j,2) = 0.0;
            raster_array_(i,j,3) = 0.0;
          }
        }
        vector<CM2Widget*>& widget = cmap2_->widgets();
        // rasterize widgets
        for(unsigned int i=0; i<widget.size(); i++) {
          widget[i]->rasterize(raster_array_, cmap2_->faux());
        }
        for(int i=0; i<raster_array_.dim1(); i++) {
          for(int j=0; j<raster_array_.dim2(); j++) {
            raster_array_(i,j,0) = Clamp(raster_array_(i,j,0), 0.0f, 1.0f);
            raster_array_(i,j,1) = Clamp(raster_array_(i,j,1), 0.0f, 1.0f);
            raster_array_(i,j,2) = Clamp(raster_array_(i,j,2), 0.0f, 1.0f);
            raster_array_(i,j,3) = Clamp(raster_array_(i,j,3), 0.0f, 1.0f);
          }
        }
      }
      //--------------------------------------------------------------
      // opacity correction
      switch(mode_) {
      case MODE_MIP:
      case MODE_SLICE: {
        for(int i=0; i<raster_array_.dim1(); i++) {
          for(int j=0; j<raster_array_.dim2(); j++) {
            double alpha = raster_array_(i,j,3);
            cmap2_array_(i,j,0) = (unsigned char)(raster_array_(i,j,0)*alpha*255);
            cmap2_array_(i,j,1) = (unsigned char)(raster_array_(i,j,1)*alpha*255);
            cmap2_array_(i,j,2) = (unsigned char)(raster_array_(i,j,2)*alpha*255);
            cmap2_array_(i,j,3) = (unsigned char)(alpha*255);
          }
        }
      } break;
      case MODE_OVER: {
        double bp = tan(1.570796327 * (0.5 - slice_alpha_*0.49999));
        for(int i=0; i<raster_array_.dim1(); i++) {
          for(int j=0; j<raster_array_.dim2(); j++) {
            double alpha = raster_array_(i,j,3);
            alpha = pow(alpha, bp);
            alpha = 1.0-pow(1.0-alpha, imode_ ? 1.0/irate_ : 1.0/sampling_rate_);
            cmap2_array_(i,j,0) = (unsigned char)(raster_array_(i,j,0)*alpha*255);
            cmap2_array_(i,j,1) = (unsigned char)(raster_array_(i,j,1)*alpha*255);
            cmap2_array_(i,j,2) = (unsigned char)(raster_array_(i,j,2)*alpha*255);
            cmap2_array_(i,j,3) = (unsigned char)(alpha*255);
          }
        }
      } break;
      default:
        break;
      }
      //--------------------------------------------------------------
      // update texture
      if(!cmap2_tex_ || size_dirty) {
        if(glIsTexture(cmap2_tex_)) {
          glDeleteTextures(1, &cmap2_tex_);
        }
        glGenTextures(1, &cmap2_tex_);
        glBindTexture(GL_TEXTURE_2D, cmap2_tex_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cmap2_array_.dim2(), cmap2_array_.dim1(),
                     0, GL_RGBA, GL_UNSIGNED_BYTE, &cmap2_array_(0,0,0));
        glBindTexture(GL_TEXTURE_2D, 0);
      } else {
        glBindTexture(GL_TEXTURE_2D, cmap2_tex_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cmap2_array_.dim2(), cmap2_array_.dim1(),
                        GL_RGBA, GL_UNSIGNED_BYTE, &cmap2_array_(0,0,0));
        glBindTexture(GL_TEXTURE_2D, 0);
      }
    }
  }
  cmap2_dirty_ = false;
  alpha_dirty_ = false;
}

void
TextureRenderer::bind_colormap1()
{
  // bind texture to unit 2
  glActiveTexture(GL_TEXTURE2_ARB);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glEnable(GL_TEXTURE_1D);
  glBindTexture(GL_TEXTURE_1D, cmap1_tex_);
  // enable data texture unit 1
  glActiveTexture(GL_TEXTURE1_ARB);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glEnable(GL_TEXTURE_3D);
  glActiveTexture(GL_TEXTURE0_ARB);
}

void
TextureRenderer::bind_colormap2()
{
  // bind texture to unit 2
  glActiveTexture(GL_TEXTURE2_ARB);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  if(!sw_raster_ && use_pbuffer_) {
    cmap2_buffer_->bind(GL_FRONT);
  } else {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, cmap2_tex_);
  }
  // enable data texture unit 1
  glActiveTexture(GL_TEXTURE1_ARB);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glEnable(GL_TEXTURE_3D);
  glActiveTexture(GL_TEXTURE0_ARB);
}

void
TextureRenderer::release_colormap1()
{
  glActiveTexture(GL_TEXTURE2_ARB);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glDisable(GL_TEXTURE_1D);
  glBindTexture(GL_TEXTURE_1D, 0);
  // enable data texture unit 1
  glActiveTexture(GL_TEXTURE1_ARB);
  glDisable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, 0);
  glActiveTexture(GL_TEXTURE0_ARB);
}

void
TextureRenderer::release_colormap2()
{
  glActiveTexture(GL_TEXTURE2_ARB);
  if(!sw_raster_ && use_pbuffer_) {
    cmap2_buffer_->release(GL_FRONT);
  } else {
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  glActiveTexture(GL_TEXTURE1_ARB);
  glDisable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, 0);
  glActiveTexture(GL_TEXTURE0_ARB);
}

} // namespace Volume
