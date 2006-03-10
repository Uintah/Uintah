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


#include <sci_defs/ogl_defs.h>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <Core/Geom/DrawInfoOpenGL.h>
#include <Core/Volume/TextureRenderer.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/Environment.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Color.h>

#include <Core/Math/MiscMath.h>
#include <Core/Volume/Utils.h>
#include <Core/Geom/Pbuffer.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Volume/VolShader.h>
#include <Core/Volume/CM2Shader.h>

#include <iostream>
using std::string;

namespace SCIRun {

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
                                   ColorMapHandle cmap1, 
				   vector<ColorMap2Handle> &cmap2,
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
    hardware_raster_(false),
    cmap2_size_(256),
    cmap1_tex_(0),
    cmap2_tex_(0),
    raster_pbuffer_(0),
    shader_factory_(0),
    cmap2_pbuffer_(0),
    cmap2_shader_nv_(new FragmentProgramARB(Cmap2ShaderStringNV)),
    cmap2_shader_ati_(new FragmentProgramARB(Cmap2ShaderStringATI)),
    vol_shader_factory_(new VolShaderFactory()),
    blend_buffer_(0),
    blend_num_bits_(8),
    use_blend_buffer_(true),
    free_tex_mem_(tex_mem),
    use_stencil_(false)
  {
  }

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
    hardware_raster_(copy.hardware_raster_),
    cmap2_size_(copy.cmap2_size_),
    cmap1_tex_(copy.cmap1_tex_),
    cmap2_tex_(copy.cmap2_tex_),
    raster_pbuffer_(copy.raster_pbuffer_),
    shader_factory_(copy.shader_factory_),
    cmap2_pbuffer_(copy.cmap2_pbuffer_),
    cmap2_shader_nv_(copy.cmap2_shader_nv_),
    cmap2_shader_ati_(copy.cmap2_shader_ati_),
    vol_shader_factory_(copy.vol_shader_factory_),
    blend_buffer_(copy.blend_buffer_),
    blend_num_bits_(copy.blend_num_bits_),
    use_blend_buffer_(copy.use_blend_buffer_),
    free_tex_mem_(copy.free_tex_mem_),
    use_stencil_(copy.use_stencil_)
  {
  }

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
  TextureRenderer::set_colormap2(vector<ColorMap2Handle> &cmap2)
  {
    mutex_.lock();
    cmap2_ = cmap2;
    cmap2_dirty_ = true;
    mutex_.unlock();
  }

  void
  TextureRenderer::set_colormap_size(int size)
  {
    if(cmap2_size_ != size) {
      mutex_.lock();
      cmap2_size_ = size;
      cmap2_dirty_ = true;
      mutex_.unlock();
    }
  }

  void
  TextureRenderer::set_slice_alpha(double alpha)
  {
    if(fabs(slice_alpha_ - alpha) > 0.0001) {
      mutex_.lock();
      slice_alpha_ = alpha;
      alpha_dirty_ = true;
      mutex_.unlock();
    }
  }

  void
  TextureRenderer::set_sw_raster(bool b)
  {
    if(hardware_raster_ == b) {
      mutex_.lock();
      hardware_raster_ = !b;
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
    const Transform &field_trans = tex_->transform();
    double mvmat[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, mvmat);
    // index space view direction
    Vector v = field_trans.unproject(Vector(-mvmat[2], -mvmat[6], -mvmat[10]));
    v.safe_normalize();
    Transform mv;
    mv.set_trans(mvmat);
    Point p = field_trans.unproject(mv.unproject(Point(0,0,0)));
    return Ray(p, v);
  }

  void
  TextureRenderer::load_brick(vector<TextureBrickHandle> &bricks, int bindex,
                              bool use_cmap2)
  {
    TextureBrickHandle brick = bricks[bindex];
    int nc = use_cmap2?brick->nc():1;
#if !defined(GL_ARB_fragment_program) && !defined(GL_ATI_fragment_shader)
    nc = 1;
#endif
    int idx[2];
    for(int c=0; c<nc; c++) {
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#  ifdef _WIN32
      if (glActiveTexture)
#  endif
        glActiveTexture(GL_TEXTURE0+c);
#endif
      int nb = brick->nb(c);
      int nx = brick->nx();
      int ny = brick->ny();
      int nz = brick->nz();
      idx[c] = -1;
      for(unsigned int i=0; i<tex_pool_.size() && idx[c]<0; i++)
      {
        if(tex_pool_[i].id != 0 && tex_pool_[i].brick == brick
           && !brick->dirty() && tex_pool_[i].comp == c
           && nx == tex_pool_[i].nx && ny == tex_pool_[i].ny
           && nz == tex_pool_[i].nz && nb == tex_pool_[i].nb
           && glIsTexture(tex_pool_[i].id))
        {
          if (tex_pool_[i].brick == brick)
          {
            idx[c] = i;
          }
          else
          {
            bool found = false;
            for (unsigned int j = 0; j < bricks.size(); j++)
            {
              if (bricks[j] == brick)
              {
                found = true;
              }
            }
            if (!found)
            {
              idx[c] = i;
            }
          }
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
        for(unsigned int i=0; i<tex_pool_.size() && idx[c]<0; i++)
          {
            if(tex_pool_[i].id != 0 && c == tex_pool_[i].comp
               && nx == tex_pool_[i].nx && ny == tex_pool_[i].ny
               && nz == tex_pool_[i].nz && nb == tex_pool_[i].nb)
              {
                if (tex_pool_[i].brick == brick)
                  {
                    idx[c] = i;
                  }
                else
                  {
                    bool found = false;
                    for (unsigned int j = 0; j < bricks.size(); j++)
                      {
                        if (bricks[j] == brick)
                          {
                            found = true;
                          }
                      }
                    if (!found)
                      {
                        idx[c] = i;
                      }
                  }
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
            if (free_idx != -1)
              {
                // delete found object
                if(glIsTexture(tex_pool_[free_idx].id))
                  glDeleteTextures(1, &tex_pool_[free_idx].id);
                tex_pool_[free_idx].id = 0;
              }
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
        glPixelStorei(GL_UNPACK_ALIGNMENT,nb);
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
#if defined( GL_TEXTURE_COLOR_TABLE_SGI ) && defined(__sgi)
        if (reuse)
          {
            glTexSubImage3DEXT(GL_TEXTURE_3D, 0, 0, 0, 0, nx, ny, nz, GL_RED,
                               brick->tex_type(), brick->tex_data(c));
          }
        else
          {
            glTexImage3DEXT(GL_TEXTURE_3D, 0, GL_INTENSITY8,
                            nx, ny, nz, 0, GL_RED,
                            brick->tex_type(), brick->tex_data(c));
          }
#elif defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
        if (ShaderProgramARB::shaders_supported())
          {
            unsigned int format = (nb == 1 ? GL_LUMINANCE : GL_RGBA);
            if (ShaderProgramARB::texture_non_power_of_two())
              {
                glPixelStorei(GL_UNPACK_ROW_LENGTH, brick->sx());
                glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, brick->sy());
                glPixelStorei(GL_UNPACK_ALIGNMENT, (nb == 1)?1:4);
              }
            if (reuse && glTexSubImage3D)
              {
                glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, nx, ny, nz, format,
                                brick->tex_type(), brick->tex_data(c));
              }
            else
              {
#  ifdef _WIN32
                if (glTexImage3D)
#  endif
                  glTexImage3D(GL_TEXTURE_3D, 0, format, nx, ny, nz, 0, format,
                               brick->tex_type(), brick->tex_data(c));
              }
            if (ShaderProgramARB::texture_non_power_of_two())
              {
                glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
                glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
              }
          }
        else
#endif
#if !defined(__sgi)
#  if defined(GL_EXT_shared_texture_palette) && !defined(__APPLE__)
          {
            if (reuse && glTexSubImage3D)
              {
                glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, nx, ny, nz,
                                GL_COLOR_INDEX,
                                brick->tex_type(), brick->tex_data(c));
              }
            else
              {
#    ifdef _WIN32
                if (glTexImage3D)
#    endif
                  glTexImage3D(GL_TEXTURE_3D, 0, GL_COLOR_INDEX8_EXT,
                               nx, ny, nz, 0, GL_COLOR_INDEX,
                               brick->tex_type(), brick->tex_data(c));
              }
          }
#  elif defined(_WIN32) || defined(GL_VERSION_1_2) // Workaround for old bad nvidia headers.
        {
          if (reuse && glTexSubImage3D)
            {
              glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, nx, ny, nz,
                              GL_LUMINANCE,
                              brick->tex_type(), brick->tex_data(c));
            }
          else
            {
#    ifdef _WIN32
              if (glTexImage3D)
#    endif
                glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE,
                             nx, ny, nz, 0, GL_LUMINANCE,
                             brick->tex_type(), brick->tex_data(c));
            }
        }
#  endif
#endif // !__sgi
      }
    }
    brick->set_dirty(false);
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#  ifdef _WIN32
    if (glActiveTexture)
#  endif
      glActiveTexture(GL_TEXTURE0);
#endif
    CHECK_OPENGL_ERROR("VolumeRenderer::load_texture end");
  }

 void
  TextureRenderer::draw_polygons(vector<float>& vertex, 
				 vector<float>& texcoord,
                                 vector<int>& poly, 
				 bool normal, bool fog,
                                 Pbuffer* buffer,
				 vector<int> *mask, 
				 FragmentProgramARB* shader)

  {
    di_->polycount_ += poly.size();
    float mvmat[16];
    if(fog) {
      glGetFloatv(GL_MODELVIEW_MATRIX, mvmat);
    }
    if(buffer) {
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#  ifdef _WIN32
      if (glActiveTexture)
#  endif
        glActiveTexture(GL_TEXTURE3);
#endif
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    }
    for(unsigned int i=0, k=0; i<poly.size(); i++) {
      if (mask && shader) {
        if (!(*mask)[i]) {
          k += poly[i];
          continue;
        }
        //	float v = float((*mask)[i]);
        float v = float(((*mask)[i] << 1) + 1);
	shader->setLocalParam(3, v,v,v,v);
      }
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
          float* v = &vertex[(k+j)*3];
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#  ifdef _WIN32
          if (glMultiTexCoord3f) {
#  endif // _WIN32
            glMultiTexCoord3f(GL_TEXTURE0, t[0], t[1], t[2]);
            if(fog) {
              float vz = mvmat[2]*v[0] + mvmat[6]*v[1] + mvmat[10]*v[2] + mvmat[14];
              glMultiTexCoord3f(GL_TEXTURE1, -vz, 0.0, 0.0);
            }
#  ifdef _WIN32
          } else {
            glTexCoord3f(t[0], t[1], t[2]);
          }
#  endif // _WIN32
#else
          glTexCoord3f(t[0], t[1], t[2]);
#endif
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
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#  ifdef _WIN32
      if (glActiveTexture)
#  endif // _WIN32
        glActiveTexture(GL_TEXTURE0);
#endif
    }
  }


  void
  TextureRenderer::draw_polygons_wireframe(vector<float>& vertex,
                                           vector<float>& texcoord,
                                           vector<int>& poly,
                                           bool normal, bool fog,
                                           Pbuffer* buffer,
					   vector<int> *mask)
  {
    di_->polycount_ += poly.size();
    float mvmat[16];
    if(fog) {
      glGetFloatv(GL_MODELVIEW_MATRIX, mvmat);
    }
    for(unsigned int i=0, k=0; i<poly.size(); i++)
      {
	if (mask) {
	  int v = (*mask)[i] ? 2:1;
	  glColor4d(v & 1 ? 1.0 : 0.0, 
		    v & 2 ? 1.0 : 0.0, 
		    v & 4 ? 1.0 : 0.0, 1.0);
	}
		  
        glBegin(GL_LINE_LOOP);
        {
          for(int j=0; j<poly[i]; j++)
            {
              float* v = &vertex[(k+j)*3];
              glVertex3f(v[0], v[1], v[2]);
            }
        }
        glEnd();
        k += poly[i];
      }
  }


  void
  TextureRenderer::build_colormap1(Array2<float>& cmap_array,
                                   unsigned int& cmap_tex, bool& cmap_dirty,
                                   bool& alpha_dirty,  double level_exponent)
  {
    if(cmap_dirty || alpha_dirty) {
      bool size_dirty = false;
      if(256 != cmap_array.dim1()) {
        cmap_array.resize(256, 4);
        size_dirty = true;
      }
      // rebuild texture
      double dv = 1.0/(cmap_array.dim1() - 1);
      switch(mode_) {
      case MODE_SLICE: {
        for(int j=0; j<cmap_array.dim1(); j++) {
          // interpolate from colormap
          const Color &c = cmap1_->getColor(j*dv);
          double alpha = cmap1_->getAlpha(j*dv);
          // pre-multiply and quantize
          cmap_array(j,0) = c.r()*alpha;
          cmap_array(j,1) = c.g()*alpha;
          cmap_array(j,2) = c.b()*alpha;
          cmap_array(j,3) = alpha;
        }
      } break;
      case MODE_MIP: {
        for(int j=0; j<cmap_array.dim1(); j++) {
          // interpolate from colormap
          const Color &c = cmap1_->getColor(j*dv);
          double alpha = cmap1_->getAlpha(j*dv);
          // pre-multiply and quantize
          cmap_array(j,0) = c.r()*alpha;
          cmap_array(j,1) = c.g()*alpha;
          cmap_array(j,2) = c.b()*alpha;
          cmap_array(j,3) = alpha;
        }
      } break;
      case MODE_OVER: {
        double bp = tan(1.570796327 * (0.5 - slice_alpha_*0.49999));
        for(int j=0; j<cmap_array.dim1(); j++) {
          // interpolate from colormap
          const Color &c = cmap1_->getColor(j*dv);
          double alpha = cmap1_->getAlpha(j*dv);
          // scale slice opacity
          alpha = pow(alpha, bp);
          // opacity correction
          alpha = 1.0 - pow((1.0 - alpha), imode_ ?
                            1.0/irate_/pow(2.0, level_exponent) :
                            1.0/sampling_rate_/pow(2.0, level_exponent) );

          // pre-multiply and quantize
          cmap_array(j,0) = c.r()*alpha;
          cmap_array(j,1) = c.g()*alpha;
          cmap_array(j,2) = c.b()*alpha;
          cmap_array(j,3) = alpha;
        }
      } break;
      default:
        break;
      }
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
      // This texture is not used if there is no shaders.
      // glColorTable is used instead.
      if (ShaderProgramARB::shaders_supported())
        {
          // Update 1D texture.
          if (cmap_tex == 0 || size_dirty)
            {
              glDeleteTextures(1, &cmap_tex);
              glGenTextures(1, &cmap_tex);
              glBindTexture(GL_TEXTURE_1D, cmap_tex);
              glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
              glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
              glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
              glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA16, cmap_array.dim1(), 0,
                           GL_RGBA, GL_FLOAT, &cmap_array(0,0));
            }
          else
            {
              glBindTexture(GL_TEXTURE_1D, cmap_tex);
              glTexSubImage1D(GL_TEXTURE_1D, 0, 0, cmap_array.dim1(),
                              GL_RGBA, GL_FLOAT, &cmap_array(0,0));
            }
        }
#endif
      cmap_dirty = false;
      alpha_dirty = false;
    }
  }


  void
  TextureRenderer::colormap2_hardware_rasterize() 
  {
    if(cmap2_dirty_) {
      raster_pbuffer_->activate();
      raster_pbuffer_->set_use_texture_matrix(false);
      glDrawBuffer(GL_FRONT);

      glClearColor(0.0, 0.0, 0.0, 0.0);
      glClear(GL_COLOR_BUFFER_BIT);
      raster_pbuffer_->swapBuffers();
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glTranslatef(-1.0, -1.0, 0.0);
      glScalef(2.0/Pow2(cmap2_.size()), 2.0, 2.0);
      glDisable(GL_DEPTH_TEST);
      glDisable(GL_LIGHTING);
      glDisable(GL_CULL_FACE);
      glDisable(GL_BLEND);
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#  ifdef _WIN32
      if (glActiveTexture)
#  endif
	glActiveTexture(GL_TEXTURE0);
#endif
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
      // rasterize widgets
      glViewport(0,0,raster_pbuffer_->width(), raster_pbuffer_->height());
      //cmap2_size_*c, 0,
      //	 cmap2_size_*(c+1), raster_pbuffer_->height());

      for (unsigned int c = 0; c < cmap2_.size(); ++c) {
	vector<CM2WidgetHandle> widgets = cmap2_[c]->widgets();
	for (unsigned int i=0; i<widgets.size(); i++) {
	  raster_pbuffer_->bind(GL_FRONT);
	  widgets[i]->rasterize(*shader_factory_, raster_pbuffer_);
	  raster_pbuffer_->release(GL_FRONT);
	  raster_pbuffer_->swapBuffers();
	}
	glTranslatef(1.0, 0.0, 0.0);
      }
      raster_pbuffer_->deactivate();
      raster_pbuffer_->set_use_texture_matrix(true);
      alpha_dirty_ = 1;
    }

    if (alpha_dirty_) {
      //--------------------------------------------------------------
      // opacity correction and quantization
      cmap2_pbuffer_->activate();
      glDrawBuffer(GL_FRONT);
      glViewport(0, 0, cmap2_pbuffer_->width(), cmap2_pbuffer_->height());
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
	raster_pbuffer_->need_shader() ? cmap2_shader_nv_ : cmap2_shader_ati_;
      shader->bind();
      double bp = mode_ == MODE_MIP ? 1.0 : 
	tan(1.570796327 * (0.5 - slice_alpha_*0.49999));
      shader->setLocalParam(0, bp, imode_ ? 1.0/irate_ : 1.0/sampling_rate_, 
			    0.0, 0.0);
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#  ifdef _WIN32
      if (glActiveTexture)
#  endif // _WIN32
	glActiveTexture(GL_TEXTURE0);
#endif
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
      raster_pbuffer_->bind(GL_FRONT);
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
      raster_pbuffer_->release(GL_FRONT);
      shader->release();
      cmap2_pbuffer_->swapBuffers();
      cmap2_pbuffer_->deactivate();
    }
  }
    


  void
  TextureRenderer::colormap2_software_rasterize() 
  {
    //--------------------------------------------------------------
    // software rasterization
    const int width = cmap2_size_ * Pow2(cmap2_.size());
    const bool size_dirty = (width != raster_array_.dim2());

    if (cmap2_dirty_ || size_dirty) {
      if(size_dirty) {
	raster_array_.resize(64, cmap2_size_, 4);
	cmap2_array_.resize(64, width, 4);
      }
      for (unsigned int c = 0; c < cmap2_.size(); ++c) 
      {
	raster_array_.initialize(0.0);
	vector<CM2WidgetHandle>& widget = cmap2_[c]->widgets();
	// rasterize widgets
	for(unsigned int i=0; i<widget.size(); i++) {
	  widget[i]->rasterize(raster_array_);
	}

	switch(mode_) {
	case MODE_MIP:
	case MODE_SLICE: {
	  for(int i=0; i<raster_array_.dim1(); i++) {
	    for(int j=0; j<raster_array_.dim2(); j++) {
	      const int k = c*cmap2_size_+j;
	      double alpha = Clamp(raster_array_(i,j,3), 0.0f, 1.0f)*255.0;
	      cmap2_array_(i,k,0) = 
		(unsigned char)(Clamp(raster_array_(i,j,0),0.0f,1.0f)*alpha);
	      cmap2_array_(i,k,1) = 
		(unsigned char)(Clamp(raster_array_(i,j,1),0.0f,1.0f)*alpha);
	      cmap2_array_(i,k,2) = 
		(unsigned char)(Clamp(raster_array_(i,j,2),0.0f,1.0f)*alpha);
	      cmap2_array_(i,k,3) = (unsigned char)(alpha);
	    }
	  }
	} break;
	case MODE_OVER: {
	  double bp = tan(1.570796327 * (0.5 - slice_alpha_*0.49999));
	  for(int i=0; i<raster_array_.dim1(); i++) {
	    for(int j=0; j<raster_array_.dim2(); j++) {
	      const int k = c*cmap2_size_+j;
	      double alpha = Clamp(raster_array_(i,j,3), 0.0f, 1.0f);
	      alpha = pow(alpha, bp);
	      alpha = 1.-pow(1.0-alpha,imode_ ? 1./irate_ : 1./sampling_rate_);
	      alpha *= 255.0;

	      cmap2_array_(i,k,0) =
		(unsigned char)(Clamp(raster_array_(i,j,0),0.0f,1.0f)*alpha);
	      cmap2_array_(i,k,1) =
		(unsigned char)(Clamp(raster_array_(i,j,1),0.0f,1.0f)*alpha);
	      cmap2_array_(i,k,2) =
		(unsigned char)(Clamp(raster_array_(i,j,2),0.0f,1.0f)*alpha);
	      cmap2_array_(i,k,3) = (unsigned char)(alpha);
	    }
	  }
	} break;
	default:
	  break;
	}
      }
    }
    //--------------------------------------------------------------
    // opacity correction
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
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
		   cmap2_array_.dim2(), cmap2_array_.dim1(),
		   0, GL_RGBA, GL_UNSIGNED_BYTE, &cmap2_array_(0,0,0));
      glBindTexture(GL_TEXTURE_2D, 0);
    } else {
      glBindTexture(GL_TEXTURE_2D, cmap2_tex_);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
		      cmap2_array_.dim2(), cmap2_array_.dim1(),
		      GL_RGBA, GL_UNSIGNED_BYTE, &cmap2_array_(0,0,0));
      glBindTexture(GL_TEXTURE_2D, 0);
    }      
  }
    

  void
  TextureRenderer::colormap2_hardware_destroy_buffers()
  {
    if (raster_pbuffer_) {
      raster_pbuffer_->destroy();
      delete raster_pbuffer_;
      raster_pbuffer_ = 0;
    }
    if (cmap2_pbuffer_) {
      cmap2_pbuffer_->destroy();
      delete cmap2_pbuffer_;
      cmap2_pbuffer_ = 0;
    }
  }

  

  void
  TextureRenderer::colormap2_hardware_rasterize_setup()
  {
    int width = cmap2_size_ * Pow2(cmap2_.size());
    if (raster_pbuffer_ && raster_pbuffer_->width() == width) 
      return;
    else
      colormap2_hardware_destroy_buffers();

    if (!raster_pbuffer_) {
      raster_pbuffer_ = new Pbuffer(width, 64, GL_FLOAT, 32, true, GL_FALSE);
    }

    if (!cmap2_pbuffer_)
      cmap2_pbuffer_ = new Pbuffer(width, 64, GL_INT, 8, true, GL_FALSE);

    if (!raster_pbuffer_->create() || !cmap2_pbuffer_->create()) {
      colormap2_hardware_destroy_buffers();
      return;
    }

    if (!shader_factory_) {
      shader_factory_ = new CM2ShaderFactory();
      if (cmap2_shader_nv_->create() || // True shader::create, means it failed
	  cmap2_shader_ati_->create())  // ...its backwards, I know.... MD
      {
	cmap2_pbuffer_->destroy();
	cmap2_shader_nv_->destroy();
	cmap2_shader_ati_->destroy();
	delete shader_factory_;
	shader_factory_ = 0;
	colormap2_hardware_destroy_buffers();
	return;
      }
    }

    raster_pbuffer_->set_use_default_shader(false);
    cmap2_pbuffer_->set_use_default_shader(false);
  }
  
  

  void
  TextureRenderer::build_colormap2()
  {
    if (!ShaderProgramARB::shaders_supported()) return;
    if (!cmap2_dirty_ && !alpha_dirty_) return;
    if(hardware_raster_)
      colormap2_hardware_rasterize_setup();
    else
      colormap2_hardware_destroy_buffers();

    if(cmap2_pbuffer_)
      colormap2_hardware_rasterize();
    else 
      colormap2_software_rasterize();

    CHECK_OPENGL_ERROR("TextureRenderer::build_colormap2()");
    
    cmap2_dirty_ = false;
    alpha_dirty_ = false;
  }

  void
  TextureRenderer::bind_colormap1(unsigned int cmap_tex)
  {
#if defined( GL_TEXTURE_COLOR_TABLE_SGI ) && defined(__sgi)
    glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
    glColorTable(GL_TEXTURE_COLOR_TABLE_SGI,
                 GL_RGBA,
                 256,
                 GL_RGBA,
                 GL_FLOAT,
                 &(cmap1_array_(0, 0)));
#elif defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
    if (ShaderProgramARB::shaders_supported() && glActiveTexture)
      {
        glActiveTexture(GL_TEXTURE2_ARB);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glEnable(GL_TEXTURE_1D);
        glBindTexture(GL_TEXTURE_1D, cmap_tex);
        // enable data texture unit 1
        glActiveTexture(GL_TEXTURE1_ARB);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glEnable(GL_TEXTURE_3D);
        glActiveTexture(GL_TEXTURE0_ARB);
      }
    else
#endif
#ifndef __sgi
#  if defined(GL_EXT_shared_texture_palette) && !defined(__APPLE__)
      {
        glEnable(GL_SHARED_TEXTURE_PALETTE_EXT);
        glColorTable(GL_SHARED_TEXTURE_PALETTE_EXT,
                     GL_RGBA,
                     256,
                     GL_RGBA,
                     GL_FLOAT,
                     &(cmap1_array_(0, 0)));
      }
#  else
    {
      static bool warned = false;
      if( !warned ) {
        std::cerr << "No volume colormaps available." << std::endl;
        warned = true;
      }
    }
#  endif
#endif
    CHECK_OPENGL_ERROR("");
  }


  void
  TextureRenderer::bind_colormap2()
  {
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
    if (ShaderProgramARB::shaders_supported() && glActiveTexture)
     {
        // bind texture to unit 2
        glActiveTexture(GL_TEXTURE2_ARB);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        if(cmap2_pbuffer_) {
          cmap2_pbuffer_->bind(GL_FRONT);
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
#endif
  }


  void
  TextureRenderer::release_colormap1()
  {
#if defined(GL_TEXTURE_COLOR_TABLE_SGI) && defined(__sgi)
    glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
#elif defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
    if (ShaderProgramARB::shaders_supported() && glActiveTexture)
      {
        // bind texture to unit 2
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
    else
#endif
#ifndef __sgi
#  if defined(GL_EXT_shared_texture_palette) && !defined(__APPLE__)
      {
        glDisable(GL_SHARED_TEXTURE_PALETTE_EXT);
      }
#  else
    {
      // Already warned in bind.  Do nothing.
    }
#  endif
#endif
    CHECK_OPENGL_ERROR("");
  }


  void
  TextureRenderer::release_colormap2()
  {
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
    if (ShaderProgramARB::shaders_supported() && glActiveTexture)
      {
        glActiveTexture(GL_TEXTURE2_ARB);
        if (cmap2_pbuffer_) {
	  cmap2_pbuffer_->release(GL_FRONT);
	} else {
	  glDisable(GL_TEXTURE_2D);
	  glBindTexture(GL_TEXTURE_2D, 0);
	}
        glActiveTexture(GL_TEXTURE1_ARB);
        glDisable(GL_TEXTURE_3D);
        glBindTexture(GL_TEXTURE_3D, 0);
        glActiveTexture(GL_TEXTURE0_ARB);
      }
#endif
  }

} // namespace SCIRun




