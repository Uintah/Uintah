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
#include <Packages/Volume/Core/Datatypes/TypedBrickData.h>
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

using Volume::Brick;

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
                                 ColorMapHandle cmap1, Colormap2Handle cmap2) :
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
  blend_numbits_(8)
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
  blend_numbits_(copy.blend_numbits_)
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
TextureRenderer::set_blend_numbits(int b)
{
  mutex_.lock();
  blend_numbits_ = b;
  mutex_.unlock();
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

void
TextureRenderer::compute_view(Ray& ray)
{
  double mvmat[16];
  Transform mat;
  Vector view;
  Point viewPt;
      
  glGetDoublev( GL_MODELVIEW_MATRIX, mvmat);
  /* remember that the glmatrix is stored as
     0  4  8 12
     1  5  9 13
     2  6 10 14
     3  7 11 15 */
  
  // transform the view vector opposite the transform that we draw polys with,
  // so that polys are normal to the view post opengl draw.
  //  GLTexture3DHandle tex = volren->get_tex3d_handle();
  //  Transform field_trans = tex->get_field_transform();

  Transform field_trans = tex_->get_field_transform();

  // this is the world space view direction
  view = Vector(-mvmat[2], -mvmat[6], -mvmat[10]);

  // but this is the view space viewPt
  viewPt = Point(-mvmat[12], -mvmat[13], -mvmat[14]);

  viewPt = field_trans.unproject( viewPt );
  view = field_trans.unproject( view );

  /* set the translation to zero */
  mvmat[12]=mvmat[13] = mvmat[14]=0;
   

  /* The Transform stores it's matrix as
     0  1  2  3
     4  5  6  7
     8  9 10 11
     12 13 14 15

     Because of this order, simply setting the tranform with the glmatrix 
     causes our tranform matrix to be the transpose of the glmatrix
     ( assuming no scaling ) */
  mat.set( mvmat );
    
  /* Since mat is the transpose, we then multiply the view space viewPt
     by the mat to get the world or model space viewPt, which we need
     for calculations */
  viewPt = mat.project( viewPt );
 
  ray =  Ray(viewPt, view);
}


void
TextureRenderer::load_brick(Brick& brick)
{
  TypedBrickData<unsigned char> *br =
    dynamic_cast<TypedBrickData<unsigned char>*>(brick.data());
  
  if(br) {
    if(!brick.texName(0) || (br->nc() > 1 && !brick.texName(1)) || brick.needsReload()) {
      if(!brick.texName(0)) {
        glGenTextures(1, brick.texNameP(0));
      }
      brick.setReload(false);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_3D, brick.texName(0));
      if(interp_) {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      } else {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      }
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      switch (br->nb(0)) {
      case 1:
        glTexImage3D(GL_TEXTURE_3D, 0,
                     GL_LUMINANCE8,
                     br->nx(),
                     br->ny(),
                     br->nz(),
                     0,
                     GL_LUMINANCE, GL_UNSIGNED_BYTE,
                     br->texture(0));
        break;
      case 4:
        glTexImage3D(GL_TEXTURE_3D, 0,
                     GL_RGBA8,
                     br->nx(),
                     br->ny(),
                     br->nz(),
                     0,
                     GL_RGBA, GL_UNSIGNED_BYTE,
                     br->texture(0));
        break;
      default:
        break;
      }
      if(br->nc() > 1) {
        if(!brick.texName(1)) {
          glGenTextures(1, brick.texNameP(1));
        }
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, brick.texName(1));
        if(interp_) {
          glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        } else {
          glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
          glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage3D(GL_TEXTURE_3D, 0,
                     GL_LUMINANCE8,
                     br->nx(),
                     br->ny(),
                     br->nz(),
                     0,
                     GL_LUMINANCE, GL_UNSIGNED_BYTE,
                     br->texture(1));
      }
    } else {
      glActiveTexture(GL_TEXTURE0_ARB);
      glBindTexture(GL_TEXTURE_3D, brick.texName(0));
      if(interp_) {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      } else {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      }
      if(br->nc() > 1) {
        glActiveTexture(GL_TEXTURE1_ARB);
        glBindTexture(GL_TEXTURE_3D, brick.texName(1));
        if(interp_) {
          glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        } else {
          glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
          glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }
        glActiveTexture(GL_TEXTURE0_ARB);
      }
    }
  }
  int errcode = glGetError();
  if (errcode != GL_NO_ERROR)
  {
    cerr << "VolumeRenderer::load_texture | "
         << (char*)gluErrorString(errcode)
         << "\n";
  }
}

void
TextureRenderer::draw_polys(vector<Polygon *> polys, bool z)
{
  double mvmat[16];
  TextureHandle tex = tex_;
  Transform field_trans = tex_->get_field_transform();
  // set double array transposed.  Our matricies are stored transposed 
  // from OpenGL matricies.
  field_trans.get_trans(mvmat);
  
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);

  glGetDoublev(GL_MODELVIEW_MATRIX, mvmat);
  
  Point p0, t0;
  unsigned int i;
  unsigned int k;
  di_->polycount += polys.size();
  for (i = 0; i < polys.size(); i++) {
    switch (polys[i]->size() ) {
    case 1:
      t0 = polys[i]->getTexCoord(0);
      p0 = polys[i]->getVertex(0);
      glBegin(GL_POINTS);
      if(z) {
        double pz = mvmat[2]*p0.x()+mvmat[6]*p0.y()+mvmat[10]*p0.z()+mvmat[14];
        glMultiTexCoord3f(GL_TEXTURE1, -pz, 0.0, 0.0);
      }
      glMultiTexCoord3f(GL_TEXTURE0, t0.x(), t0.y(), t0.z());
      glVertex3f(p0.x(), p0.y(), p0.z());
      glEnd();
      break;
    case 2:
      glBegin(GL_LINES);
      for(k =0; k < (unsigned int)polys[i]->size(); k++)
      {
        t0 = polys[i]->getTexCoord(k);
        p0 = polys[i]->getVertex(k);
        if(z) {
          double pz = mvmat[2]*p0.x()+mvmat[6]*p0.y()+mvmat[10]*p0.z()+mvmat[14];
          glMultiTexCoord3f(GL_TEXTURE1, -pz, 0.0, 0.0);
        }
        glMultiTexCoord3f(GL_TEXTURE0, t0.x(), t0.y(), t0.z());
        glVertex3f(p0.x(), p0.y(), p0.z());
      }
      glEnd();
      break;
    case 3:
      {
        Vector n = Cross(Vector((*(polys[i]))[0] - (*polys[i])[1]),
                         Vector((*(polys[i]))[0] - (*polys[i])[2]));
        n.normalize();
        glBegin(GL_TRIANGLES);
        glNormal3f(n.x(), n.y(), n.z());
        for(k =0; k < (unsigned int)polys[i]->size(); k++)
        {
          t0 = polys[i]->getTexCoord(k);
          p0 = polys[i]->getVertex(k);
          if(z) {
            double pz = mvmat[2]*p0.x()+mvmat[6]*p0.y()+mvmat[10]*p0.z()+mvmat[14];
            glMultiTexCoord3f(GL_TEXTURE1, -pz, 0.0, 0.0);
          }
          glMultiTexCoord3f(GL_TEXTURE0, t0.x(), t0.y(), t0.z());
          glVertex3f(p0.x(), p0.y(), p0.z());
        }
	glEnd();
      }
      break;
    case 4:
    case 5:
    case 6:
      {
	int k;
	glBegin(GL_POLYGON);
	Vector n = Cross(Vector((*(polys[i]))[0] - (*polys[i])[1]),
			 Vector((*(polys[i]))[0] - (*polys[i])[2]));
	n.normalize();
	glNormal3f(n.x(), n.y(), n.z());
	for(k =0; k < polys[i]->size(); k++)
	{
	  t0 = polys[i]->getTexCoord(k);
	  p0 = polys[i]->getVertex(k);
          if(z) {
            double pz = mvmat[2]*p0.x()+mvmat[6]*p0.y()+mvmat[10]*p0.z()+mvmat[14];
            glMultiTexCoord3f(GL_TEXTURE1, -pz, 0.0, 0.0);
          }
          glMultiTexCoord3f(GL_TEXTURE0, t0.x(), t0.y(), t0.z());
	  glVertex3f(p0.x(), p0.y(), p0.z());
	}
	glEnd();
	break;
      }
    }
  }
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
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
            raster_array_(i,j,0) = CLAMP(raster_array_(i,j,0), 0.0f, 1.0f);
            raster_array_(i,j,1) = CLAMP(raster_array_(i,j,1), 0.0f, 1.0f);
            raster_array_(i,j,2) = CLAMP(raster_array_(i,j,2), 0.0f, 1.0f);
            raster_array_(i,j,3) = CLAMP(raster_array_(i,j,3), 0.0f, 1.0f);
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
