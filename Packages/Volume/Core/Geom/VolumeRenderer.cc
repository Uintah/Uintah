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
#include <sstream>
#include <Core/Geom/GeomOpenGL.h>
#include <Packages/Volume/Core/Geom/VolumeRenderer.h>
#include <Packages/Volume/Core/Util/ShaderProgramARB.h>
#include <Packages/Volume/Core/Util/SliceTable.h>
#include <Packages/Volume/Core/Util/Pbuffer.h>
#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Core/Util/DebugStream.h>

using std::string;
using std::ostringstream;

namespace Volume {

static SCIRun::DebugStream dbg("VolumeRenderer", false);

#define VOL_HEAD \
"!!ARBfp1.0 \n"

#define VOL_TAIL \
"END"

#define VOL_VLUP_HEAD \
"ATTRIB t = fragment.texcoord[0]; \n" \
"TEMP v; \n"
#define VOL_VLUP_1_1 \
"TEX v, t, texture[0], 3D; \n"
#define VOL_VLUP_1_4 \
"TEX v.w, t, texture[0], 3D; \n"
#define VOL_VLUP_2_1 VOL_VLUP_1_1
#define VOL_VLUP_2_4 VOL_VLUP_1_4
#define VOL_GLUP_2_1 \
"TEX v.y, t, texture[1], 3D; \n"
#define VOL_GLUP_2_4 \
"TEX v.x, t, texture[1], 3D; \n"

#define VOL_TFLUP_HEAD \
"TEMP c; \n"
#define VOL_TFLUP_1_1 \
"TEX c, v.x, texture[2], 1D; \n"
#define VOL_TFLUP_1_4 \
"TEX c, v.w, texture[2], 1D; \n"
#define VOL_TFLUP_2_1 \
"TEX c, v, texture[2], 2D; \n"
#define VOL_TFLUP_2_4 \
"TEX c, v.wxyz, texture[2], 2D; \n"

#define VOL_FOG_HEAD \
"PARAM fc = state.fog.color; \n" \
"PARAM fp = state.fog.params; \n" \
"ATTRIB tf = fragment.texcoord[1];\n"
#define VOL_FOG_BODY \
"SUB v.x, fp.z, tf.x; \n" \
"MUL_SAT v.x, v.x, fp.w; \n" \
"LRP c.xyz, v.x, c.xyzz, fc.xyzz; \n"

#define VOL_LIT_HEAD \
"PARAM l = program.local[0]; # {lx, ly, lz, alpha} \n" \
"PARAM k = program.local[1]; # {ka, kd, ks, ns} \n" \
"TEMP n; \n"
#define VOL_LIT_BODY \
"MAD n, v, 2.0, -1.0; \n" \
"DP3 n.w, n, n; \n" \
"RSQ n.w, n.w; \n" \
"MUL n, n, n.w; \n" \
"DP3 n.w, l, n; \n" \
"ABS_SAT n.w, n.w; # two-sided lighting \n" \
"POW n.z, n.w, k.w; \n" \
"MAD n.w, n.w, k.y, k.x; \n" \
"MAD n.w, n.z, k.z, n.w; \n"
#define VOL_LIT_END \
"MUL c.xyz, c.xyzz, n.w; \n"

#define VOL_FRAGMENT_BLEND_HEAD \
"TEMP n;"

#define VOL_FRAGMENT_BLEND_OVER \
"TEX v, fragment.position.xyyy, texture[3], RECT; \n" \
"SUB n.w, 1.0, c.w; \n" \
"MAD result.color, v, n.w, c; \n"

#define VOL_FRAGMENT_BLEND_MIP \
"TEX v, fragment.position.xyyy, texture[3], RECT; \n" \
"MAX result.color, v, c; \n"

#define VOL_RASTER_BLEND \
"MOV result.color, c; \n"

class VolShader
{
public:
  VolShader(int dim, int vsize, bool shading, bool fog, int blend);
  ~VolShader();

  bool create();
  
  inline int dim() { return dim_; }
  inline int vsize() { return vsize_; }
  inline bool shading() { return shading_; }
  inline bool fog() { return fog_; }
  inline bool blend() { return blend_; }

  inline bool match(int dim, int vsize, bool shading, bool fog, int blend)
  { return dim_ == dim && vsize_ == vsize && shading_ == shading
      && fog_ == fog && blend_ == blend; }

  inline FragmentProgramARB* program() { return program_; }
  
protected:
  bool emit(string& s);

  int dim_;
  int vsize_;
  bool shading_;
  bool fog_;
  int blend_;
  FragmentProgramARB* program_;
};

VolShader::VolShader(int dim, int vsize, bool shading, bool fog, int blend)
  : dim_(dim), vsize_(vsize), shading_(shading), fog_(fog), blend_(blend),
    program_(0)
{}

VolShader::~VolShader()
{
  delete program_;
}

bool
VolShader::create()
{
  string s;
  if(emit(s)) return true;
  program_ = new FragmentProgramARB(s);
  return false;
}

bool
VolShader::emit(string& s)
{
  if(dim_!=1 && dim_!=2) return true;
  if(vsize_!=1 && vsize_!=4) return true;
  if(blend_!=0 && blend_!=1 && blend_!=2) return true;
  ostringstream z;
  z << VOL_HEAD;
  z << VOL_VLUP_HEAD;
  z << VOL_TFLUP_HEAD;
  // dim, vsize, and shading
  if(shading_) {
    z << VOL_LIT_HEAD;
  }
  if(fog_) {
    z << VOL_FOG_HEAD;
  }
  if(dim_ == 1) {
    if(shading_) {
      z << VOL_VLUP_1_1;
      z << VOL_LIT_BODY;
      z << VOL_TFLUP_1_4;
      z << VOL_LIT_END;
    } else { // !shading_
      if(blend_) {
        z << VOL_FRAGMENT_BLEND_HEAD;
      }
      if(vsize_ == 1) {
        z << VOL_VLUP_1_1;
        z << VOL_TFLUP_1_1;
      } else { // vsize_ == 4
        z << VOL_VLUP_1_4;
        z << VOL_TFLUP_1_4;
      }
    }
  } else { // dim_ == 2
    if(shading_) {
      z << VOL_VLUP_2_1;
      z << VOL_LIT_BODY;
      z << VOL_GLUP_2_4;
      z << VOL_TFLUP_2_4;
      z << VOL_LIT_END;
    } else { // !shading_
      if(blend_) {
        z << VOL_FRAGMENT_BLEND_HEAD;
      }
      if(vsize_ == 1) {
        z << VOL_VLUP_2_1;
        z << VOL_GLUP_2_1;
        z << VOL_TFLUP_2_1;
      } else { // vsize_ == 4
        z << VOL_VLUP_2_4;
        z << VOL_GLUP_2_4;
        z << VOL_TFLUP_2_4;
      }
    }
  }
  // fog
  if(fog_) {
    z << VOL_FOG_BODY;
  }
  // blend
  if(blend_ == 0) {
    z << VOL_RASTER_BLEND;
  } else if(blend_ == 1) {
    z << VOL_FRAGMENT_BLEND_OVER;
  } else if(blend_ == 2) {
    z << VOL_FRAGMENT_BLEND_MIP;
  }
  z << VOL_TAIL;

  s = z.str();
  return false;
}

class VolShaderFactory
{
public:
  VolShaderFactory();
  ~VolShaderFactory();
  
  FragmentProgramARB* shader(int dim, int vsize, bool shading, bool fog, bool blend);

protected:
  vector<VolShader*> shader_;
  int prev_shader_;
};

VolShaderFactory::VolShaderFactory()
  : prev_shader_(-1)
{}

VolShaderFactory::~VolShaderFactory()
{
  for(unsigned int i=0; i<shader_.size(); i++) {
    delete shader_[i];
  }
}

FragmentProgramARB*
VolShaderFactory::shader(int dim, int vsize, bool shading, bool fog, bool blend)
{
  if(prev_shader_ >= 0) {
    if(shader_[prev_shader_]->match(dim, vsize, shading, fog, blend)) {
      return shader_[prev_shader_]->program();
    }
  }
  for(unsigned int i=0; i<shader_.size(); i++) {
    if(shader_[i]->match(dim, vsize, shading, fog, blend)) {
      prev_shader_ = i;
      return shader_[i]->program();
    }
  }
  VolShader* s = new VolShader(dim, vsize, shading, fog, blend);
  if(s->create()) {
    delete s;
    return 0;
  }
  shader_.push_back(s);
  prev_shader_ = shader_.size()-1;
  return s->program();
}

using namespace Volume;
using SCIRun::DrawInfoOpenGL;

VolumeRenderer::VolumeRenderer(TextureHandle tex,
                               ColorMapHandle cmap1, Colormap2Handle cmap2):
  TextureRenderer(tex, cmap1, cmap2),
  shading_(false),
  ambient_(0.5),
  diffuse_(0.5),
  specular_(0.0),
  shine_(30.0),
  light_(0),
  adaptive_(true),
  shader_factory_(new VolShaderFactory())
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
  adaptive_(copy.adaptive_),
  shader_factory_(copy.shader_factory_)
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
  //AuditAllocator(default_allocator);
  if(!pre_draw(di, mat, shading_)) return;
  mutex_.lock();
  di_ = di;
  if(di->get_drawtype() == DrawInfoOpenGL::WireFrame ) {
    draw_wireframe();
  } else {
    //AuditAllocator(default_allocator);
    draw();
  }
  di_ = 0;
  mutex_.unlock();
}

void
VolumeRenderer::draw()
{
  Ray viewRay;
  compute_view(viewRay);

  if(adaptive_ && ((cmap2_.get_rep() && cmap2_->updating()) || di_->mouse_action))
    set_interactive_mode(true);
  else
    set_interactive_mode(false);
  
  vector<Brick*> bricks;
  tex_->get_sorted_bricks(bricks, viewRay);
  vector<Brick*>::iterator it = bricks.begin();
  vector<Brick*>::iterator it_end = bricks.end();
  BBox brickbounds;
  tex_->get_bounds(brickbounds);
  Vector data_size(tex_->max()-tex_->min()+Vector(1.0,1.0,1.0));
  double rate = imode_ ? irate_ : sampling_rate_;
  int slices = (int)(data_size.length()*rate);
  SliceTable st(brickbounds.min(), brickbounds.max(), viewRay, slices);
  if(bricks.size() == 0) return;
  
  //--------------------------------------------------------------------------

  int nc = (*bricks.begin())->data()->nc();
  int nb0 = (*bricks.begin())->data()->nb(0);
  bool use_cmap2 = cmap2_.get_rep() && nc == 2;
  bool use_shading = shading_ && nb0 == 4;
  GLboolean use_fog;
  glGetBooleanv(GL_FOG, &use_fog);

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
  // now set up blending

  int vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  int psize[2];
  psize[0] = isPowerOf2(vp[2]) ? vp[2] : nextPowerOf2(vp[2]);
  psize[1] = isPowerOf2(vp[3]) ? vp[3] : nextPowerOf2(vp[3]);
    
//   if(num_bits_ != 8) {
//     if(!comp_buffer_ || num_bits_ != comp_buffer_->num_color_bits()
//        || psize[0] != comp_buffer_->width()
//        || psize[1] != comp_buffer_->height()) {
//       comp_buffer_ = new Pbuffer(psize[0], psize[1], GL_FLOAT, num_bits_, true,
//                                  GL_FALSE, GL_DONT_CARE, 24);
//       if(comp_buffer_->create()) {
//         comp_buffer_->destroy();
//         delete comp_buffer_;
//         comp_buffer_ = 0;
//         num_bits = 8;
//       }
//     }
//   }
  
//   if(num_bits_ == 8) {
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
//   } else {
//     glActiveTexture(GL_TEXTURE3);
//     //glTexEnvi();
    
//     glActiveTexture(GL_TEXTURE0);
//   }
  
  glColor4f(1.0, 1.0, 1.0, 1.0);
  glDepthMask(GL_FALSE);

  //--------------------------------------------------------------------------
  // set up shaders
  FragmentProgramARB* shader = 0;
  //int blend_mode = (num_bits == 8) ? 0 : (mode_ == MODE_OVER ? 1 : 2);
  int blend_mode = 0;
  shader = shader_factory_->shader(use_cmap2 ? 2 : 1, nb0, use_shading,
                                   use_fog, blend_mode);
  if(shader) {
    if(!shader->valid()) {
      shader->create();
    }
    shader->bind();
  }
  
  if(use_shading) {
    // set shader parameters
    GLfloat pos[4];
    glGetLightfv(GL_LIGHT0+light_, GL_POSITION, pos);
    Vector l(pos[0], pos[1], pos[2]);
    //cerr << "LIGHTING: " << pos << endl;
    double m[16], m_tp[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, m);
    for (int ii=0; ii<4; ii++)
      for (int jj=0; jj<4; jj++)
        m_tp[ii*4+jj] = m[jj*4+ii];
    Transform mv;
    mv.set(m_tp);
    Transform t = tex_->get_field_transform();
    l = mv.unproject(l);
    l = t.unproject(l);
    shader->setLocalParam(0, l.x(), l.y(), l.z(), 1.0);
    shader->setLocalParam(1, ambient_, diffuse_, specular_, shine_);
  }
  
  //--------------------------------------------------------------------------
  // render bricks
  vector<Polygon*> polys;
  vector<Polygon*>::iterator pit;
  for( ; it != it_end; it++ ) {
    for(pit = polys.begin(); pit != polys.end(); pit++) delete *pit;
    polys.clear();
    Brick& b = *(*it);
    double ts[8];
    for(int i=0; i<8; i++)
      ts[i] = intersectParam(-viewRay.direction(), b[i], viewRay);
    sortParameters(ts, 8);
    double tmin, tmax, dt;
    st.getParameters(b, tmin, tmax, dt);
    b.ComputePolys(viewRay, tmin, tmax, dt, ts, polys);
    load_brick(b);
    draw_polys(polys, use_fog);
  }

  //--------------------------------------------------------------------------
  // release shader

  if(shader && shader->valid())
    shader->release();
  
  //--------------------------------------------------------------------------

  glDisable(GL_BLEND);
  glDepthMask(GL_TRUE);
  // glEnable(GL_DEPTH_TEST);  

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


  // Look for errors
  GLenum errcode;
  if((errcode=glGetError()) != GL_NO_ERROR) {
    cerr << "VolumeRenderer::end | "
         << (char*)gluErrorString(errcode) << "\n";
  }
}

void
VolumeRenderer::draw_wireframe()
{
  Ray viewRay;
  compute_view(viewRay);
  TextureHandle tex = tex_;
  Transform field_trans = tex_->get_field_transform();
  // set double array transposed.  Our matricies are stored transposed 
  // from OpenGL matricies.
  double mvmat[16];
  field_trans.get_trans(mvmat);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);
  glEnable(GL_DEPTH_TEST);
  int lighting;
  glGetIntegerv(GL_LIGHTING, &lighting);
  glDisable(GL_LIGHTING);
  vector<Brick*> bricks;
  tex_->get_sorted_bricks(bricks, viewRay);
  vector<Brick*>::iterator it = bricks.begin();
  vector<Brick*>::iterator it_end = bricks.end();
  for(; it != it_end; ++it) {
    Brick& brick = *(*it);
    glColor4f(0.8, 0.8, 0.8, 1.0);
    glBegin(GL_LINES);
    for(int i=0; i<4; i++) {
      glVertex3d(brick[i].x(), brick[i].y(), brick[i].z());
      glVertex3d(brick[i+4].x(), brick[i+4].y(), brick[i+4].z());
    }
    glEnd();
    glBegin(GL_LINE_LOOP);
    glVertex3d(brick[0].x(), brick[0].y(), brick[0].z());
    glVertex3d(brick[1].x(), brick[1].y(), brick[1].z());
    glVertex3d(brick[3].x(), brick[3].y(), brick[3].z());
    glVertex3d(brick[2].x(), brick[2].y(), brick[2].z());
    glEnd();
    glBegin(GL_LINE_LOOP);
    glVertex3d(brick[4].x(), brick[4].y(), brick[4].z());
    glVertex3d(brick[5].x(), brick[5].y(), brick[5].z());
    glVertex3d(brick[7].x(), brick[7].y(), brick[7].z());
    glVertex3d(brick[6].x(), brick[6].y(), brick[6].z());
    glEnd();
  }
  if(lighting)
    glEnable(GL_LIGHTING);
  //glDisable(GL_DEPTH_TEST);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

#endif // SCI_OPENGL

} // End namespace Volume
