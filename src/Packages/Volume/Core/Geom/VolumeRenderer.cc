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
#include <Core/Geom/GeomOpenGL.h>
#include <Packages/Volume/Core/Geom/VolumeRenderer.h>
#include <Packages/Volume/Core/Util/ShaderProgramARB.h>
#include <Packages/Volume/Core/Util/SliceTable.h>
#include <Packages/Volume/Core/Util/Pbuffer.h>
#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Core/Util/DebugStream.h>

using std::string;

namespace Volume {

static SCIRun::DebugStream dbg("VolumeRenderer", false);

static const string ShaderString1 =
"!!ARBfp1.0 \n"
"TEMP v; \n"
"ATTRIB t = fragment.texcoord[0]; \n"
"TEX v, t, texture[0], 3D; \n"
"TEX result.color, v, texture[2], 1D; \n"
"END";

static const string ShaderString4 =
"!!ARBfp1.0 \n"
"TEMP v; \n"
"ATTRIB t = fragment.texcoord[0]; \n"
"TEX v, t, texture[0], 3D; \n"
"TEX result.color, v.w, texture[2], 1D; \n"
"END";

static const string ShaderString1_2 =
"!!ARBfp1.0 \n"
"TEMP v, c; \n"
"ATTRIB t = fragment.texcoord[0]; \n"
"TEX v.x, t, texture[0], 3D; \n"
"TEX v.y, t, texture[1], 3D; \n"
"TEX result.color, v, texture[2], 2D; \n"
"END";

static const string ShaderString4_2 =
"!!ARBfp1.0 \n"
"TEMP v, c; \n"
"ATTRIB t = fragment.texcoord[0]; \n"
"TEX v.w, t, texture[0], 3D; \n"
"TEX v.x, t, texture[1], 3D; \n"
"TEX result.color, v.wxyz, texture[2], 2D; \n"
"END";

//fogParam = {density, start, end, 1/(end-start) 
//fogCoord.x = z
//f = (end - fogCoord)/(end-start)
static const string FogShaderString1 =
"!!ARBfp1.0 \n"
"TEMP c0, c, fogFactor, finalColor; \n"
"PARAM fogColor = state.fog.color; \n"
"PARAM fogParam = state.fog.params; \n"
"ATTRIB fogCoord = fragment.texcoord[1];\n"
"# this does not work: ATTRIB fogCoord = fragment.fogcoord; \n"
"ATTRIB tf = fragment.texcoord[0]; \n"
"SUB c, fogParam.z, fogCoord.x; \n"
"MUL_SAT fogFactor.x, c, fogParam.w; \n"
"TEX c0, tf, texture[0], 3D; \n"
"TEX finalColor, c0, texture[2], 1D; \n"
"LRP finalColor.xyz, fogFactor.x, finalColor.xyzz, fogColor.xyzz; \n"
"MOV result.color, finalColor; \n"
"END";

static const string FogShaderString1_2 =
"!!ARBfp1.0 \n"
"TEMP v, c, fogFactor, finalColor; \n"
"PARAM fogColor = state.fog.color; \n"
"PARAM fogParam = state.fog.params; \n"
"ATTRIB fogCoord = fragment.texcoord[1];\n"
"# this does not work: ATTRIB fogCoord = fragment.fogcoord; \n"
"ATTRIB tf = fragment.texcoord[0]; \n"
"SUB c, fogParam.z, fogCoord.x; \n"
"MUL_SAT fogFactor.x, c, fogParam.w; \n"
"TEX v.x, tf, texture[0], 3D; \n"
"TEX v.y, tf, texture[1], 3D; \n"
"TEX finalColor, v, texture[2], 2D; \n"
"LRP finalColor.xyz, fogFactor.x, finalColor.xyzz, fogColor.xyzz; \n"
"MOV result.color, finalColor; \n"
"END";

static const string FogShaderString4 =
"!!ARBfp1.0 \n"
"TEMP c0, c, fogFactor, finalColor; \n"
"PARAM fogColor = state.fog.color; \n"
"PARAM fogParam = state.fog.params; \n"
"ATTRIB fogCoord = fragment.texcoord[1];\n"
"# this does not work: ATTRIB fogCoord = fragment.fogcoord; \n"
"ATTRIB tf = fragment.texcoord[0]; \n"
"SUB c, fogParam.z, fogCoord.x; \n"
"MUL_SAT fogFactor.x, c, fogParam.w; \n"
"TEX c0, tf, texture[0], 3D; \n"
"TEX finalColor, c0.w, texture[2], 1D; \n"
"LRP finalColor.xyz, fogFactor.x, finalColor.xyzz, fogColor.xyzz; \n"
"MOV result.color, finalColor; \n"
"END";

static const string FogShaderString4_2 =
"!!ARBfp1.0 \n"
"TEMP v, c, fogFactor, finalColor; \n"
"PARAM fogColor = state.fog.color; \n"
"PARAM fogParam = state.fog.params; \n"
"ATTRIB fogCoord = fragment.texcoord[1];\n"
"# this does not work: ATTRIB fogCoord = fragment.fogcoord; \n"
"ATTRIB tf = fragment.texcoord[0]; \n"
"SUB c, fogParam.z, fogCoord.x; \n"
"MUL_SAT fogFactor.x, c, fogParam.w; \n"
"TEX v.w, tf, texture[0], 3D; \n"
"TEX v.x, tf, texture[1], 3D; \n"
"TEX finalColor, v.wxyz, texture[2], 2D; \n"
"LRP finalColor.xyz, fogFactor.x, finalColor.xyzz, fogColor.xyzz; \n"
"MOV result.color, finalColor; \n"
"END";

static const string LitVolShaderString =
"!!ARBfp1.0 \n"
"ATTRIB t = fragment.texcoord[0];\n"
"PARAM l = program.local[0]; # {lx, ly, lz, alpha} \n"
"PARAM k = program.local[1]; # {ka, kd, ks, ns} \n"
"TEMP v, n, c, d, s; \n"
"TEX v, t, texture[0], 3D; \n"
"MAD n, v, 2.0, -1.0; \n"
"DP3 n.w, n, n; \n"
"RSQ n.w, n.w; \n"
"MUL n, n, n.w; \n"
"DP3 d.w, l, n; \n"
"ABS_SAT d.w, d.w; # two-sided lighting \n"
"POW s.w, d.w, k.w; \n"
"MAD d.w, d.w, k.y, k.x; \n"
"MAD d.w, s.w, k.z, d.w; \n"
"TEX c, v.w, texture[2], 1D; \n"
"MUL c.xyz, c.xyzz, d.w; \n"
"MOV result.color, c; \n"
"END";

static const string LitVolShaderString_2 =
"!!ARBfp1.0 \n"
"ATTRIB t = fragment.texcoord[0];\n"
"PARAM l = program.local[0]; # {lx, ly, lz, alpha} \n"
"PARAM k = program.local[1]; # {ka, kd, ks, ns} \n"
"TEMP v, n, c, d, s; \n"
"TEX v, t, texture[0], 3D; \n"
"MAD n, v, 2.0, -1.0; \n"
"DP3 n.w, n, n; \n"
"RSQ n.w, n.w; \n"
"MUL n, n, n.w; \n"
"DP3 d.w, l, n; \n"
"ABS_SAT d.w, d.w; # two-sided lighting \n"
"POW s.w, d.w, k.w; \n"
"MAD d.w, d.w, k.y, k.x; \n"
"MAD d.w, s.w, k.z, d.w; \n"
"TEX v.x, t, texture[1], 3D; \n"
"TEX c, v.wxyz, texture[2], 2D; \n"
"MUL c.xyz, c.xyzz, d.w; \n"
"MOV result.color, c; \n"
"END";

static const string LitFogVolShaderString =
"!!ARBfp1.0 \n"
"ATTRIB t = fragment.texcoord[0];\n"
"PARAM l = program.local[0]; # {lx, ly, lz, alpha} \n"
"PARAM k = program.local[1]; # {ka, kd, ks, ns} \n"
"PARAM fc = state.fog.color; \n"
"PARAM fp = state.fog.params; \n"
"ATTRIB f = fragment.texcoord[1];\n"
"TEMP v, n, c, d, s; \n"
"TEX v, t, texture[0], 3D; \n"
"MAD n, v, 2.0, -1.0; \n"
"DP3 n.w, n, n; \n"
"RSQ n.w, n.w; \n"
"MUL n, n, n.w; \n"
"DP3 d.w, l, n; \n"
"ABS_SAT d.w, d.w; # two-sided lighting \n"
"POW s.w, d.w, k.w; \n"
"MAD d.w, d.w, k.y, k.x; \n"
"MAD d.w, s.w, k.z, d.w; \n"
"TEX c, v.w, texture[2], 1D; \n"
"MUL c.xyz, c.xyzz, d.w; \n"
"SUB d.x, fp.z, f.x; \n"
"MUL_SAT d.x, d.x, fp.w; \n"
"LRP c.xyz, d.x, c.xyzz, fc.xyzz; \n"
"MOV result.color, c; \n"
"END";

static const string LitFogVolShaderString_2 =
"!!ARBfp1.0 \n"
"ATTRIB t = fragment.texcoord[0];\n"
"PARAM l = program.local[0]; # {lx, ly, lz, alpha} \n"
"PARAM k = program.local[1]; # {ka, kd, ks, ns} \n"
"PARAM fc = state.fog.color; \n"
"PARAM fp = state.fog.params; \n"
"ATTRIB f = fragment.texcoord[1];\n"
"TEMP v, n, c, d, s; \n"
"TEX v, t, texture[0], 3D; \n"
"MAD n, v, 2.0, -1.0; \n"
"DP3 n.w, n, n; \n"
"RSQ n.w, n.w; \n"
"MUL n, n, n.w; \n"
"DP3 d.w, l, n; \n"
"ABS_SAT d.w, d.w; # two-sided lighting \n"
"POW s.w, d.w, k.w; \n"
"MAD d.w, d.w, k.y, k.x; \n"
"MAD d.w, s.w, k.z, d.w; \n"
"TEX v.x, t, texture[1], 3D; \n"
"TEX c, v.wxyz, texture[2], 2D; \n"
"MUL c.xyz, c.xyzz, d.w; \n"
"SUB d.x, fp.z, f.x; \n"
"MUL_SAT d.x, d.x, fp.w; \n"
"LRP c.xyz, d.x, c.xyzz, fc.xyzz; \n"
"MOV result.color, c; \n"
"END";

static const string FogVertexShaderString =
"!!ARBvp1.0 \n"
"ATTRIB iPos = vertex.position; \n"
"ATTRIB iTex0 = vertex.texcoord[0]; \n"
"OUTPUT oPos = result.position; \n"
"OUTPUT oTex0 = result.texcoord[0]; \n"
"OUTPUT oTex1 = result.texcoord[1]; \n"
"PARAM mvp[4] = { state.matrix.mvp }; \n"
"PARAM mv[4] = { state.matrix.modelview }; \n"
"MOV oTex0, iTex0; \n"
"DP4 oTex1.x, -mv[2], iPos; \n"
"DP4 oPos.x, mvp[0], iPos; \n"
"DP4 oPos.y, mvp[1], iPos; \n"
"DP4 oPos.z, mvp[2], iPos; \n"
"DP4 oPos.w, mvp[3], iPos; \n"
"END";

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
  adaptive_(true)
{
  mode_ = MODE_OVER;
  vol_shader1_ = new FragmentProgramARB(ShaderString1);
  vol_shader4_ = new FragmentProgramARB(ShaderString4);
  fog_vol_shader1_ = new FragmentProgramARB(FogShaderString1);
  fog_vol_shader4_ = new FragmentProgramARB(FogShaderString4);
  lit_vol_shader_ = new FragmentProgramARB(LitVolShaderString);
  lit_fog_vol_shader_ = new FragmentProgramARB(LitFogVolShaderString);
  vol_shader1_2_ = new FragmentProgramARB(ShaderString1_2);
  vol_shader4_2_ = new FragmentProgramARB(ShaderString4_2);
  fog_vol_shader1_2_ = new FragmentProgramARB(FogShaderString1_2);
  fog_vol_shader4_2_ = new FragmentProgramARB(FogShaderString4_2);
  lit_vol_shader_2_ = new FragmentProgramARB(LitVolShaderString_2);
  lit_fog_vol_shader_2_ = new FragmentProgramARB(LitFogVolShaderString_2);
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
{
  vol_shader1_ = copy.vol_shader1_;
  vol_shader4_ = copy.vol_shader4_;
  fog_vol_shader1_ = copy.fog_vol_shader1_;
  fog_vol_shader4_ = copy.fog_vol_shader4_;
  lit_vol_shader_ = copy.lit_vol_shader_;
  lit_fog_vol_shader_ = copy.lit_fog_vol_shader_;
  vol_shader1_2_ = copy.vol_shader1_2_;
  vol_shader4_2_ = copy.vol_shader4_2_;
  fog_vol_shader1_2_ = copy.fog_vol_shader1_2_;
  fog_vol_shader4_2_ = copy.fog_vol_shader4_2_;
  lit_vol_shader_2_ = copy.lit_vol_shader_2_;
  lit_fog_vol_shader_2_ = copy.lit_fog_vol_shader_2_;
}

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

  if(adaptive_ && ((cmap2_.get_rep() && cmap2_->is_updating()) || di_->mouse_action))
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
  
  glColor4f(1.0, 1.0, 1.0, 1.0);
  glDepthMask(GL_FALSE);

  //--------------------------------------------------------------------------
  // set up shaders
  FragmentProgramARB* fragment_shader = 0;

  if(mode_ == MODE_OVER) {
    if(use_cmap2) {
      if(use_shading) {
        if(use_fog) {
          fragment_shader = lit_fog_vol_shader_2_;
        } else {
          fragment_shader = lit_vol_shader_2_;
        }
      } else { // !use_shading
        if(use_fog) {
          switch(nb0) {
          case 1:
            fragment_shader = fog_vol_shader1_2_;
            break;
          case 4:
            fragment_shader = fog_vol_shader4_2_;
            break;
          }
        } else { // !use_fog
          switch(nb0) {
          case 1:
            fragment_shader = vol_shader1_2_;
            break;
          case 4:
            fragment_shader = vol_shader4_2_;
            break;
          }
        }
      }
    } else { // nc == 1
      if(use_shading) {
        if(use_fog) {
          fragment_shader = lit_fog_vol_shader_;
        } else {
          fragment_shader = lit_vol_shader_;
        }
      } else { // !use_shading
        if(use_fog) {
          switch (nb0) {
          case 1:
            fragment_shader = fog_vol_shader1_;
            break;
          case 4:
            fragment_shader = fog_vol_shader4_;
            break;
          }
        } else { // !use_fog
          switch(nb0) {
          case 1:
            fragment_shader = vol_shader1_;
            break;
          case 4:
            fragment_shader = vol_shader4_;
            break;
          }
        }
      }
    }
  } else if(mode_ == MODE_MIP) {
    if(use_cmap2) {
      switch(nb0) {
      case 1:
        fragment_shader = vol_shader1_2_;
        break;
      case 4:
        fragment_shader = vol_shader4_2_;
        break;
      }
    } else { // !use_cmap2
      switch(nb0) {
      case 1:
        fragment_shader = vol_shader1_;
        break;
      case 4:
        fragment_shader = vol_shader4_;
        break;
      }
    }
  }

  if(fragment_shader) {
    if(!fragment_shader->valid()) {
      fragment_shader->create();
    }
    fragment_shader->bind();
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
    fragment_shader->setLocalParam(0, l.x(), l.y(), l.z(), 1.0);
    fragment_shader->setLocalParam(1, ambient_, diffuse_, specular_, shine_);
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
  // release shaders

  if(fragment_shader && fragment_shader->valid())
    fragment_shader->release();
  
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
  glDisable(GL_DEPTH_TEST);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

#endif // SCI_OPENGL

} // End namespace Volume
