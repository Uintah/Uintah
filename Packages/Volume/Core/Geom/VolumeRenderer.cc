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
#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Core/Util/DebugStream.h>

using std::string;

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
"#MUL c.xyz, v.x, 0.1; \n"
"#MOV c.w, 0.1; \n"
"#MOV result.color, c; \n"
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

static const string TexShaderString =
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

using namespace Volume;
using SCIRun::DrawInfoOpenGL;

VolumeRenderer::VolumeRenderer() :
  slices_(64), slice_alpha_(0.5), mode_(OVEROP),
  shading_(false), ambient_(0.5), diffuse_(0.5), specular_(0.0),
  shine_(30.0), light_(0), pbuffer_(0), shader_factory_(0),
  texbuffer_(0), use_pbuffer_(true)
{
  VolShader1 = new FragmentProgramARB(ShaderString1);
  VolShader4 = new FragmentProgramARB(ShaderString4);
  FogVolShader1 = new FragmentProgramARB(FogShaderString1);
  FogVolShader4 = new FragmentProgramARB(FogShaderString4);
  LitVolShader = new FragmentProgramARB(LitVolShaderString);
  LitFogVolShader = new FragmentProgramARB(LitFogVolShaderString);
  VolShader1_2 = new FragmentProgramARB(ShaderString1_2);
  VolShader4_2 = new FragmentProgramARB(ShaderString4_2);
  FogVolShader1_2 = new FragmentProgramARB(FogShaderString1_2);
  FogVolShader4_2 = new FragmentProgramARB(FogShaderString4_2);
  LitVolShader_2 = new FragmentProgramARB(LitVolShaderString_2);
  LitFogVolShader_2 = new FragmentProgramARB(LitFogVolShaderString_2);
  FogVertexShader = new VertexProgramARB(FogVertexShaderString);
  texshader_ = new FragmentProgramARB(TexShaderString);
}

VolumeRenderer::VolumeRenderer(TextureHandle tex, ColorMapHandle cmap, Colormap2Handle cmap2):
  TextureRenderer(tex, cmap, cmap2),
  slices_(64), slice_alpha_(0.5), mode_(OVEROP),
  shading_(false), ambient_(0.5), diffuse_(0.5), specular_(0.0),
  shine_(30.0), light_(0), pbuffer_(0), shader_factory_(0),
  texbuffer_(0), use_pbuffer_(true)
{
  VolShader1 = new FragmentProgramARB(ShaderString1);
  VolShader4 = new FragmentProgramARB(ShaderString4);
  FogVolShader1 = new FragmentProgramARB(FogShaderString1);
  FogVolShader4 = new FragmentProgramARB(FogShaderString4);
  LitVolShader = new FragmentProgramARB(LitVolShaderString);
  LitFogVolShader = new FragmentProgramARB(LitFogVolShaderString);
  VolShader1_2 = new FragmentProgramARB(ShaderString1_2);
  VolShader4_2 = new FragmentProgramARB(ShaderString4_2);
  FogVolShader1_2 = new FragmentProgramARB(FogShaderString1_2);
  FogVolShader4_2 = new FragmentProgramARB(FogShaderString4_2);
  LitVolShader_2 = new FragmentProgramARB(LitVolShaderString_2);
  LitFogVolShader_2 = new FragmentProgramARB(LitFogVolShaderString_2);
  FogVertexShader = new VertexProgramARB(FogVertexShaderString);
  texshader_ = new FragmentProgramARB(TexShaderString);
}

VolumeRenderer::VolumeRenderer( const VolumeRenderer& copy):
  TextureRenderer(copy.tex_, copy.cmap_, copy.cmap2_),
  slices_(copy.slices_),
  slice_alpha_(copy.slice_alpha_),
  mode_(copy.mode_),
  pbuffer_(copy.pbuffer_),
  shader_factory_(copy.shader_factory_),
  texbuffer_(copy.texbuffer_),
  use_pbuffer_(copy.use_pbuffer_)
{
  VolShader1 = copy.VolShader1;
  VolShader4 = copy.VolShader4;
  FogVolShader1 = copy.FogVolShader1;
  FogVolShader4 = copy.FogVolShader4;
  LitVolShader = copy.LitVolShader;
  LitFogVolShader = copy.LitFogVolShader;
  VolShader1_2 = copy.VolShader1_2;
  VolShader4_2 = copy.VolShader4_2;
  FogVolShader1_2 = copy.FogVolShader1_2;
  FogVolShader4_2 = copy.FogVolShader4_2;
  LitVolShader_2 = copy.LitVolShader_2;
  LitFogVolShader_2 = copy.LitFogVolShader_2;
  FogVertexShader = copy.FogVertexShader;
  texshader_ = copy.texshader_;
}

GeomObj*
VolumeRenderer::clone()
{
  return scinew VolumeRenderer(*this);
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
    drawWireFrame();
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
  GLenum errcode;

  Ray viewRay;
  compute_view( viewRay );

  vector<Brick*> bricks;
  tex_->get_sorted_bricks( bricks, viewRay );

  vector<Brick*>::iterator it = bricks.begin();
  vector<Brick*>::iterator it_end = bricks.end();

  BBox brickbounds;
  tex_->get_bounds( brickbounds );
  
  SliceTable st(brickbounds.min(), brickbounds.max(), viewRay, slices_);
  
  vector<Polygon* > polys;
  vector<Polygon* >::iterator pit;
  double tmin, tmax, dt;
  double ts[8];
  //Brick *brick;

  //--------------------------------------------------------------------------

  if(cmap2_.get_rep()) {
    if(cmap2_dirty_) {
      BuildTransferFunction2();
      cmap2_dirty_ = false;
    }
    glActiveTexture(GL_TEXTURE2_ARB);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    if(use_pbuffer_) {
      texbuffer_->bind(GL_FRONT);
    } else {
      glEnable(GL_TEXTURE_2D);
    }
    glActiveTexture(GL_TEXTURE1_ARB);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_3D);
    glActiveTexture(GL_TEXTURE0_ARB);
  }
  
  if(cmap_.get_rep()) {
    if(cmap_has_changed_ || r_count_ != 1) {
      BuildTransferFunction();
      // cmap_has_changed_ = false;
    }
    load_colormap();
    glActiveTexture(GL_TEXTURE2_ARB);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_1D);
    glActiveTexture(GL_TEXTURE0_ARB);
  }

  // First set up the Textures.
  glActiveTexture(GL_TEXTURE0_ARB);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glEnable(GL_TEXTURE_3D);

  glColor4f(1.0, 1.0, 1.0, 1.0);
  glDepthMask(GL_FALSE);
  
  // now set up the Blending
  glEnable(GL_BLEND);
  switch(mode_) {
  case OVEROP:
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    break;
  case MIP:
    glBlendEquation(GL_MAX);
    glBlendFunc(GL_ONE, GL_ONE);
    break;
  }
  
  //--------------------------------------------------------------------------

  if(mode_ == OVEROP) {
    GLboolean fog;
    glGetBooleanv(GL_FOG, &fog);

    int nc = (*bricks.begin())->data()->nc();
    int nb0 = (*bricks.begin())->data()->nb(0);
    
    if (fog) {
      if (!FogVertexShader->valid()) {
        FogVertexShader->create();
      }
      FogVertexShader->bind();
    }

    if(nc == 2 && cmap2_.get_rep()) {
      if (shading_ && nb0 == 4) {
        if(fog) {
          if (!LitFogVolShader_2->valid()) {
            LitFogVolShader_2->create();
          }
          LitFogVolShader_2->bind();
        } else {
          if (!LitVolShader_2->valid()) {
            LitVolShader_2->create();
          }
          LitVolShader_2->bind();
        }
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
        if(fog) {
          LitFogVolShader_2->setLocalParam(0, l.x(), l.y(), l.z(), 1.0);
          LitFogVolShader_2->setLocalParam(1, ambient_, diffuse_, specular_, shine_);
        } else {
          LitVolShader_2->setLocalParam(0, l.x(), l.y(), l.z(), 1.0);
          LitVolShader_2->setLocalParam(1, ambient_, diffuse_, specular_, shine_);
        }
      } else { // !shading
        if(fog) {
          switch (nb0) {
          case 1:
            if(!FogVolShader1_2->valid()) {
              FogVolShader1_2->create();
            }
            FogVolShader1_2->bind();
            break;
          case 4:
            if(!FogVolShader4_2->valid()) {
              FogVolShader4_2->create();
            }
            FogVolShader4_2->bind();
            break;
          }
        } else { // !fog
          switch(nb0) {
          case 1:
            if(!VolShader1_2->valid()) {
              VolShader1_2->create();
            }
            VolShader1_2->bind();
            break;
          case 4:
            if(!VolShader4_2->valid()) {
              VolShader4_2->create();
            }
            VolShader4_2->bind();
            break;
          }
        }
      }
    } else {// nc == 1
      if (shading_ && nb0 == 4) {
        if(fog) {
          if (!LitFogVolShader->valid()) {
            LitFogVolShader->create();
          }
          LitFogVolShader->bind();
        } else {
          if (!LitVolShader->valid()) {
            LitVolShader->create();
          }
          LitVolShader->bind();
        }
        // set shader parameters
        GLfloat pos[4];
        glGetLightfv(GL_LIGHT0+light_, GL_POSITION, pos);
        Vector l(pos[0], pos[1], pos[2]);
        dbg << pos << endl;
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
        if(fog) {
          LitFogVolShader->setLocalParam(0, l.x(), l.y(), l.z(), 1.0);
          LitFogVolShader->setLocalParam(1, ambient_, diffuse_, specular_, shine_);
        } else {
          LitVolShader->setLocalParam(0, l.x(), l.y(), l.z(), 1.0);
          LitVolShader->setLocalParam(1, ambient_, diffuse_, specular_, shine_);
        }
      } else { // !shading
        if(fog) {
          switch (nb0) {
          case 1:
            if(!FogVolShader1->valid()) {
              FogVolShader1->create();
            }
            FogVolShader1->bind();
            break;
          case 4:
            if(!FogVolShader4->valid()) {
              FogVolShader4->create();
            }
            FogVolShader4->bind();
            break;
          }
        } else { // !fog
          switch(nb0) {
          case 1:
            if(!VolShader1->valid()) {
              VolShader1->create();
            }
            VolShader1->bind();
            break;
          case 4:
            if(!VolShader4->valid()) {
              VolShader4->create();
            }
            VolShader4->bind();
            break;
          }
        }
      }
    }
  } else if(mode_ == MIP) {
    int nc = (*bricks.begin())->data()->nc();
    int nb0 = (*bricks.begin())->data()->nb(0);
    if(nc > 1) {
      if(nb0 == 4) {
        if (!VolShader4_2->valid()) {
          VolShader4_2->create();
        }
        VolShader4_2->bind();
      } else {
        if (!VolShader1_2->valid()) {
          VolShader1_2->create();
        }
        VolShader1_2->bind();
      }
    } else {
      if(nb0 == 4) {
        if (!VolShader4->valid()) {
          VolShader4->create();
        }
        VolShader4->bind();
      } else {
        if (!VolShader1->valid()) {
          VolShader1->create();
        }
        VolShader1->bind();
      }
    }
  }
  
  //--------------------------------------------------------------------------
  
  for( ; it != it_end; it++ ) {
    for(pit = polys.begin(); pit != polys.end(); pit++) delete *pit;
    polys.clear();
    Brick& b = *(*it);
    for(int i = 0; i < 8; i++)
      ts[i] = intersectParam(-viewRay.direction(), b[i], viewRay);
    sortParameters(ts, 8);
    st.getParameters(b, tmin, tmax, dt);
    b.ComputePolys(viewRay, tmin, tmax, dt, ts, polys);
    load_texture(b);
    drawPolys(polys);
  }

  //--------------------------------------------------------------------------

  if(mode_ == OVEROP) {
    GLboolean fog;
    glGetBooleanv(GL_FOG, &fog);
    int nc = (*bricks.begin())->data()->nc();
    int nb0 = (*bricks.begin())->data()->nb(0);
    //int nb1 = (*bricks.begin())->data()->nb(1);

    if (fog) {
      FogVertexShader->release();
    }

    if(nc == 2 && cmap2_.get_rep()) {
      switch(nb0) {
      case 1: {
        VolShader1_2->release();
      } break;
      case 4: {
        VolShader4_2->release();
      }
      default:
        break;
      }
    } else { // nc == 1
      if(shading_ && nb0 == 4) {
        if(fog) {
          LitFogVolShader->release();
        } else {
          LitVolShader->release();
        }
      } else {
        if(fog) {
          switch(nb0) {
          case 1:
            FogVolShader1->release();
            break;
          case 4:
            FogVolShader4->release();
            break;
          }
        } else {
          switch(nb0) {
          case 1:
            VolShader1->release();
            break;
          case 4:
            VolShader4->release();
            break;
          }
        }
      }
    }
  } else if(mode_ == MIP) {
    int nc = (*bricks.begin())->data()->nc();
    int nb0 = (*bricks.begin())->data()->nb(0);
    if(nc > 1) {
      if(nb0 == 4) {
        VolShader4_2->release();
      } else {
        VolShader1_2->release();
      }
    } else {
      if(nb0 == 4) {
        VolShader4->release();
      } else {
        VolShader1->release();
      }
    }
  }
  
  //--------------------------------------------------------------------------

  glDisable(GL_BLEND);
  glDepthMask(GL_TRUE);

  glActiveTexture(GL_TEXTURE2_ARB);
  if(cmap2_.get_rep()) {
    if(use_pbuffer_) {
      texbuffer_->release(GL_FRONT);
    } else {
      glDisable(GL_TEXTURE_2D);
    }
  } else {
    glDisable(GL_TEXTURE_1D);
  }
  glActiveTexture(GL_TEXTURE1_ARB);
  glDisable(GL_TEXTURE_3D);
  glActiveTexture(GL_TEXTURE0_ARB);
  glDisable(GL_TEXTURE_3D);
  // glEnable(GL_DEPTH_TEST);  
  
  // Look for errors
  if ((errcode=glGetError()) != GL_NO_ERROR)
  {
    cerr << "VolumeRenderer::end | "
         << (char*)gluErrorString(errcode)
         << "\n";
  }
}

void
VolumeRenderer::drawWireFrame()
{
  Ray viewRay;
  compute_view( viewRay );

  double mvmat[16];
  TextureHandle tex = tex_;
  Transform field_trans = tex_->get_field_transform();
  // set double array transposed.  Our matricies are stored transposed 
  // from OpenGL matricies.
  field_trans.get_trans(mvmat);
  
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);
  glEnable(GL_DEPTH_TEST);


  vector<Brick*> bricks;
  tex_->get_sorted_bricks( bricks, viewRay );
  vector<Brick*>::iterator it = bricks.begin();
  vector<Brick*>::iterator it_end = bricks.end();
  for(; it != it_end; ++it){
    Brick& brick = *(*it);
    glColor4f(0.8,0.8,0.8,1.0);

    glBegin(GL_LINES);
    for(int i = 0; i < 4; i++){
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

void 
VolumeRenderer::load_colormap()
{
  const unsigned char *arr = transfer_function_;

  glActiveTexture(GL_TEXTURE2);
  {
    glEnable(GL_TEXTURE_1D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if( cmap_texture_ == 0 || cmap_has_changed_ ){
      glDeleteTextures(1, &cmap_texture_);
      glGenTextures(1, &cmap_texture_);
      glBindTexture(GL_TEXTURE_1D, cmap_texture_);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);      
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexImage1D(GL_TEXTURE_1D, 0,
		   GL_RGBA,
		   256, 0,
		   GL_RGBA, GL_UNSIGNED_BYTE,
		   arr);
      cmap_has_changed_ = false;
    } else {
      glBindTexture(GL_TEXTURE_1D, cmap_texture_);
    }
  }
  glActiveTexture(GL_TEXTURE0_ARB);

  int errcode = glGetError(); 
  if (errcode != GL_NO_ERROR)
  {
    cerr << "VolumeRenderer::load_colormap | "
         << (char*)gluErrorString(errcode)
         << "\n";
  }
}
#endif // SCI_OPENGL

void
VolumeRenderer::BuildTransferFunction()
{
  const int tSize = 256;
  int defaultSamples = 512;
  float mul = 1.0/(tSize - 1);
  double bp = 0;
  if( mode_ != MIP) {
    bp = tan( 1.570796327 * (0.5 - slice_alpha_*0.49999));
  } else {
    bp = tan( 1.570796327 * 0.5 );
  }
  double sliceRatio =  defaultSamples/(double(slices_));
  for ( int j = 0; j < tSize; j++ )
  {
    const Color c = cmap_->getColor(j*mul);
    const double alpha = cmap_->getAlpha(j*mul);
    
    const double alpha1 = pow(alpha, bp);
    const double alpha2 = 1.0 - pow((1.0 - alpha1), sliceRatio);
    if( mode_ != MIP ) {
      transfer_function_[4*j + 0] = (unsigned char)(c.r()*alpha2*255);
      transfer_function_[4*j + 1] = (unsigned char)(c.g()*alpha2*255);
      transfer_function_[4*j + 2] = (unsigned char)(c.b()*alpha2*255);
      transfer_function_[4*j + 3] = (unsigned char)(alpha2*255);
    } else {
      transfer_function_[4*j + 0] = (unsigned char)(c.r()*alpha*255);
      transfer_function_[4*j + 1] = (unsigned char)(c.g()*alpha*255);
      transfer_function_[4*j + 2] = (unsigned char)(c.b()*alpha*255);
      transfer_function_[4*j + 3] = (unsigned char)(alpha*255);
    }
  }
}

void
VolumeRenderer::BuildTransferFunction2()
{
  if(use_pbuffer_ && !pbuffer_) {
    pbuffer_ = new Pbuffer(256, 64, GL_FLOAT, 32, true, GL_FALSE);
    texbuffer_ = new Pbuffer(256, 64, GL_INT, 8, true, GL_FALSE);
    shader_factory_ = new CM2ShaderFactory();
    if(pbuffer_->create() || texbuffer_->create() || texshader_->create()
       || shader_factory_->create()) {
      pbuffer_->destroy();
      texbuffer_->destroy();
      texshader_->destroy();
      shader_factory_->destroy();
      delete pbuffer_;
      delete texbuffer_;
      delete shader_factory_;
      pbuffer_ = 0;
      texbuffer_ = 0;
      shader_factory_ = 0;
      use_pbuffer_ = false;
    } else {
      pbuffer_->set_use_default_shader(false);
      texbuffer_->set_use_default_shader(false);
    }
  }

  if(use_pbuffer_) {

    pbuffer_->activate();

    glDrawBuffer(GL_FRONT);
    glViewport(0, 0, pbuffer_->width(), pbuffer_->height());

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(-1.0, -1.0, 0.0);
    glScalef(2.0, 2.0, 2.0);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    
    // rasterize widgets
    cmap2_->lock_widgets();
    vector<CM2Widget*> widgets = cmap2_->widgets();
    for (unsigned int i=0; i<widgets.size(); i++)
    {
      widgets[i]->rasterize(*shader_factory_);
    }
    cmap2_->unlock_widgets();
    
    glDisable(GL_BLEND);
    
    pbuffer_->swapBuffers();
    pbuffer_->deactivate();

    // opacity correction and quantization
    double bp = 0;
    if(mode_ != MIP) {
      bp = tan(1.570796327 * (0.5 - slice_alpha_*0.49999));
    } else {
      bp = tan(1.570796327 * 0.5 );
    }
    int defaultSamples = 512;
    double sliceRatio =  defaultSamples/(double(slices_));

    
    texbuffer_->activate();

    glDrawBuffer(GL_FRONT);
    glViewport(0, 0, texbuffer_->width(), texbuffer_->height());
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(-1.0, -1.0, 0.0);
    glScalef(2.0, 2.0, 2.0);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);

    texshader_->bind();
    texshader_->setLocalParam(0, bp, sliceRatio, 0.0, 0.0);

    glActiveTexture(GL_TEXTURE0);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    pbuffer_->bind(GL_FRONT);
    glBegin(GL_QUADS);
    {
      glTexCoord2f( 0.0,  0.0);
      glVertex2f( 0.0,  0.0);
      glTexCoord2f( 1.0,  0.0);
      glVertex2f( 1.0,  0.0);
      glTexCoord2f( 1.0,  1.0);
      glVertex2f( 1.0,  1.0);
      glTexCoord2f( 0.0,  1.0);
      glVertex2f( 0.0,  1.0);
    }
    glEnd();
    pbuffer_->release(GL_FRONT);
    texshader_->release();
    
    texbuffer_->swapBuffers();
    texbuffer_->deactivate();
  }
  
#if 0
  bool size_dirty = false;
  cmap2_->lock_array();
  Array3<float>& c = cmap2_->array();
  if(c.dim1() != cmap2_array_.dim1()
     || c.dim2() != cmap2_array_.dim2()) {
    cmap2_array_.resize(c.dim1(), c.dim2(), 4);
    size_dirty = true;
  }

  double bp = 0;
  if( mode_ != MIP) {
    bp = tan( 1.570796327 * (0.5 - slice_alpha_*0.49999));
  } else {
    bp = tan( 1.570796327 * 0.5 );
  }
  int defaultSamples = 512;
  double sliceRatio =  defaultSamples/(double(slices_));
  for(int i=0; i<c.dim1(); i++) {
    for(int j=0; j<c.dim2(); j++) {
      double alpha = c(i,j,3);
      if( mode_ != MIP ) {
        double alpha1 = pow(alpha, bp);
        double alpha2 = 1.0-pow(1.0-alpha1, sliceRatio);
        cmap2_array_(i,j,0) = (unsigned char)(c(i,j,0)*alpha2*255);
        cmap2_array_(i,j,1) = (unsigned char)(c(i,j,1)*alpha2*255);
        cmap2_array_(i,j,2) = (unsigned char)(c(i,j,2)*alpha2*255);
        cmap2_array_(i,j,3) = (unsigned char)(alpha2*255);
      } else {
        cmap2_array_(i,j,0) = (unsigned char)(c(i,j,0)*alpha*255);
        cmap2_array_(i,j,1) = (unsigned char)(c(i,j,1)*alpha*255);
        cmap2_array_(i,j,2) = (unsigned char)(c(i,j,2)*alpha*255);
        cmap2_array_(i,j,3) = (unsigned char)(alpha*255);
      }
    }
  }
  if(size_dirty) {
    if(glIsTexture(cmap2_texture_)) {
      glDeleteTextures(1, &cmap2_texture_);
      cmap2_texture_ = 0;
    }
    glGenTextures(1, &cmap2_texture_);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, cmap2_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cmap2_array_.dim2(), cmap2_array_.dim1(),
                 0, GL_RGBA, GL_UNSIGNED_BYTE, &cmap2_array_(0,0,0));
  } else {
    glBindTexture(GL_TEXTURE_2D, cmap2_texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cmap2_array_.dim2(), cmap2_array_.dim1(),
                    GL_RGBA, GL_UNSIGNED_BYTE, &cmap2_array_(0,0,0));
  }
  cmap2_->lock_array();
#endif
}
