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
//    File   : VolShader.cc
//    Author : Milan Ikits
//    Date   : Tue Jul 13 02:28:09 2004

#include <sstream>
#include <Packages/Volume/Core/Geom/VolShader.h>
#include <Packages/Volume/Core/Util/ShaderProgramARB.h>

using std::string;
using std::vector;
using std::ostringstream;

namespace Volume {

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

#define VOL_FRAG_HEAD \
"ATTRIB cf = fragment.color; \n"
#define VOL_FRAG_BODY \
"MUL c, c, cf; \n"

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
"MUL n.z, k.z, n.z; \n"
#define VOL_LIT_END \
"MUL n.z, n.z, c.w;" \
"MAD c.xyz, c.xyzz, n.w, n.z; \n"

// "MAD n.w, n.z, k.z, n.w; \n"
// #define VOL_LIT_END \
// "MUL c.xyz, c.xyzz, n.w; \n"

#define VOL_FRAGMENT_BLEND_HEAD \
"TEMP n;"

#define VOL_FRAGMENT_BLEND_OVER_NV \
"TEX v, fragment.position.xyyy, texture[3], RECT; \n" \
"SUB n.w, 1.0, c.w; \n" \
"MAD result.color, v, n.w, c; \n"
#define VOL_FRAGMENT_BLEND_MIP_NV \
"TEX v, fragment.position.xyyy, texture[3], RECT; \n" \
"MAX result.color, v, c; \n"

#define VOL_FRAGMENT_BLEND_OVER_ATI \
"MUL n.xy, fragment.position.xyyy, program.local[2].xyyy;\n" \
"TEX v, n.xyyy, texture[3], 2D; \n" \
"SUB n.w, 1.0, c.w; \n" \
"MAD result.color, v, n.w, c; \n"
#define VOL_FRAGMENT_BLEND_MIP_ATI \
"MUL n.xy, fragment.position.xyyy, program.local[2].xyyy;\n" \
"TEX v, n.xyyy, texture[3], 2D; \n" \
"MAX result.color, v, c; \n"

#define VOL_RASTER_BLEND \
"MOV result.color, c; \n"

VolShader::VolShader(int dim, int vsize, bool shading, bool frag, bool fog, int blend)
  : dim_(dim), vsize_(vsize), shading_(shading), fog_(fog), blend_(blend),
    frag_(frag), program_(0)
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
  if(frag_) {
    z << VOL_FRAG_HEAD;
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
  // frag
  if(frag_) {
    z << VOL_FRAG_BODY;
  }
  // fog
  if(fog_) {
    z << VOL_FOG_BODY;
  }
  // blend
  if(blend_ == 0) {
    z << VOL_RASTER_BLEND;
  } else if(blend_ == 1) {
    z << VOL_FRAGMENT_BLEND_OVER_NV;
  } else if(blend_ == 2) {
    z << VOL_FRAGMENT_BLEND_MIP_NV;
  } else if(blend_ == 3) {
    z << VOL_FRAGMENT_BLEND_OVER_ATI;
  } else if(blend_ == 4) {
    z << VOL_FRAGMENT_BLEND_MIP_ATI;
  }
  z << VOL_TAIL;

  s = z.str();
  return false;
}


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
VolShaderFactory::shader(int dim, int vsize, bool shading, bool frag, bool fog,
                         int blend)
{
  if(prev_shader_ >= 0) {
    if(shader_[prev_shader_]->match(dim, vsize, shading, frag, fog, blend)) {
      return shader_[prev_shader_]->program();
    }
  }
  for(unsigned int i=0; i<shader_.size(); i++) {
    if(shader_[i]->match(dim, vsize, shading, frag, fog, blend)) {
      prev_shader_ = i;
      return shader_[i]->program();
    }
  }
  VolShader* s = new VolShader(dim, vsize, shading, frag, fog, blend);
  if(s->create()) {
    delete s;
    return 0;
  }
  shader_.push_back(s);
  prev_shader_ = shader_.size()-1;
  return s->program();
}

} // end namespace Volume
