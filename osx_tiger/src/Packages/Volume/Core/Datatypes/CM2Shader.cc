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
//    File   : CM2Shader.cc
//    Author : Milan Ikits
//    Date   : Tue Jul 13 02:27:42 2004

#include <Packages/Volume/Core/Util/ShaderProgramARB.h>
#include <Packages/Volume/Core/Datatypes/CM2Shader.h>

#include <iostream>
#include <sstream>

using namespace std;

namespace Volume {

#define CM2_TRIANGLE_BASE \
"!!ARBfp1.0 \n" \
"PARAM color = program.local[0]; \n" \
"PARAM geom0 = program.local[1]; # {base, top_x, top_y, 0.0} \n" \
"PARAM geom1 = program.local[2]; # {width, bottom, 0.0, 0.0} \n" \
"PARAM sz = program.local[3]; # {1/sx, 1/sy, 0.0, 0.0} \n" \
"TEMP c, p, t;" \
"MUL p.xy, fragment.position.xyyy, sz.xyyy; \n" \
"MUL p.z, geom1.y, geom0.z; \n" \
"SUB p.z, p.y, p.z; \n" \
"KIL p.z; \n" \
"RCP t.z, geom0.z; \n" \
"MUL t.x, p.y, t.z; \n" \
"LRP c.x, t.x, geom0.y, geom0.x; \n" \
"MUL c.y, t.x, geom1.x; \n" \
"MUL c.y, c.y, 0.5; \n" \
"RCP c.y, c.y; \n" \
"SUB c.z, p.x, c.x; \n" \
"MUL c.z, c.y, c.z; \n" \
"ABS c.z, c.z; \n" \
"SUB t.w, 1.0, c.z; \n"

#define CM2_RECTANGLE_1D_BASE \
"!!ARBfp1.0 \n" \
"PARAM color = program.local[0]; \n" \
"PARAM geom0 = program.local[1]; # {left_x, left_y, width, height} \n" \
"PARAM geom1 = program.local[2]; # {offset, 1/offset, 1/(1-offset), 0.0} \n" \
"PARAM sz = program.local[3]; # {1/sx, 1/sy, 0.0, 0.0} \n" \
"TEMP c, p, t; \n" \
"MUL p.xy, fragment.position.xyyy, sz.xyyy; \n" \
"SUB p.xy, p.xyyy, geom0.xyyy; \n" \
"RCP p.z, geom0.z; \n" \
"RCP p.w, geom0.w; \n" \
"MUL p.xy, p.xyyy, p.zwww; \n" \
"SUB t.x, p.x, geom1.x; \n" \
"MUL t.y, t.x, geom1.y; \n" \
"MUL t.z, t.x, geom1.z; \n" \
"CMP t.w, t.y, t.y, t.z; \n" \
"ABS t.w, t.w; \n" \
"SUB t.w, 1.0, t.w; \n" \

#define CM2_RECTANGLE_ELLIPSOID_BASE \
"!!ARBfp1.0 \n" \
"PARAM color = program.local[0]; \n" \
"TEMP c, p, t;"

#define CM2_REGULAR \
"MUL c.w, color.w, t.w; \n" \
"MOV c.xyz, color.xyzz; \n"
#define CM2_FAUX \
"MUL c, color, t.w; \n"

#define CM2_RASTER_BLEND \
"MOV result.color, c; \n" \
"END"
#define CM2_FRAGMENT_BLEND_ATI \
"MUL p.xy, fragment.position.xyyy, program.local[4].xyyy; \n" \
"TEX t, p.xyyy, texture[0], 2D; \n" \
"SUB p.w, 1.0, c.w; \n" \
"MAD_SAT result.color, t, p.w, c; \n" \
"END"
#define CM2_FRAGMENT_BLEND_NV \
"TEX t, fragment.position.xyyy, texture[0], RECT; \n" \
"SUB p.w, 1.0, c.w; \n" \
"MAD_SAT result.color, t, p.w, c; \n" \
"END"

CM2Shader::CM2Shader(CM2ShaderType type, bool faux, CM2BlendType blend)
  : type_(type), faux_(faux), blend_(blend),
    program_(0)
{}

CM2Shader::~CM2Shader()
{
  delete program_;
}

bool
CM2Shader::create()
{
  string s;
  if(emit(s)) return true;
  program_ = new FragmentProgramARB(s);
  return false;
}

bool
CM2Shader::emit(string& s)
{
  ostringstream z;
  switch(type_) {
  case CM2_SHADER_TRIANGLE:
    z << CM2_TRIANGLE_BASE;
    break;
  case CM2_SHADER_RECTANGLE_1D:
    z << CM2_RECTANGLE_1D_BASE;
    break;
  case CM2_SHADER_RECTANGLE_ELLIPSOID:
    z << CM2_RECTANGLE_ELLIPSOID_BASE;
    break;
  default:
    break;
  }
  if(faux_) {
    z << CM2_FAUX;
  } else {
    z << CM2_REGULAR;
  }
  switch(blend_) {
  case CM2_BLEND_RASTER:
    z << CM2_RASTER_BLEND;
    break;
  case CM2_BLEND_FRAGMENT_ATI:
    z << CM2_FRAGMENT_BLEND_ATI;
    break;
  case CM2_BLEND_FRAGMENT_NV:
    z << CM2_FRAGMENT_BLEND_NV;
    break;
  default:
    break;
  }
  
  s = z.str();
  return false;
}

CM2ShaderFactory::CM2ShaderFactory()
  : prev_shader_(-1)
{}

CM2ShaderFactory::~CM2ShaderFactory()
{
  for(unsigned int i=0; i<shader_.size(); i++) {
    delete shader_[i];
  }
}

void
CM2ShaderFactory::destroy()
{
  for(unsigned int i=0; i<shader_.size(); i++) {
    if(shader_[i]->program() && shader_[i]->program()->valid())
      shader_[i]->program()->destroy();
  }
}

FragmentProgramARB*
CM2ShaderFactory::shader(CM2ShaderType type, bool faux, CM2BlendType blend)
{
  if(prev_shader_ >= 0) {
    if(shader_[prev_shader_]->match(type, faux, blend)) {
      return shader_[prev_shader_]->program();
    }
  }
  for(unsigned int i=0; i<shader_.size(); i++) {
    if(shader_[i]->match(type, faux, blend)) {
      prev_shader_ = i;
      return shader_[i]->program();
    }
  }
  CM2Shader* s = new CM2Shader(type, faux, blend);
  if(s->create()) {
    delete s;
    return 0;
  }
  shader_.push_back(s);
  prev_shader_ = shader_.size()-1;
  return s->program();
}

} // namespace Volume
