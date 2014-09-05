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
//    File   : CM2Shader.h
//    Author : Milan Ikits
//    Date   : Tue Jul 13 02:27:30 2004

#ifndef CM2Shader_h
#define CM2Shader_h

#include <string>
#include <vector>

namespace Volume {

class FragmentProgramARB;

enum CM2ShaderType
{
  CM2_SHADER_TRIANGLE = 0,
  CM2_SHADER_RECTANGLE_1D = 1,
  CM2_SHADER_RECTANGLE_ELLIPSOID = 2
};

enum CM2BlendType
{
  CM2_BLEND_RASTER = 0,
  CM2_BLEND_FRAGMENT_ATI = 1,
  CM2_BLEND_FRAGMENT_NV = 2
};

class CM2Shader
{
public:
  CM2Shader(CM2ShaderType type, bool faux, CM2BlendType blend);
  ~CM2Shader();

  bool create();
  
  inline bool match(CM2ShaderType type, bool faux, CM2BlendType blend)
  { return type_ == type && faux_ == faux && blend_ == blend; }

  inline FragmentProgramARB* program() { return program_; }
  
protected:
  bool emit(std::string& s);

  CM2ShaderType type_;
  bool faux_;
  CM2BlendType blend_;

  FragmentProgramARB* program_;
};

class CM2ShaderFactory
{
public:
  CM2ShaderFactory();
  ~CM2ShaderFactory();

  void destroy();
  
  FragmentProgramARB* shader(CM2ShaderType type, bool faux, CM2BlendType blend);

protected:
  std::vector<CM2Shader*> shader_;
  int prev_shader_;
};

} // namespace Volume

#endif // CM2Shader_h
