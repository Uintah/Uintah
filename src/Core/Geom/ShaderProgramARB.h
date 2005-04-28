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
//    File   : ShaderProgramARB.h
//    Author : Milan Ikits
//    Date   : Wed Jul  7 23:20:59 2004

#ifndef ShaderProgramARB_h 
#define ShaderProgramARB_h

#include <string>

namespace SCIRun {

class ShaderProgramARB
{
public:
  ShaderProgramARB(const std::string& program);
  virtual ~ShaderProgramARB();
  
  bool create();
  bool valid();
  void destroy();

  void bind();
  void release();
  void enable();
  void disable();
  void makeCurrent();

  void setLocalParam(int, float, float, float, float);

  // Call init_shaders_supported before shaders_supported queries!
  static void init_shaders_supported();
  static bool shaders_supported();
  static int max_texture_size_1();
  static int max_texture_size_4();
  static bool texture_non_power_of_two();

protected:
  unsigned int mType;
  unsigned int mId;
  std::string mProgram;

  static bool mInit;
  static bool mSupported;
  static bool mNon2Textures;
  static int max_texture_size_1_;
  static int max_texture_size_4_;
};

class VertexProgramARB : public ShaderProgramARB
{
public:
  VertexProgramARB(const std::string& program);
  ~VertexProgramARB();
};

class FragmentProgramARB : public ShaderProgramARB
{
public:
  FragmentProgramARB(const std::string& program);
  ~FragmentProgramARB();
};

} // end namespace SCIRun

#endif // ShaderProgramARB_h
