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
//    File   : Shader.h
//    Author : Milan Ikits
//    Date   : Sun Jun 27 17:39:44 2004

#ifndef Shader_h
#define Shader_h

#include <string>

namespace Volume {

class Shader
{
public:
  Shader (const std::string& name)
    : mName(name), mId(0), mDirty(false) {}
  virtual ~Shader () {}

  virtual void create () = 0;
  virtual void update () = 0;
  virtual void destroy () = 0;

  virtual void bind () = 0;
  virtual void release () = 0;
  virtual void makeCurrent () = 0;
  
  inline const std::string& getName () const { return mName; }
  inline uint getId () { return mId; }
  inline bool isDirty () { return mDirty; }
  
protected:
  std::string mName;
  uint mId;
  bool mDirty;
};

//-------------------------------------------------------------------------------
// ARB -- R3xx/NV3x and up
//-------------------------------------------------------------------------------

class ShaderProgramARB : public Shader
{
public:
  ShaderProgramARB (const std::string& name, const std::string& program = "");
  ~ShaderProgramARB () {}

  void setFileName (const std::string& name);
  inline const std::string& getFilename () const { return mFilename; }

  void load (const std::string& filename); // same as setFileName + reload
  void reload ();

  void create ();
  void update ();
  void destroy ();

  void bind ();
  void release ();
  void makeCurrent ();
  
  void setLocalParam (int i, float x, float y, float z, float w);

protected:
  uint mType;
  std::string mFilename;
  std::string mProgram;
};

class VertexProgramARB : public ShaderProgramARB
{
public:
  VertexProgramARB (const std::string& name, const std::string& program = "");
  ~VertexProgramARB () {}
};

class FragmentProgramARB : public ShaderProgramARB
{
public:
  FragmentProgramARB (const std::string& name, const std::string& program = "");
  ~FragmentProgramARB () {}
};

} // end namespace Volume

#endif // Shader_h
