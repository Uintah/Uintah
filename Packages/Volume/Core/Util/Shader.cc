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
//    File   : Shader.cc
//    Author : Milan Ikits
//    Date   : Sun Jun 27 17:40:35 2004

#include <iostream>
#include <fstream>
#include <sci_gl.h>
#include <Packages/Volume/Core/Util/Shader.h>

using namespace std;

namespace Volume {

ShaderProgramARB::ShaderProgramARB (const string& name, const string& program)
  : Shader(name), mProgram(program.length(), 0)
{
  if (program != "")
  {
    for (unsigned int i=0; i<program.length(); i++)
      mProgram[i] = program[i];
    mDirty = true;
  }
}

void
ShaderProgramARB::setFileName (const string& name)
{
  mFilename = name;
}

void
ShaderProgramARB::load (const string& name)
{
  mFilename = name;
  reload();
}

void
ShaderProgramARB::create ()
{
  glGenProgramsARB(1, &mId);
  update();
}

void
ShaderProgramARB::destroy ()
{
  glDeleteProgramsARB(1, &mId);
}

void
ShaderProgramARB::update ()
{
  if (mDirty)
  {
    //cerr << mProgram << endl;
    glBindProgramARB(mType, mId);
    glProgramStringARB(mType, GL_PROGRAM_FORMAT_ASCII_ARB,
                       mProgram.length(), mProgram.c_str());
    if (glGetError() != GL_NO_ERROR)
    {
      int position;
      glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &position);
      int start = position;
      for (; start > 0 && mProgram[start] != '\n'; start--);
      if (mProgram[start] == '\n') start++;
      uint end = position;
      for (; end < mProgram.length()-1 && mProgram[end] != '\n'; end++);
      if (mProgram[end] == '\n') end--;
      int ss = start;
      int l = 1;
      for (; ss >= 0; ss--) { if (mProgram[ss] == '\n') l++; }
      string line((char*)(mProgram.c_str()+start), end-start+1);
      string underline = line;
      for (uint i=0; i<end-start+1; i++) underline[i] = '-';
      underline[position-start] = '#';
      glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, 0);
      mDirty = false;
      cerr << "[ShaderProgramARB(\"" << mName << "\")::create] "
           << "Program error at line " << l << ", character "
           << position-start << endl << endl << line << endl
           << underline << endl;
    }
    glBindProgramARB(mType, 0);
    mDirty = false;
  }
}

void
ShaderProgramARB::reload ()
{
  ifstream f(mFilename.c_str(), ios::in | ios::binary);
  if (f.fail())
  {
    cerr << "[ShaderProgramARB(\"" << mName << "\")::reload] "
         << "Failed to open file: " << mFilename << endl;
    return;
  }
  // get length of file
  f.seekg(0, ios::end);
  int length = f.tellg();
  f.seekg(0, ios::beg);
  char* buffer = new char[length+1];
  f.read(buffer, length);
  buffer[length] = 0;
  f.close();
  mProgram = string(buffer);
  delete [] buffer;
  mDirty = true;
}

void
ShaderProgramARB::bind ()
{
  glEnable(mType);
  glBindProgramARB(mType, mId);
}

void
ShaderProgramARB::release ()
{
  glBindProgramARB(mType, 0);
  glDisable(mType);
}

void
ShaderProgramARB::makeCurrent ()
{
  glBindProgramARB(mType, mId);
}

void
ShaderProgramARB::setLocalParam (int i, float x, float y, float z, float w)
{
  glProgramLocalParameter4fARB(mType, i, x, y, z, w);
}

//-------------------------------------------------------------------------------

VertexProgramARB::VertexProgramARB (const string& name,
				    const string& program)
  : ShaderProgramARB(name, program)
{
  mType = GL_VERTEX_PROGRAM_ARB;
}

FragmentProgramARB::FragmentProgramARB (const string& name,
					const string& program)
  : ShaderProgramARB(name, program)
{
  mType = GL_FRAGMENT_PROGRAM_ARB;
}

} // end namespace Volume
