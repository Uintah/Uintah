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
//    File   : VolShader.h
//    Author : Milan Ikits
//    Date   : Tue Jul 13 02:27:58 2004

#ifndef VolShader_h
#define VolShader_h

#include <string>
#include <vector>

namespace SCIRun {

class FragmentProgramARB;

class VolShader
{
public:
  VolShader(int dim, int vsize, bool shading, bool frag, bool fog, int blend);
  ~VolShader();

  bool create();
  
  inline int dim() { return dim_; }
  inline int vsize() { return vsize_; }
  inline bool shading() { return shading_; }
  inline bool fog() { return fog_; }
  inline int blend() { return blend_; }
  inline bool frag() { return frag_; }

  inline bool match(int dim, int vsize, bool shading, bool frag, bool fog, int blend)
  { return dim_ == dim && vsize_ == vsize && shading_ == shading && frag_ == frag
      && fog_ == fog && blend_ == blend; }

  inline FragmentProgramARB* program() { return program_; }
  
protected:
  bool emit(std::string& s);

  int dim_;
  int vsize_;
  bool shading_;
  bool fog_;
  int blend_;
  bool frag_;
  FragmentProgramARB* program_;
};

class VolShaderFactory
{
public:
  VolShaderFactory();
  ~VolShaderFactory();
  
  FragmentProgramARB* shader(int dim, int vsize, bool shading, bool frag,
                             bool fog, int blend);

protected:
  std::vector<VolShader*> shader_;
  int prev_shader_;
};

} // end namespace SCIRun

#endif // VolShader_h
