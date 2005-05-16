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
//    File   : Pbuffer.h
//    Author : Milan Ikits
//    Date   : Sun Jun 27 17:49:31 2004

#ifndef SCI_Geom_Pbuffer_h
#define SCI_Geom_Pbuffer_h 1

// TODO:  Maybe get rid of this?  Used in Dataflow/Modules/Render/OpenGL.cc
#if (defined(__linux) && !defined(__ECC)) || defined(__APPLE__) || defined(_WIN32)
#define HAVE_PBUFFER
#endif

#include <sci_gl.h>


namespace SCIRun {

class Pbuffer
{
public:
  Pbuffer(int width, int height, int format = GL_INT /* GL_INT or GL_FLOAT */,
          int numColorBits = 8 /* 8, 16, 32 */,
          bool isRenderTex = true, int isDoubleBuffer = GL_TRUE,
          int numAuxBuffers = GL_DONT_CARE, int numDepthBits = GL_DONT_CARE,
          int numStencilBits = GL_DONT_CARE, int numAccumBits = GL_DONT_CARE);
  ~Pbuffer();

  bool create();
  void destroy();
  void makeCurrent(); // no save state
  bool is_current();

  void swapBuffers(); // flush if single
  void bind(unsigned int buffer = GL_FRONT); // bind as 2D texture
  void release(unsigned int buffer = GL_FRONT); 

  void activate();
  void deactivate();
  
  inline int width() const { return width_; }
  inline int height() const { return height_; }
  inline int visual() const { return visual_id_; }
  inline unsigned int target() const { return tex_target_; }

  bool need_shader();
  void set_use_default_shader(bool b);
  void set_use_texture_matrix(bool b);

  inline int num_color_bits() const { return num_color_bits_; }
  
protected:
  int width_, height_;
  int format_, num_color_bits_; /* , mChannels; */
  bool render_tex_;
  int double_buffer_, num_aux_buffers_;
  int num_depth_bits_, num_stencil_bits_, num_accum_bits_;
  bool separate_; // has separate context
  int visual_id_; // associated visual
  unsigned int tex_; // associated texture id
  unsigned int tex_target_; // associated texture target
  unsigned int tex_format_; // associated texture format
  bool use_default_shader_;
  bool use_texture_matrix_;
  // GL_TEXTURE_2D or GL_TEXTURE_RECTANGLE_NV
  struct PbufferImpl* impl_; // implementation specific details
};

} // end namespace SCIRun

#endif // SCI_Geom_Pbuffer_h
