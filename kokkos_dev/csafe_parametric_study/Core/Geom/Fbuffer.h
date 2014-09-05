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
//    File   : Fbuffer.h
//    Author : Allen R. Sanderson
//    Date   : Jan 17 2006
//    Update : Feb 13 2006,  Kurt Zimmerman

#ifndef Fbuffer_h
#define Fbuffer_h


#include <sci_gl.h>
#include <iostream>

namespace SCIRun{


class Fbuffer
{
public:
  Fbuffer( int width, int height );

  bool create();
  bool check_buffer();

  void enable();
  void disable();

  void attach_texture(GLenum attach, GLenum tex_type, GLuint tex_id,
                      int mip_level = 0, int z_slice = 0);
  void attach_buffer(GLenum attach, GLuint id);
  
  void unattach(GLenum attach);

  bool is_valid(std::ostream& ostr = std::cerr ) {
                return true;
        }

protected:
  int mWidth, mHeight;

  GLuint mFB;        // associated Render buffer
  GLuint mRB;        // associated Render buffer

  GLenum get_attached_type( GLenum attachment );

};

} // end namespace SCIRun
#endif // Fbuffer_h

