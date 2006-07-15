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
//    File   : Fbuffer.cc
//    Author : Allen R. Sanderson
//    Date   : Jan 17 2006


#include "Fbuffer.h"
#include <iostream>

#include <sci_glu.h>
#include <sci_glx.h>

#include <string>

using namespace SCIRun;

static bool mNV_float_buffer = true;
static bool mNV_texture_rectangle = false;

Fbuffer::Fbuffer( int width, int height ) :
  mWidth( width ),
  mHeight( height )
{}

bool
Fbuffer::create ()
{
  // Create the objects
  glGenFramebuffersEXT( 1, &mFB );

  return check_buffer();
}

bool
Fbuffer::check_buffer ()
{
  //----------------------
  // Framebuffer Objects initializations
  //----------------------
  GLuint status = glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT );

  switch( status ) {
  case GL_FRAMEBUFFER_COMPLETE_EXT:
    std::cerr << " GL_FRAMEBUFFER_COMPLETE_EXT \n";
    return true;

  case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT \n";
    return false;

  case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT \n";
    return false;

  case GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT \n";
    return false;

  case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT \n";
    return false;

  case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT \n";
    return false;

  case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT \n";
    return false;

  case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT \n";
    return false;

  case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
    std::cerr << " GL_FRAMEBUFFER_UNSUPPORTED_EXT \n";
    return false;

  default:
    std::cerr << " GL_FRAMEBUFFER_UNKNOWN " << status << std::endl;
    return false;
  }
}

void
Fbuffer::enable ()
{
  glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, mFB );
}

void
Fbuffer::disable ()
{
  glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
}

void
Fbuffer::attach_texture(GLenum attach, GLenum tex_type, GLuint tex_id,
                        int mip_level, int z_slice)
{
  if (tex_type == GL_TEXTURE_1D) {
    glFramebufferTexture1DEXT( GL_FRAMEBUFFER_EXT, attach,
                               GL_TEXTURE_1D, tex_id, mip_level );
  }
  else if (tex_type == GL_TEXTURE_3D) {
    glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, attach,
                               GL_TEXTURE_3D, tex_id, mip_level, z_slice );
  }
  else {
    // Default is GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_ARB, or cube faces
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, attach,
                               tex_type, tex_id, mip_level );
  }
}


void
Fbuffer::attach_buffer(GLenum attach, GLuint id)
{
  glFramebufferRenderbufferEXT( GL_FRAMEBUFFER_EXT, attach, 
                                GL_RENDERBUFFER_EXT, id);
}

void 
Fbuffer::unattach(GLenum attach)
{
  GLenum type = get_attached_type(attach);
  
  switch(type) {
  case GL_NONE:
    break;
  case GL_RENDERBUFFER_EXT:
    attach_buffer( attach, 0 );
    break;
  case GL_TEXTURE:
    attach_texture( attach, GL_TEXTURE_2D, 0 );
    break;
  default:
    std::cerr << "FramebufferObject::unbind_attachment  ";
    std::cerr << "ERROR: Unknown attached resource type\n";
  }

}

GLenum
Fbuffer::get_attached_type( GLenum attachment )
{
  // Returns GL_RENDERBUFFER_EXT or GL_TEXTURE
//   _GuardedBind();
  GLint type = 0;
  glGetFramebufferAttachmentParameterivEXT(GL_FRAMEBUFFER_EXT, attachment,
                                           GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE_EXT,
                                           &type);
//   _GuardedUnbind();
  return GLenum(type);
}
