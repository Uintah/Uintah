//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : Texture.h
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:04:36 2006

#ifndef SKINNER_TEXTURE_H
#define SKINNER_TEXTURE_H

#include <Core/Skinner/Drawable.h>
#include <Core/Skinner/Color.h>
#include <sci_gl.h>

namespace SCIRun {
  class TextureObj;
  namespace Skinner {
    typedef pair<GLenum, GLenum> blendfunc_t;
    class Texture : public Drawable {
    public:
      Texture (Variables *);
      virtual ~Texture();
      static string                     class_name() { return "Texture"; }
      static DrawableMakerFunc_t        maker;
    private:
      CatcherFunction_t                 redraw;
      TextureObj *                      tex_;
      string                            filename_;
      blendfunc_t                       blendfunc_;
      Var<Color>                        color_;
      unsigned int                      anchor_;
      bool                              flipx_;
      bool                              flipy_;
      bool                              repeatx_;
      bool                              repeaty_;
      double                            degrees_;

    };
  }
}

#endif
