//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
//    File   : Colormap1D.h
//    Author : McKay Davis
//    Date   : Thu Jul  6 13:47:57 2006


#ifndef SKINNER_COLORMAP1D_H
#define SKINNER_COLORMAP1D_H

#include <Core/Skinner/Parent.h>
#include <Core/Skinner/Color.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>

namespace SCIRun {
  namespace Skinner {
    class Colormap1D : public Parent {
    public:
      Colormap1D(Variables *);
      virtual ~Colormap1D();
      virtual MinMax                    minmax(unsigned int);
      static string                     class_name() { return "Colormap1D"; }
      static DrawableMakerFunc_t        maker;
      virtual int                       get_signal_id(const string &) const;

    private:
      CatcherFunction_t                 redraw;
      NrrdDataHandle                    nrrd_handle_;
      ColorMapHandle                    colormap_;
      ColorMappedNrrdTextureObjHandle   tex_;
    };
  }
}

#endif
