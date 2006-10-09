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
//    File   : Texture.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:02:57 2006

#include <Core/Skinner/Texture.h>
#include <Core/Skinner/Variables.h>
#include <Core/Geom/TextureObj.h>
#include <Core/Util/Environment.h>
#include <Core/Util/FileUtils.h>
#include <Core/Math/MiscMath.h>
#include <sci_gl.h>
#include <sci_glu.h>

namespace SCIRun {
  namespace Skinner {
    Drawable *
    Texture::maker(Variables *vars)
    {
      return new Texture(vars);
    }

    Texture::Texture(Variables *vars) :
      Drawable(vars),
      tex_(0),
      filename_(Var<string>(vars,"file","")()),
      blendfunc_(make_pair(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)),
      color_(vars,"color",Color(1., 1., 1., 1.)),
      anchor_(Var<int>(vars,"anchor",0)()),
      flipx_(Var<bool>(vars,"flipx",false)()),
      flipy_(Var<bool>(vars,"flipy",false)()),
      repeatx_(Var<bool>(vars,"repeatx",false)()),
      repeaty_(Var<bool>(vars,"repeaty",false)()),
      degrees_(Var<double>(vars,"rotate",0.0)())
    {
      NrrdDataHandle nin = new NrrdData();
      string fullfile = filename_;

      // Search SKINNER_PATH for filename
      const char *skinner_path = sci_getenv("SKINNER_PATH");
      if (!validFile(fullfile) && skinner_path) {
        fullfile = findFileInPath(filename_, skinner_path)+filename_;
      if (!validFile(fullfile)) {
        filename_="Help.png";
        fullfile = findFileInPath(filename_, skinner_path)+filename_;
      }

      }

      if (!validFile(fullfile)) {
        fullfile = skinner_path+string("Help.png");
        //        throw "Texture Invalid filename: "+fullfile;
      }

      if (nrrdLoad(nin->nrrd_, fullfile.c_str(), 0)) {
        char *err = biffGetDone(NRRD);
        string str(err);
        free(err);
        throw "Texture error loading: "+fullfile+"\n"+str;
      }
      tex_ = new TextureObj(nin);

      REGISTER_CATCHER_TARGET(Texture::redraw);
    }

    Texture::~Texture() 
    {
    }

    BaseTool::propagation_state_e
    Texture::redraw(event_handle_t) {
      if (!tex_) return STOP_E;
      
      Color color = color_();
      //      color.random();
      tex_->set_color(color.r, color.g, color.b, color.a);

      glMatrixMode(GL_TEXTURE);
      glPushMatrix();
      glLoadIdentity();
      CHECK_OPENGL_ERROR();

      if (flipx_) {
        glScaled(-1.0, 1.0, 1.0);
        glTranslated(-1.0, 0.0, 0.0);
      }

      if (!flipy_) {
        glScaled(1.0, -1.0, 1.0);
        glTranslated(0.0, -1.0, 0.0);
      }


      //      glScaled(flipx_ ? -1.0: 1.0, flipy_ ? 1.0 : -1.0, 1.0);
      glTranslated(0.5, 0.5, 0.0);
      glRotated(degrees_, 0,0,1);
      glTranslated(-0.5, -0.5, 0.0);

      const RectRegion &region = get_region();

        
      float tw = repeatx_ ? (region.width()/float(tex_->width())) : 1.0;
      float th = repeaty_ ? (region.height()/float(tex_->height())) : 1.0;

      float tex_coords[8] = {0, Floor(th)-th, 
                             tw, Floor(th)-th,
                             tw, Floor(th),
                             0.0, Floor(th) };
        
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                      //                        GL_CLAMP_TO_EDGE);
                      //repeatx_ ? GL_REPEAT : GL_CLAMP_TO_EDGE);
                      GL_REPEAT);
      
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                      //                        GL_CLAMP_TO_EDGE);
                      //                        repeaty_ ? GL_REPEAT : GL_CLAMP_TO_EDGE);
                      GL_REPEAT);
      //      }
      




      CHECK_OPENGL_ERROR();


      Point coords[4];
      coords[0] = Point(region.x1(), region.y1(), 0.0);
      coords[1] = Point(region.x2(), region.y1(), 0.0);
      coords[2] = Point(region.x2(), region.y2(), 0.0);
      coords[3] = Point(region.x1(), region.y2(), 0.0);



      tex_->draw(4, coords, tex_coords);
      glMatrixMode(GL_TEXTURE);
      glPopMatrix();
      return CONTINUE_E;
    }
  }
}
