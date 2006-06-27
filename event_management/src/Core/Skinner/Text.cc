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
//  
//    File   : Text.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:00:29 2006
#include <Core/Skinner/Text.h>
#include <Core/Skinner/Variables.h>
#include <Core/Geom/FontManager.h>
#include <Core/Containers/StringUtil.h>

namespace SCIRun {
  namespace Skinner {
    Text::Text(Variables *variables,
               const string &font,
               double size,
               const Color &fgcolor,
               const Color &bgcolor,
               unsigned int anchor,
               bool vertical,
               bool shadow,
               bool extruded,
               bool reverse,
               int offsetx,
               int offsety) :
      Drawable(variables),
      fgcolor_(fgcolor),
      bgcolor_(bgcolor),
      flags_(anchor),
      renderer_(FontManager::get_renderer(size, font)),
      offsetx_(offsetx),
      offsety_(offsety)
    {
      flags_ |= vertical ? TextRenderer::VERTICAL : 0;
      flags_ |= shadow   ? TextRenderer::SHADOW   : 0;
      flags_ |= extruded ? TextRenderer::EXTRUDED : 0;
      flags_ |= reverse  ? TextRenderer::REVERSE  : 0;
    }

    Text::~Text() {}
  
    BaseTool::propagation_state_e
    Text::process_event(event_handle_t event)
    {
      WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
      if (window && window->get_window_state() == WindowEvent::REDRAW_E) {
        string text = "";
        get_vars()->maybe_get_string("text", text);
          
        renderer_->set_shadow_offset(offsetx_, offsety_);
        renderer_->set_color(fgcolor_.r, fgcolor_.g, fgcolor_.b, fgcolor_.a);
        renderer_->set_shadow_color(bgcolor_.r, bgcolor_.g, bgcolor_.b, bgcolor_.a);

        float mx = (region_.x2() + region_.x1())/2.0;
        float my = (region_.y2() + region_.y1())/2.0;

        float x = mx;
        float y = my;
        
        switch (flags_ & TextRenderer::ANCHOR_MASK) {
        case TextRenderer::N:  x = mx; y = region_.y2(); break;
        case TextRenderer::S:  x = mx; y = region_.y1(); break;

        case TextRenderer::E:  x = region_.x2(); y = my; break;
        case TextRenderer::W:  x = region_.x1(); y = my; break;

        case TextRenderer::NE: x = region_.x2(); y = region_.y2(); break;
        case TextRenderer::SE: x = region_.x2(); y = region_.y1(); break;
        case TextRenderer::SW: x = region_.x1(); y = region_.y1(); break;
        case TextRenderer::NW: x = region_.x1(); y = region_.y2(); break;

        case TextRenderer::C:  x = mx; y = my; break;
        }
        renderer_->render(text, x, y, flags_);
      }
      return CONTINUE_E;
    }

    Drawable *
    Text::maker(Variables *vars, const Drawables_t &children, void *data)
    {

      Color fgcolor(1.0, 1.0, 1.0, 1.0);
      vars->maybe_get_color("color", fgcolor);
      vars->maybe_get_color("fgcolor", fgcolor);

      Color bgcolor(0.0, 0.0, 0.0, 1.0); 
      vars->maybe_get_color("bgcolor", bgcolor);

      double size = 20.0;
      vars->maybe_get_double("size", size);
      
      string font = "scirun.ttf";
      vars->maybe_get_string("font", font);
      
      int offsetx = 1;
      vars->maybe_get_int("offset", offsetx);
      vars->maybe_get_int("offsetx", offsetx);

      int offsety = -1;
      vars->maybe_get_int("offset", offsety);
      vars->maybe_get_int("offsety", offsety);
      
      unsigned int anchor = TextRenderer::SW;
      string anchorstr = "SW";
      vars->maybe_get_string("anchor", anchorstr);
      anchorstr = string_toupper(anchorstr);
      if      (anchorstr ==  "N") { anchor = TextRenderer::N;  }
      else if (anchorstr ==  "E") { anchor = TextRenderer::E;  }
      else if (anchorstr ==  "S") { anchor = TextRenderer::S;  }
      else if (anchorstr ==  "W") { anchor = TextRenderer::W;  }
      else if (anchorstr == "NE") { anchor = TextRenderer::NE; }
      else if (anchorstr == "SE") { anchor = TextRenderer::SE; }
      else if (anchorstr == "SW") { anchor = TextRenderer::SW; }
      else if (anchorstr == "NW") { anchor = TextRenderer::NW; }
      else if (anchorstr ==  "C") { anchor = TextRenderer::C;  }
      else { cerr << vars->get_id() << " anchor invalid: " 
                  << anchorstr << "\n"; }
      

      return new Text(vars, 
                      font,
                      size,
                      fgcolor, 
                      bgcolor,
                      anchor,
                      vars->get_bool("vertical"),
                      vars->get_bool("shadow"),
                      vars->get_bool("extruded"),
                      vars->get_bool("reverse"),
                      offsetx,
                      offsety);

    }

  }
}
