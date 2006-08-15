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
    Text::Text(Variables *vars) :
      Drawable(vars),
      fgcolor_(1., 1., 1., 1.),
      bgcolor_(0., 0., 0., 1.),
      flags_(0),
      renderer_(0),
      offsetx_(0),
      offsety_(0),
      cursor_position_(0)
    {
      vars->maybe_get_color("color", fgcolor_);
      vars->maybe_get_color("fgcolor", fgcolor_);
      vars->maybe_get_color("bgcolor", bgcolor_);

      double size = 20.0;
      vars->maybe_get_double("size", size);
      
      string font = "scirun.ttf";
      vars->maybe_get_string("font", font);

      renderer_ = FontManager::get_renderer(size, font);
      
      vars->maybe_get_int("offset", offsetx_);
      vars->maybe_get_int("offsetx", offsetx_);

      vars->maybe_get_int("offset", offsety_);
      vars->maybe_get_int("offsety", offsety_);
      
      
      string anchorstr = "SW";
      vars->maybe_get_string("anchor", anchorstr);
      anchorstr = string_toupper(anchorstr);

      flags_ = TextRenderer::SW;
      if      (anchorstr ==  "N") { flags_ = TextRenderer::N;  }
      else if (anchorstr ==  "E") { flags_ = TextRenderer::E;  }
      else if (anchorstr ==  "S") { flags_ = TextRenderer::S;  }
      else if (anchorstr ==  "W") { flags_ = TextRenderer::W;  }
      else if (anchorstr == "NE") { flags_ = TextRenderer::NE; }
      else if (anchorstr == "SE") { flags_ = TextRenderer::SE; }
      else if (anchorstr == "SW") { flags_ = TextRenderer::SW; }
      else if (anchorstr == "NW") { flags_ = TextRenderer::NW; }
      else if (anchorstr ==  "C") { flags_ = TextRenderer::C;  }
      else { cerr << vars->get_id() << " anchor invalid: " 
                  << anchorstr << "\n"; }
      
      flags_ |= vars->get_bool("vertical") ? TextRenderer::VERTICAL : 0;
      flags_ |= vars->get_bool("shadow")   ? TextRenderer::SHADOW   : 0;
      flags_ |= vars->get_bool("extruded") ? TextRenderer::EXTRUDED : 0;
      flags_ |= vars->get_bool("reverse")  ? TextRenderer::REVERSE  : 0;
      flags_ |= vars->get_bool("cursor") ? TextRenderer::CURSOR : 0;
      
      //      REGISTER_CATCHER_TARGET(Text::redraw);
    }

    Text::~Text() {}
  
    BaseTool::propagation_state_e
    Text::process_event(event_handle_t event)
    {
      WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
      if (window && window->get_window_state() == WindowEvent::REDRAW_E) {
        redraw(0);
      }
      return CONTINUE_E;
    }

    BaseTool::propagation_state_e
    Text::redraw(event_handle_t)
    {
      const RectRegion &region = get_region();
      string text = "";
      get_vars()->maybe_get_string("text", text);
      
      if (region.height() < 1 || renderer_->height(text) > region.height()) {
        return CONTINUE_E;
      }
      
      renderer_->set_shadow_offset(offsetx_, offsety_);
      renderer_->set_color(fgcolor_.r, fgcolor_.g, fgcolor_.b, fgcolor_.a);
      renderer_->set_shadow_color(bgcolor_.r, bgcolor_.g, bgcolor_.b, bgcolor_.a);
      
      float mx = (region.x2() + region.x1())/2.0;
      float my = (region.y2() + region.y1())/2.0;
      
      float x = mx;
      float y = my;
      
      switch (flags_ & TextRenderer::ANCHOR_MASK) {
      case TextRenderer::N:  x = mx; y = region.y2(); break;
      case TextRenderer::S:  x = mx; y = region.y1(); break;
        
      case TextRenderer::E:  x = region.x2(); y = my; break;
      case TextRenderer::W:  x = region.x1(); y = my; break;
        
      case TextRenderer::NE: x = region.x2(); y = region.y2(); break;
      case TextRenderer::SE: x = region.x2(); y = region.y1(); break;
      case TextRenderer::SW: x = region.x1(); y = region.y1(); break;
      case TextRenderer::NW: x = region.x1(); y = region.y2(); break;
        
      case TextRenderer::C:  x = mx; y = my; break;
      }
      renderer_->render(text, x, y, flags_);
      
      return CONTINUE_E;
    }


  }
}
