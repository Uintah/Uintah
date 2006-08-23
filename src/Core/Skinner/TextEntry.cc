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
//    File   : TextEntry.cc
//    Author : McKay Davis
//    Date   : Mon Jul  3 01:01:16 2006


#include <Core/Skinner/Variables.h>
#include <Core/Skinner/TextEntry.h>
#include <Core/Events/EventManager.h>
#include <Core/Util/Assert.h>
#include <Core/Events/keysyms.h>
#include <Core/Util/FileUtils.h>
#include <Core/Math/MinMax.h>
#include <Core/Geom/TextRenderer.h>
#include <Core/Containers/StringUtil.h>


namespace SCIRun {
  namespace Skinner {
    TextEntry::TextEntry(Variables *variables) :
      Text(variables),
      str_(),
      inside_(false),
      numeric_(variables->get_bool("numeric"))
    { 
      flags_ &= ~TextRenderer::CURSOR;      
    }

    TextEntry::~TextEntry()
    {
    }

    MinMax
    TextEntry::get_minmax(unsigned int ltype)
    {
      return SPRING_MINMAX;
    }

    int
    TextEntry::get_signal_id(const string &id) const {
      if (id == "TextEntry::return") return 1;
      return 0;
    }

    void
    TextEntry::autocomplete() {

      pair<string, string> dirfile = split_filename(str_);
      if (!validDir(dirfile.first)) return;

      vector<string> files = 
        GetFilenamesStartingWith(dirfile.first, dirfile.second);

      if (files.empty()) {
        return;
      } if (files.size() == 1) {
        str_ = dirfile.first + files[0];
        if (validDir(str_)) {
          str_ = str_ + "/";
        }
      } else {
        unsigned int j0 = dirfile.second.size();
        unsigned int j = j0;
        do {
          for (unsigned int i = 1; i < files.size(); ++i) {
            if ((j == files[i].size()) || (files[i][j] != files[i-1][j])) {
              str_ = str_ + files[i].substr(j0, j-j0);
              cursor_position_ = str_.length();
              return;
            }
          }
          ++j;
        } while (1);
      }
      cursor_position_ = str_.length();
    }


    BaseTool::propagation_state_e
    TextEntry::process_event(event_handle_t event)
    {
      const RectRegion &region = get_region();
      PointerEvent *pointer = dynamic_cast<PointerEvent *>(event.get_rep());
      if (pointer) {
        if (region.inside(pointer->get_x(), pointer->get_y()) &&
            get_vars()->get_bool("cursor")) {
          flags_ |= TextRenderer::CURSOR;
        } else {
          flags_ &= ~TextRenderer::CURSOR;
        }
      }

      KeyEvent *key = dynamic_cast<KeyEvent *>(event.get_rep());
      if (key && (key->get_key_state() & KeyEvent::KEY_PRESS_E)) {
        int code = key->get_keyval();
        bool shift = (key->get_modifiers() & EventModifiers::SHIFT_E);
        string character = "";
        if (code == SCIRun_Return) {
          throw_signal("TextEntry::enter");
        } else  if ((code >= SCIRun_a) && (code <= SCIRun_z)) {
          code -= SCIRun_a; 
          
          if (shift) {
            code += char_traits<wchar_t>::to_int_type('A');
          } else {
            code += char_traits<wchar_t>::to_int_type('a');
          }
          character = string(1, char_traits<wchar_t>::to_char_type(code));
        } else if ((code >= SCIRun_0) && (code <= SCIRun_9)) {
          code -= SCIRun_0; 

          if (key->get_modifiers() & EventModifiers::SHIFT_E) {
            // insert special chars here
            code += char_traits<wchar_t>::to_int_type('0');
          } else {
            code += char_traits<wchar_t>::to_int_type('0');
          }
          character = string(1, char_traits<wchar_t>::to_char_type(code));
        } else if (code == SCIRun_slash) {
          character = string("/");
        } else if (code == SCIRun_period) {
          character = string(".");
        } else if (code == SCIRun_minus && !shift) {
          character = string("-");
        } else if (code == SCIRun_minus && shift) {
          character = string("_");
        } else if (code == SCIRun_space) {
          character = string(" ");
        } else if (code == SCIRun_Tab) {
          autocomplete();
        } else if (code == SCIRun_BackSpace) {
          if (cursor_position_) {
            cursor_position_--;
            str_.erase(cursor_position_, 1);
          }
        } else if (code == SCIRun_Left) {
          cursor_position_ = Max(int(cursor_position_-1), 0);
        } else if (code == SCIRun_Right) {
          cursor_position_ = Min(int(cursor_position_+1), int(str_.length()));
        } else {
          //          cerr << get_id() << " cannot handle keycode: " << code << std::endl;
        }
        
        if (character.length()) {
          string temp = str_;
          temp.insert(cursor_position_, character);
          if (numeric_) {
            string temp = str_;
            temp.insert(cursor_position_, character);
            double val;
            if (string_to_double(temp, val)) {
              str_ = temp;
              cursor_position_++;
            }
          } else {            
            str_.insert(cursor_position_, character);
            cursor_position_++;
          }
        }
        renderer_->set_cursor_position(cursor_position_);                
        EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
        
        get_vars()->change_parent(get_vars()->get_string("variable"), str_, "string", true);
        get_vars()->insert("text", str_, "string", false);
      } else {
        get_vars()->maybe_get_string(get_vars()->get_string("variable"), str_);
        get_vars()->insert("text", str_, "string", false);
      }
        

      return Text::process_event(event);
    }


    Drawable *
    TextEntry::maker(Variables *variables)
    {
      return new TextEntry(variables);
    }
  }
}
