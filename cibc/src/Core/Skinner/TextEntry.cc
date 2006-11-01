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
#include <Core/Math/MiscMath.h>
#include <Core/Geom/TextRenderer.h>
#include <Core/Containers/StringUtil.h>


namespace SCIRun {
  namespace Skinner {
    TextEntry::TextEntry(Variables *variables) :
      Text(variables),
      numeric_(variables, "numeric", false)
    { 
      REGISTER_CATCHER_TARGET_BY_NAME(TextEntry::redraw, Text::redraw);
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


    BaseTool::propagation_state_e
    TextEntry::process_event(event_handle_t event)
    {
      KeyEvent *key = dynamic_cast<KeyEvent *>(event.get_rep());
      if (key && (key->get_key_state() & KeyEvent::KEY_PRESS_E)) {
        string str = get_vars()->get_string("text");
        int code = key->get_keyval();
        bool shift = (key->get_modifiers() & EventModifiers::SHIFT_E);
        string character = "";
        cursor_position_ = Clamp(cursor_position_,0, str.length());
        if (code == SCIRun_Return) {
          throw_signal("TextEntry::enter");
        } else if (code == SCIRun_Tab) {
          str = autocomplete(str);
          cursor_position_ = str.length();
        } else if (code == SCIRun_BackSpace) {
          if (cursor_position_) {
            cursor_position_--;
            str.erase(cursor_position_, 1);
          }
        } else if (code == SCIRun_Left) {
          cursor_position_ = Max(int(cursor_position_-1), 0);
        } else if (code == SCIRun_Right) {
          cursor_position_ = Min(int(cursor_position_+1),int(str.length()));
        } else {
          character = string(1, char_traits<wchar_t>::to_char_type(code));
        }
        
        if (character.length()) {
          if (numeric_) {
            double val;
            string temp = str;
            temp.insert(cursor_position_, character);
            if (string_to_double(temp, val)) {
              str = temp;
              cursor_position_++;
            }
          } else {            
            str.insert(cursor_position_, character);
            cursor_position_++;
          }
        }

        renderer_->set_cursor_position(cursor_position_);
        get_vars()->set_by_string("text", str);
        EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
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
