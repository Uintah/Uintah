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


namespace SCIRun {
  namespace Skinner {
    TextEntry::TextEntry(Variables *variables) :
      Parent(variables),
      str_(),
      inside_(false)
    { 
    }

    TextEntry::~TextEntry()
    {
    }

    MinMax
    TextEntry::get_minmax(unsigned int ltype)
    {
      return SPRING_MINMAX;
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
              return;
            }
          }
          ++j;
        } while (1);
      }
    }


    BaseTool::propagation_state_e
    TextEntry::process_event(event_handle_t event)
    {
      const RectRegion &region = get_region();
      PointerEvent *pointer = dynamic_cast<PointerEvent *>(event.get_rep());
      if (pointer && region.inside(pointer->get_x(), pointer->get_y())) {
        inside_ = true;
      }

      KeyEvent *key = dynamic_cast<KeyEvent *>(event.get_rep());
      if (key && (key->get_key_state() & KeyEvent::KEY_PRESS_E)) {
        int code = key->get_keyval();
        bool shift = (key->get_modifiers() & EventModifiers::SHIFT_E);
        if ((code >= SCIRun_a) && (code <= SCIRun_z)) {
          code -= SCIRun_a; 
          
          if (shift) {
            code += char_traits<wchar_t>::to_int_type('A');
          } else {
            code += char_traits<wchar_t>::to_int_type('a');
          }
          str_ = str_ + string(1, char_traits<wchar_t>::to_char_type(code));
        } else if ((code >= SCIRun_0) && (code <= SCIRun_9)) {
          code -= SCIRun_0; 

          if (key->get_modifiers() & EventModifiers::SHIFT_E) {
            // insert special chars here
            code += char_traits<wchar_t>::to_int_type('0');
          } else {
            code += char_traits<wchar_t>::to_int_type('0');
          }
          str_ = str_ + string(1, char_traits<wchar_t>::to_char_type(code));
        } else if (code == SCIRun_slash) {
          str_ = str_ + string("/");
        } else if (code == SCIRun_period) {
          str_ = str_ + string(".");
        } else if (code == SCIRun_minus && !shift) {
          str_ = str_ + string("-");
        } else if (code == SCIRun_minus && shift) {
          str_ = str_ + string("_");
        } else if (code == SCIRun_space) {
          str_ = str_ + string(" ");
        } else if (code == SCIRun_Tab) {
          autocomplete();
        } else if (code == SCIRun_BackSpace) {
          str_ = str_.substr(0, str_.length()-1);
        } else {
          //          cerr << get_id() << " cannot handle keycode: " << code << std::endl;
        }
                
        EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
        
        get_vars()->change_parent("TextEntry::value", str_, "string", true);
      }
      return Parent::process_event(event);
    }


    Drawable *
    TextEntry::maker(Variables *variables)
    {
      return new TextEntry(variables);
    }
  }
}
