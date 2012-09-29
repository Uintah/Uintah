/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_SimpleString_H
#define UINTAH_HOMEBREW_SimpleString_H

#include <string>

namespace Uintah {
   /**************************************
     
     CLASS
       SimpleString
      
       Short Description...
      
     GENERAL INFORMATION
      
       SimpleString.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
     KEYWORDS
       SimpleString
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class SimpleString {
   public:
      SimpleString() {
         str=0;
         freeit=0;
      }
      ~SimpleString() {
         if(freeit)
            free((void*)str);
      }
      SimpleString(const char* str)
         : str(str), freeit(false) {
      }
      SimpleString(const std::string& s)
         : str(strdup(s.c_str())), freeit(true) {
      }
      SimpleString(const SimpleString& copy)
         : str(copy.str), freeit(copy.freeit) {
            if(freeit)
               str=strdup(str);
      }
      SimpleString& operator=(const SimpleString& copy) {
         if(freeit && str)
            free((void*)str);
         freeit=copy.freeit;
         str=copy.str;
         if(freeit)
            str=strdup(str);
         return *this;
      }
      operator const char*() const {
         return str;
      }
   private:
      const char* str;
      bool freeit;
   };

} // End namespace Uintah

#endif
