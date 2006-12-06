#ifndef UINTAH_HOMEBREW_SimpleString_H
#define UINTAH_HOMEBREW_SimpleString_H

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

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
      
       Copyright (C) 2000 SCI Group
      
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
