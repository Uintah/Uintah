#ifndef SCICore_OS_Dir_H
#define SCICore_OS_Dir_H

#include <string>

namespace SCICore {
   namespace OS {
   
   /**************************************
     
     CLASS
       Dir
      
       Short Description...
      
     GENERAL INFORMATION
      
       Dir.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       Dir
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class Dir {
   public:
      Dir();
      Dir(const Dir&);
      Dir(const std::string&);
      ~Dir();
      Dir& operator=(const Dir&);

      static Dir create(const std::string& name);
      
      void remove();
      Dir createSubdir(const std::string& name);
      Dir getSubdir(const std::string& name);

      std::string getName() const {
	 return d_name;
      }
   private:

      std::string d_name;
   };
}
} // end namespace Uintah

#endif

