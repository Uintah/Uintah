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
      ~Dir();
      Dir& operator=(const Dir&);

      static Dir create(const std::string& name);
      Dir createSubdir(const std::string& name);
      Dir getSubdir(const std::string& name);

      std::string getName() const {
	 return d_name;
      }
   private:
      Dir(const std::string&);
      Dir(const Dir&);

      std::string d_name;
   };
}
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/05/15 19:28:12  sparker
// New directory: OS for operating system interface classes
// Added a "Dir" class to create and iterate over directories (eventually)
//
//

#endif

