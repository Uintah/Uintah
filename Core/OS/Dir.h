#ifndef Core_OS_Dir_H
#define Core_OS_Dir_H

#include <string>

namespace SCIRun {
   
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
      void forceRemove(); // removes even if the directory has contents

      void remove(const std::string& filename); // remove a file

      // copy this directory to under the destination directory
      void copy(Dir& destDir);
      void move(Dir& destDir);

      // copy a file in this directory to the destination directory
      void copy(const std::string& filename, Dir& destDir);
      void move(const std::string& filename, Dir& destDir);

      Dir createSubdir(const std::string& name);
      Dir getSubdir(const std::string& name);

      std::string getName() const {
	 return d_name;
      }
   private:

      std::string d_name;
   };
} // End namespace SCIRun

#endif

