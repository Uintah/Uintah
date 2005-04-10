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
}
} // end namespace Uintah

//
// $Log$
// Revision 1.4  2001/01/08 17:19:31  witzel
// Added copy, move, forceRemove, and remove(file) methods.
//
// Revision 1.3  2000/09/25 18:02:33  sparker
// include errno.h instead of explicitly defining extern int errno
//
// Revision 1.2  2000/05/31 15:20:44  jehall
// - Added ability to remove() directories
//
// Revision 1.1  2000/05/15 19:28:12  sparker
// New directory: OS for operating system interface classes
// Added a "Dir" class to create and iterate over directories (eventually)
//
//

#endif

