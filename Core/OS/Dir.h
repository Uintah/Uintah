/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#ifndef Core_OS_Dir_H
#define Core_OS_Dir_H

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

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
      bool exists();

     void getFilenamesBySuffix(const std::string& suffix,
			       std::vector<std::string>& filenames);

      std::string getName() const {
	 return name_;
      }
   private:

      std::string name_;
   };
} // End namespace SCIRun

#endif

