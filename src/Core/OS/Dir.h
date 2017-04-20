/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Core_OS_Dir_H
#define Core_OS_Dir_H

#include <string>
#include <vector>

#include <sys/stat.h>

namespace Uintah {
   
/**************************************
     
     CLASS
       Dir
      
       Short Description...
      
     GENERAL INFORMATION
      
       Dir.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
      
     KEYWORDS
       Dir
      
     DESCRIPTION
       Long description...
      
     WARNING
      
****************************************/
    
class Dir {
public:
  Dir();
  Dir( const Dir& );
  Dir( const std::string& );
  ~Dir();
  Dir& operator=( const Dir& );
  
  static Dir create( const std::string & name );
  
  void remove( bool throwOnError = true ) const;
    
  // Removes even if the directory has contents.
  void forceRemove( bool throwOnError = true );

  // Remove a file.
  void remove( const std::string & filename, bool throwOnError = true ) const;

  // Copy this directory to under the destination directory.
  void copy( const Dir & destDir ) const;
  void move(       Dir & destDir );

  // Copy a file in this directory to the destination directory.
  void copy( const std::string & filename, const Dir & destDir ) const;
  void move( const std::string & filename, Dir & destDir );
  
  Dir  createSubdir( const std::string & name );
  
  // tries 500 times to create a subdir
  Dir  createSubdirPlus( const std::string & sub );
  
  Dir  getSubdir(    const std::string & name ) const;
  bool exists();
  
  void getFilenamesBySuffix( const std::string              & suffix,
                                   std::vector<std::string> & filenames ) const;

  std::string getName() const {
    return name_;
  }

  // Delete the dir and all of its files/sub directories using C++ calls.
  //
  static bool removeDir( const char * dirName );
  
private:

  std::string name_;
};

} // End namespace Uintah

#define MKDIR(dir, perm) mkdir(dir, perm)
#define LSTAT(file, buf) lstat(file, buf)

#endif // Core_OS_Dir_H
