/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#include <Core/OS/Dir.h>

#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/FileUtils.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <errno.h>

#include <iostream>

#ifndef _WIN32
#  include <unistd.h>
#  include <dirent.h>
#else
#  include <Core/Malloc/Allocator.h>
#endif

using namespace std;
using namespace SCIRun;

Dir Dir::create(const string& name)
{
   int code = MKDIR(name.c_str(), 0777);
   if(code != 0)
      throw ErrnoException("Dir::create (mkdir call): " + name, errno, __FILE__, __LINE__);
   return Dir(name);
}

Dir::Dir()
{
}

Dir::Dir(const string& name)
   : name_(name)
{
}

Dir::Dir(const Dir& dir)
  : name_(dir.name_)
{
}

Dir::~Dir()
{
}

Dir& Dir::operator=(const Dir& copy)
{
   name_ = copy.name_;
   return *this;
}

void
Dir::remove(bool throwOnError)
{
  int code = rmdir(name_.c_str());
  if (code != 0) {
    ErrnoException exception("Dir::remove()::rmdir: " + name_, errno, __FILE__, __LINE__);
    if (throwOnError)
      throw exception;
    else
      cerr << "WARNING: " << exception.message() << "\n";
  }
  return;
}

// Removes a directory (and all its contents) using C++ calls.  Returns true if it succeeds.
bool
Dir::removeDir( const char * dirName )
{
  // Walk through the dir, delete files, find sub dirs, recurse, delete sub dirs, ...

  DIR *    dir = opendir( dirName );
  dirent * file = 0;

  if( dir == NULL ) {
    cout << "Error in Dir::removeDir():\n";
    cout << "  opendir failed for " << dirName << "\n";
    cout << "  - errno is " << errno << "\n";
    return false;
  }

  file = readdir(dir);
  while( file ) {
    if (strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") !=0) {
      string fullpath = string(dirName) + "/" + file->d_name;
      if( file->d_type & DT_DIR ) {
        removeDir( fullpath.c_str() );
      } else {
        int rc = ::remove( fullpath.c_str() );
        if (rc != 0) {
          cout << "WARNING: remove() failed for '" << fullpath.c_str() 
               << "'.  Return code is: " << rc << ", errno: " << errno << ": " << strerror(errno) << "\n";
          return false;
        }
      }
    }
    file = readdir(dir);
  }

  closedir(dir);
  int code = rmdir( dirName );

  if (code != 0) {
    cerr << "Error, rmdir failed for '" << dirName << "'\n";
    cerr << "  errno is " << errno << "\n";
    return false;
  }
  return true;
}

void
Dir::forceRemove(bool throwOnError)
{
   int code = system((string("rm -f -r ") + name_).c_str());
   if (code != 0) {
     ErrnoException exception(string("Dir::forceRemove() failed to remove: ") + name_, errno, __FILE__, __LINE__);
     if (throwOnError)
       throw exception;
     else
       cerr << "WARNING: " << exception.message() << "\n";
   }
}

void
Dir::remove(const string& filename, bool throwOnError)
{
   string filepath = name_ + "/" + filename;
   int code = system((string("rm -f ") + filepath).c_str());
   if (code != 0) {
     ErrnoException exception(string("Dir::remove failed to remove: ") + filepath, errno, __FILE__, __LINE__);
     if (throwOnError)
       throw exception;
     else
       cerr << "WARNING: " << exception.message() << "\n";
   }
}

Dir
Dir::createSubdir(const string& sub)
{
   return create(name_+"/"+sub);
}

Dir
Dir::getSubdir(const string& sub)
{
   // This should probably do more
   return Dir(name_+"/"+sub);
}

void
Dir::copy(Dir& destDir)
{
   int code = system((string("cp -r ") + name_ + " " + destDir.name_).c_str());
   if (code != 0)
      throw InternalError(string("Dir::copy failed to copy: ") + name_, __FILE__, __LINE__);
   return;
}

void
Dir::move(Dir& destDir)
{
   int code = system((string("mv ") + name_ + " " + destDir.name_).c_str());
   if (code != 0)
      throw InternalError(string("Dir::move failed to move: ") + name_, __FILE__, __LINE__);
   return;
}

void
Dir::copy(const std::string& filename, Dir& destDir)
{
   string filepath = name_ + "/" + filename;
   int code =system((string("cp ") + filepath + " " + destDir.name_).c_str());
   if (code != 0)
      throw InternalError(string("Dir::copy failed to copy: ") + filepath, __FILE__, __LINE__);
   return;
}

void
Dir::move(const std::string& filename, Dir& destDir)
{
   string filepath = name_ + "/" + filename;
   int code =system((string("mv ") + filepath + " " + destDir.name_).c_str());
   if (code != 0)
      throw InternalError(string("Dir::move failed to move: ") + filepath, __FILE__, __LINE__);
   return;
}

void
Dir::getFilenamesBySuffix( const std::string& suffix,
                           vector<string>& filenames )
{
  DIR* dir = opendir(name_.c_str());
  if (!dir) 
    return;
  const char* ext = suffix.c_str();
  for(dirent* file = readdir(dir); file != 0; file = readdir(dir)){
    if ((strlen(file->d_name)>=strlen(ext)) && 
	(strcmp(file->d_name+strlen(file->d_name)-strlen(ext),ext)==0)) {
      filenames.push_back(file->d_name);
      cout << "  Found " << file->d_name << "\n";
    }
  }
}

bool
Dir::exists()
{
  struct stat buf;
  if(LSTAT(name_.c_str(),&buf) != 0)
    return false;
  if (S_ISDIR(buf.st_mode))
    return true;
  return false;
}

#ifdef _WIN32
struct DIR
{
  long file_handle;
  _finddata_t finddata;
  _finddata_t nextfinddata;
  bool done;
  dirent return_on_read;
};

DIR *
opendir(const char *name)
{
  // grab the first file in the directory by ending name with "/*"
  DIR *dir = 0;
  if (name != NULL) {
    unsigned length = strlen(name);
    if (length > 0) {
      dir = scinew DIR;
      dir->done = false;

      // create the file search path with the wildcard
      string search_path = name; 
      if (name[length-1] == '/' || name[length-1] == '\\')
        search_path += "*";
      else
        search_path += "/*";

      if ((dir->file_handle = (long) _findfirst(search_path.c_str(), &dir->nextfinddata)) == -1) {
        delete dir;
        dir = 0;
      }
      return dir;
    }
  }
  errno = EINVAL;
  return 0;
}

int
closedir(DIR *dir)
{
  int result = -1;
  if (dir) {
    result = _findclose(dir->file_handle);
    delete dir;
  }
  if (result == -1) errno = EBADF;
  return result;
}

dirent *readdir(DIR *dir)
{
  if (dir->done) 
    return 0;
  if (dir) {
    dir->finddata = dir->nextfinddata;
    if (_findnext(dir->file_handle, &dir->nextfinddata) == -1)
      dir->done = true;
    dir->return_on_read.d_name = dir->finddata.name;
    return &dir->return_on_read;
  }
  else {
    errno = EBADF;
  }
  return 0;
}

void
rewinddir(DIR *dir)
{
  cout << "  WARNING!!!  Core/OS/Dir.cc rewinddir() is not implemented under Windows!\n";
}
#endif
