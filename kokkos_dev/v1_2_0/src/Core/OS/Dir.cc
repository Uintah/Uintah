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


#include <Core/OS/Dir.h>
#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <iostream>
#include <unistd.h>

using namespace std;

using namespace SCIRun;

Dir Dir::create(const string& name)
{
   int code = mkdir(name.c_str(), 0777);
   if(code != 0)
      throw ErrnoException("Dir::create (mkdir call)", errno);
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

void Dir::remove()
{
  int code = rmdir(name_.c_str());
  if (code != 0)
    throw ErrnoException("Dir::remove (rmdir call)", errno);
  return;
}

void Dir::forceRemove()
{
   int code = system((string("rm -f -r ") + name_).c_str());
   if (code != 0)
      throw InternalError(string("Dir::remove failed to remove: ") + name_);
   return;
}

void Dir::remove(const string& filename)
{
   string filepath = name_ + "/" + filename;
   int code = system((string("rm -f ") + filepath).c_str());
   if (code != 0)
      throw InternalError(string("Dir::remove failed to remove: ") + filepath);
   return;
}

Dir Dir::createSubdir(const string& sub)
{
   return create(name_+"/"+sub);
}

Dir Dir::getSubdir(const string& sub)
{
   // This should probably do more
   return Dir(name_+"/"+sub);
}

void Dir::copy(Dir& destDir)
{
   int code = system((string("cp -r ") + name_ + " " + destDir.name_).c_str());
   if (code != 0)
      throw InternalError(string("Dir::copy failed to copy: ") + name_);
   return;
}

void Dir::move(Dir& destDir)
{
   int code = system((string("mv ") + name_ + " " + destDir.name_).c_str());
   if (code != 0)
      throw InternalError(string("Dir::move failed to move: ") + name_);
   return;
}

void Dir::copy(const std::string& filename, Dir& destDir)
{
   string filepath = name_ + "/" + filename;
   int code =system((string("cp ") + filepath + " " + destDir.name_).c_str());
   if (code != 0)
      throw InternalError(string("Dir::copy failed to copy: ") + filepath);
   return;
}

void Dir::move(const std::string& filename, Dir& destDir)
{
   string filepath = name_ + "/" + filename;
   int code =system((string("mv ") + filepath + " " + destDir.name_).c_str());
   if (code != 0)
      throw InternalError(string("Dir::move failed to move: ") + filepath);
   return;
}

