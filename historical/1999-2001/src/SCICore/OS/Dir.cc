
#include <SCICore/OS/Dir.h>
#include <SCICore/Exceptions/ErrnoException.h>
#include <SCICore/Exceptions/InternalError.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>

using namespace SCICore::OS;
using namespace SCICore::Exceptions;
using namespace std;

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
   : d_name(name)
{
}

Dir::~Dir()
{
}

Dir& Dir::operator=(const Dir& copy)
{
   d_name = copy.d_name;
   return *this;
}

void Dir::remove()
{
   int code = rmdir(d_name.c_str());
   if (code != 0)
      throw ErrnoException("Dir::remove (rmdir call)", errno);
   return;
}

void Dir::forceRemove()
{
   int code = system((string("rm -f -r ") + d_name).c_str());
   if (code != 0)
      throw InternalError(string("Dir::remove failed to remove: ") + d_name);
   return;
}

void Dir::remove(const string& filename)
{
   string filepath = d_name + "/" + filename;
   int code = system((string("rm -f ") + filepath).c_str());
   if (code != 0)
      throw InternalError(string("Dir::remove failed to remove: ") + filepath);
   return;
}

Dir Dir::createSubdir(const string& sub)
{
   return create(d_name+"/"+sub);
}

Dir Dir::getSubdir(const string& sub)
{
   // This should probably do more
   return Dir(d_name+"/"+sub);
}

void Dir::copy(Dir& destDir)
{
   int code = system((string("cp -r ") + d_name + " " + destDir.d_name).c_str());
   if (code != 0)
      throw InternalError(string("Dir::copy failed to copy: ") + d_name);
   return;
}

void Dir::move(Dir& destDir)
{
   int code = system((string("mv ") + d_name + " " + destDir.d_name).c_str());
   if (code != 0)
      throw InternalError(string("Dir::move failed to move: ") + d_name);
   return;
}

void Dir::copy(const std::string& filename, Dir& destDir)
{
   string filepath = d_name + "/" + filename;
   int code =system((string("cp ") + filepath + " " + destDir.d_name).c_str());
   if (code != 0)
      throw InternalError(string("Dir::copy failed to copy: ") + filepath);
   return;
}

void Dir::move(const std::string& filename, Dir& destDir)
{
   string filepath = d_name + "/" + filename;
   int code =system((string("mv ") + filepath + " " + destDir.d_name).c_str());
   if (code != 0)
      throw InternalError(string("Dir::move failed to move: ") + filepath);
   return;
}

//
// $Log$
// Revision 1.4  2001/01/08 17:19:31  witzel
// Added copy, move, forceRemove, and remove(file) methods.
//
// Revision 1.3  2000/06/15 19:52:38  sparker
// Removed chatter
//
// Revision 1.2  2000/05/31 15:20:44  jehall
// - Added ability to remove() directories
//
// Revision 1.1  2000/05/15 19:28:12  sparker
// New directory: OS for operating system interface classes
// Added a "Dir" class to create and iterate over directories (eventually)
//
//
