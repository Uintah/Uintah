
#include <Core/OS/Dir.h>
#include <Core/Exceptions/ErrnoException.h>
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

Dir Dir::createSubdir(const string& sub)
{
   return create(d_name+"/"+sub);
}

Dir Dir::getSubdir(const string& sub)
{
   // This should probably do more
   return Dir(d_name+"/"+sub);
}
