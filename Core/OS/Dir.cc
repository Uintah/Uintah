
#include <SCICore/OS/Dir.h>
#include <SCICore/Exceptions/ErrnoException.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <iostream>
#include <unistd.h>

using namespace SCICore::OS;
using namespace SCICore::Exceptions;
using namespace std;

Dir Dir::create(const string& name)
{
   cerr << "Creating: " << name << '\n';
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
    cerr << "Removing: " << d_name << "\n";
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

//
// $Log$
// Revision 1.2  2000/05/31 15:20:44  jehall
// - Added ability to remove() directories
//
// Revision 1.1  2000/05/15 19:28:12  sparker
// New directory: OS for operating system interface classes
// Added a "Dir" class to create and iterate over directories (eventually)
//
//
