
#include <Packages/rtrt/Core/Names.h>
#include <map>
using namespace std;
using namespace rtrt;

static map<const Object*, const string> names;

void Names::nameObject(const string& name, const Object* ptr)
{
  names.insert(make_pair(ptr, name));
}

bool Names::hasName(const Object* ptr)
{
  return names.find(ptr) != names.end();
}

const string& Names::getName(const Object* ptr)
{
  static string empty("");
  map<const Object*, const string>::iterator iter = names.find(ptr);
  if(iter == names.end())
    return empty;
  else
    return iter->second;
}
