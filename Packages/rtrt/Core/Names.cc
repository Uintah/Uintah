
#include <Packages/rtrt/Core/Names.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <sgi_stl_warnings_on.h>

using namespace std;
using namespace rtrt;

static map<const Object*, const string> names;
static int unique_name_index = 0;

void Names::nameObjectWithUnique(const Object* ptr) {
  char buf[20];
  sprintf(buf, "Object%2d", unique_name_index);
  unique_name_index++;
  nameObject(string(buf), ptr);
}

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
