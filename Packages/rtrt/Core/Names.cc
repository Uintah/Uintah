
#include <Packages/rtrt/Core/Names.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <sgi_stl_warnings_on.h>

using namespace std;
using namespace rtrt;

static map<const Object*, const string> obj_names;
static int unique_obj_name_index = 0;

void Names::nameObjectWithUnique(const Object* ptr) {
  char buf[20];
  sprintf(buf, "Object%2d", unique_obj_name_index);
  unique_obj_name_index++;
  nameObject(string(buf), ptr);
}

void Names::nameObject(const string& name, const Object* ptr)
{
  obj_names.insert(make_pair(ptr, name));
}

bool Names::hasName(const Object* ptr)
{
  return obj_names.find(ptr) != obj_names.end();
}

const string& Names::getName(const Object* ptr)
{
  static string empty("");
  map<const Object*, const string>::iterator iter = obj_names.find(ptr);
  if(iter == obj_names.end())
    return empty;
  else
    return iter->second;
}

/////////////////////////////////////////////////////////////
//
// Stuff for materials
//

static map<const Material*, const string> mat_names;
static int unique_mat_name_index = 0;

void Names::nameMaterialWithUnique(const Material* ptr) {
  char buf[20];
  sprintf(buf, "Material%2d", unique_mat_name_index);
  unique_mat_name_index++;
  nameMaterial(string(buf), ptr);
}

void Names::nameMaterial(const string& name, const Material* ptr)
{
  mat_names.insert(make_pair(ptr, name));
}

bool Names::hasName(const Material* ptr)
{
  return mat_names.find(ptr) != mat_names.end();
}

const string& Names::getName(const Material* ptr)
{
  static string empty("");
  map<const Material*, const string>::iterator iter = mat_names.find(ptr);
  if(iter == mat_names.end())
    return empty;
  else
    return iter->second;
}
