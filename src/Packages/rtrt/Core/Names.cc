/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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



#include <Packages/rtrt/Core/Names.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <cstdio>
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
