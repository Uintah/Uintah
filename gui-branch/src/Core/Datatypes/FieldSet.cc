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

/*
 *  FieldSet.cc: Templated Meshs defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#include <Core/Datatypes/FieldSet.h>
#include <Core/Persistent/PersistentSTL.h>

namespace SCIRun {

using namespace std;

FieldSet::FieldSet()
{
}


FieldSet::FieldSet(const FieldSet &copy)
  : fields_(copy.fields_), fieldsets_(copy.fieldsets_)
{
}


FieldSet::~FieldSet()
{
}


void
FieldSet::add(FieldHandle field)
{
  fields_.push_back(field);
}


void
FieldSet::add(FieldSetHandle fieldset)
{
  fieldsets_.push_back(fieldset);
}


bool
FieldSet::remove(FieldHandle field)
{
  vector<FieldHandle>::iterator fi = fields_.begin();
  while (fi != fields_.end())
  {
    if ((*fi).get_rep() == field.get_rep())
    {
      break;
    }
  }

  if (fi != fields_.end())
  {
    fields_.erase(fi);
    return true;
  }
  return false;
}


bool
FieldSet::remove(FieldSetHandle fieldset)
{
  vector<FieldSetHandle>::iterator fi = fieldsets_.begin();
  while (fi != fieldsets_.end())
  {
    if ((*fi).get_rep() == fieldset.get_rep())
    {
      break;
    }
  }

  if (fi != fieldsets_.end())
  {
    fieldsets_.erase(fi);
    return true;
  }
  return false;
}


bool
FieldSet::find_first_field(FieldHandle &result, const string name)
{
  // Depth first search for name. 
  vector<FieldHandle>::iterator fi = fields_.begin();
  while (fi != fields_.end())
  {
    string s;
    if ((*fi)->get("name", s) && s == name)
    {
      result = *fi;
      return true;
    }
    ++fi;
  }

  vector<FieldSetHandle>::iterator fsi = fieldsets_.begin();
  while (fsi != fieldsets_.end())
  {
    if ((*fsi)->find_first_field(result, name))
    {
      return true;
    }
    ++fsi;
  }
  return false;
}


void
FieldSet::io(Piostream &stream)
{
  PropertyManager::io(stream);
  Pio(stream, fields_);
  Pio(stream, fieldsets_);
}


Persistent *
FieldSet::maker()
{
  return new FieldSet();
}


PersistentTypeID FieldSet::type_id("FieldSet", "PropertyManager", maker);

const string
FieldSet::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "FieldSet";
  return name;
}


} // namespace SCIRun
