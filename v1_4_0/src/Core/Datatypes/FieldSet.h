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
 *  FieldSet.h: Templated Meshs defined on a 3D Regular Grid
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

#ifndef SCI_project_FieldSet_h
#define SCI_project_FieldSet_h 1

#include <Core/Datatypes/Field.h>


#include <vector>

namespace SCIRun {

using std::vector;

class FieldSet;
typedef LockingHandle<FieldSet> FieldSetHandle;

class SCICORESHARE FieldSet : public PropertyManager
{
private:

  vector<FieldHandle>    fields_;
  vector<FieldSetHandle> fieldsets_;

  static Persistent *maker();

public:

  FieldSet();
  FieldSet(const FieldSet &copy);
  virtual ~FieldSet();

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  // Some interface to add/remove fields.
  void add(FieldHandle field);
  void add(FieldSetHandle fieldset);

  bool remove(FieldHandle field);
  bool remove(FieldSetHandle fieldset);

  bool find_first_field(FieldHandle &result, const string name);

  vector<FieldSetHandle>::iterator fieldset_begin()
  { return fieldsets_.begin(); }
  vector<FieldSetHandle>::iterator fieldset_end()
  { return fieldsets_.end(); }
  vector<FieldHandle>::iterator field_begin()
  { return fields_.begin(); }
  vector<FieldHandle>::iterator field_end()
  { return fields_.end(); }

  void push_back(FieldSetHandle fh) { fieldsets_.push_back(fh); }
  void push_back(FieldHandle fh)    { fields_.push_back(fh); }
};


} // namespace SCIRun


#endif // SCI_project_FieldSet_h
