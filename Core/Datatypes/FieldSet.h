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
  static  const string type_name(int);
  virtual const string get_type_name(int n) const;

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
