// PropertyManager.h
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   January 2001
//
//  Copyright (C) 2001 SCI Institute
//
//  Manage dynamic properties of persistent objects.
//

#ifndef SCI_project_PropertyManager_h
#define SCI_project_PropertyManager_h 1

#include <Core/Datatypes/Datatype.h>
#include <map>

namespace SCIRun {


class PropertyManager : public Datatype
{
public:
  PropertyManager();
  PropertyManager(const PropertyManager &copy);
  virtual ~PropertyManager();

  void set_string(const string name, const string property);
  const string get_string(const string name);

  void set_data(const string name, Datatype *data);
  Datatype *get_data(const string name);


  // GROUP: Support of persistent representation
  //////////
  //
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;

private:
  
  struct Property
  {
    Property();

    string    stringval;
    Datatype *dataval;
    bool      tmp;
  };

  typedef std::map<string, Property *, std::less<string> > map_type;

  map_type properties_;
};


} // namespace SCIRun

#endif // SCI_project_PropertyManager_h
