/*
 *  ManageFieldSet: Manage the members of a field set -- 
 *                                  create, add, delete, extract
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/FieldSet.h>
#include <Dataflow/Ports/FieldSetPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

#include <map>
#include <list>

using std::map;
using std::list;


namespace SCIRun {

class ManageFieldSet : public Module
{
  GuiString          state_gui_;

  typedef map<string, FieldSetHandle, less<string> > FSHMap;
  typedef map<string, FieldHandle, less<string> >    FHMap;
  FSHMap fsidmap_;
  FHMap  fidmap_;

public:
  ManageFieldSet(const clString& id);
  virtual ~ManageFieldSet();
  virtual void execute();

  void add_fieldset(clString path, FieldSetHandle fieldset);
  void add_field(clString path, FieldHandle field);

  void update_hiertable();

  const clString get_name(PropertyManager *pm);
  const clString data_at_to_string(Field::data_location loc);

  virtual void connection(Module::ConnectionMode mode, int a, int b);
};


extern "C" Module* make_ManageFieldSet(const clString& id)
{
  return new ManageFieldSet(id);
}


ManageFieldSet::ManageFieldSet(const clString& id)
  : Module("ManageFieldSet", id, Filter, "Fields", "SCIRun"),
    state_gui_("state", id, this)
{
}


ManageFieldSet::~ManageFieldSet()
{
}


void
ManageFieldSet::add_fieldset(clString path, FieldSetHandle fs)
{
  const clString name = get_name(fs.get_rep());
  clString newpath;
  TCL::eval(id + " add_sitem " + path + " " + name, newpath);
  const string sindex = newpath();
  fsidmap_[sindex] = fs;

  vector<FieldSetHandle>::iterator fsi = fs->fieldset_begin();
  while (fsi != fs->fieldset_end())
  {
    add_fieldset(newpath, *fsi);
    fsi++;
  }

  vector<FieldHandle>::iterator fi = fs->field_begin();
  while (fi != fs->field_end())
  {
    add_field(newpath, *fi);
    fi++;
  }
}


void
ManageFieldSet::add_field(clString path, FieldHandle f)
{
  const clString name = get_name(f.get_rep());
  const clString data = clString("{") +
    " Datatype " + "unknown" +
    " Location " + data_at_to_string(f->data_at()) +
    " }";

  clString index;
  TCL::eval(id + " add_fitem " + path + " " +  name + " " + data, index);

  const string sindex = index();
  fidmap_[sindex] = f;
}


void
ManageFieldSet::update_hiertable()
{
  clString result;
  TCL::eval(id + " ui", result);
  TCL::eval(id + " clear_all", result);

  fsidmap_.clear();
  fidmap_.clear();

  dynamic_port_range range = get_iports("Input FieldSet");
  port_iter pi = range.first;
  while (pi != range.second)
  {
    FieldSetIPort *port = (FieldSetIPort *)get_iport(pi->second);

    // Do something with port.
    FieldSetHandle h;
    if (port->get(h))
    {
      add_fieldset("0", h);
    }

    ++pi;
  }

  range = get_iports("Input Field");
  pi = range.first;
  while (pi != range.second)
  {
    FieldIPort *port = (FieldIPort *)get_iport(pi->second);

    // Do something with port.
    FieldHandle h;
    if (port->get(h))
    {
      add_field("0", h);
    }

    ++pi;
  }
}


const clString
ManageFieldSet::get_name(PropertyManager *pm)
{
  string n;
  if ( pm->get("name", n) ) {
    return n.c_str();
  }
  else
  {
    char buffer[256];
    sprintf(buffer, "#<field-%lu>", (unsigned long)pm->generation);
    return clString(buffer);
  }
}

const clString
ManageFieldSet::data_at_to_string(Field::data_location loc)
{
  switch(loc)
  {
  case Field::NODE:
    return "node";
  case Field::EDGE:
    return "edge";
  case Field::FACE:
    return "face";
  case Field::CELL:
    return "cell";
  case Field::NONE:
    return "none";
  default:
    return "unknown";
  }
}


static void
split(list<string> &result, const string vals)
{
  string::size_type index0 = 0;

  do {
    string::size_type index1 = vals.find(' ', index0);
    if (index1 > vals.size()) { index1 = vals.size(); }
    int len = index1 - index0;
    if (len > 0)
    {
      result.push_back(vals.substr(index0, len));
    }
    index0 = index1 + 1;
  } while (index0 < vals.size());
}


void
ManageFieldSet::connection(Module::ConnectionMode mode, int a, int b)
{
  Module::connection(mode, a, b);
  //update_hiertable();
}


void
ManageFieldSet::execute()
{
  clString state_gui = state_gui_.get();
  
  if (state_gui != "output")
  {
    update_hiertable();
  }
  else
  {
    cout << "outputting\n";

    state_gui_.set("update");

    FieldSet *ofs = NULL;

    clString result;
    TCL::eval(".ui" + id + ".sel.h curselection", result);

    list<string> selected;
    split(selected, result());

    list<string>::iterator si = selected.begin();
    while (si != selected.end())
    {
      const string val = *si;
      ++si;

      cout << "Looking for '" << val << "'\n";

      FSHMap::iterator fsloc = fsidmap_.find(val);
      if (fsloc != fsidmap_.end())
      {
	if (ofs == NULL)
	{
	  ofs = new FieldSet();
	}
	ofs->push_back((*fsloc).second);
      }
      else
      {
	FHMap::iterator floc = fidmap_.find(val);
	if (floc != fidmap_.end())
	{
	  if (ofs == NULL)
	  {
	    ofs = new FieldSet();
	  }
	  ofs->push_back((*floc).second);
	}
	else
	{
	  cout << "Could not find '" << val << "' in the iports.\n";
	}
      }
    }

    if (ofs != NULL)
    {
      cout << "Dumping out a field set\n";
      ofs->store("name", string("glomfield"));
      FieldSetHandle ofsh(ofs);
      FieldSetOPort *ofsp = (FieldSetOPort *)get_oport("Output FieldSet");
      ofsp->send(ofsh);
    }
  }
}

} // End namespace SCIRun
