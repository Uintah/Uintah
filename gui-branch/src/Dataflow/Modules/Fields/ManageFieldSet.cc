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
  ManageFieldSet(const string& id);
  virtual ~ManageFieldSet();
  virtual void execute();

  void add_fieldset(string path, FieldSetHandle fieldset);
  void add_field(string path, FieldHandle field);

  void update_hiertable();

  const string get_name(PropertyManager *pm);
  const string data_at_to_string(Field::data_location loc);

  virtual void connection(Module::ConnectionMode mode, int a, int b);
};


extern "C" Module* make_ManageFieldSet(const string& id)
{
  return new ManageFieldSet(id);
}


ManageFieldSet::ManageFieldSet(const string& id)
  : Module("ManageFieldSet", id, Filter, "Fields", "SCIRun"),
    state_gui_("state", id, this)
{
}


ManageFieldSet::~ManageFieldSet()
{
}


void
ManageFieldSet::add_fieldset(string path, FieldSetHandle fs)
{
  const string name = get_name(fs.get_rep());
  string newpath;
  TCL::eval(id + " add_sitem " + path + " " + name, newpath);
  const string sindex = newpath;
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
ManageFieldSet::add_field(string path, FieldHandle f)
{
  const string name = get_name(f.get_rep());
  const string data = string("{") +
    " Datatype " + "unknown" +
    " Location " + data_at_to_string(f->data_at()) +
    " }";

  string index;
  TCL::eval(id + " add_fitem " + path + " " +  name + " " + data, index);

  const string sindex = index;
  fidmap_[sindex] = f;
}


void
ManageFieldSet::update_hiertable()
{
  string result;
  TCL::eval(id + " ui", result);
  TCL::eval(id + " clear_all", result);

  fsidmap_.clear();
  fidmap_.clear();

  dynamic_port_range range = get_iports("Input FieldSet");
  port_map::iterator pi = range.first;
  while (pi != range.second)
  {
    FieldSetIPort *port = (FieldSetIPort *)get_iport(pi->second);
    if (!port) {
      postMessage("Unable to initialize "+name+"'s iport\n");
      return;
    }
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
    if (!port) {
      postMessage("Unable to initialize "+name+"'s iport\n");
      return;
    }
    // Do something with port.
    FieldHandle h;
    if (port->get(h))
    {
      add_field("0", h);
    }

    ++pi;
  }
}


const string
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
    return string(buffer);
  }
}

const string
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
  string state_gui = state_gui_.get();
  
  if (state_gui != "output")
  {
    update_hiertable();
  }
  else
  {
    state_gui_.set("update");

    FieldSet *ofs = NULL;

    string result;
    TCL::eval(".ui" + id + ".sel.h curselection", result);

    list<string> selected;
    split(selected, result);

    list<string>::iterator si = selected.begin();
    while (si != selected.end())
    {
      const string val = *si;
      ++si;

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
	  remark("Could not find '" + val + "' in the iports.");
	}
      }
    }

    if (ofs != NULL)
    {
      remark("Dumping out a field set.");
      ofs->store("name", string("glomfield"), false);
      FieldSetHandle ofsh(ofs);
      FieldSetOPort *ofsp = (FieldSetOPort *)get_oport("Output FieldSet");
      if (!ofsp) {
	postMessage("Unable to initialize "+name+"'s oport\n");
	return;
      }
      ofsp->send(ofsh);
    }
  }
}

} // End namespace SCIRun
