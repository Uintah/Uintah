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
  GuiString          op_gui_;

  typedef map<string, FieldSetHandle, less<string> > FSHMap;
  typedef map<string, FieldHandle, less<string> >    FHMap;
  FSHMap fsidmap_;
  FHMap  fidmap_;

public:
  ManageFieldSet(const clString& id);
  virtual ~ManageFieldSet();
  virtual void execute();

  void add_fieldset(clString widget, clString path, FieldSetHandle fieldset);
  void add_field(clString widget, clString path, FieldHandle field);

  void update_hiertable();

  virtual void connection(Module::ConnectionMode, int, int);
    
};


extern "C" Module* make_ManageFieldSet(const clString& id)
{
  return new ManageFieldSet(id);
}


ManageFieldSet::ManageFieldSet(const clString& id)
  : Module("ManageFieldSet", id, Filter, "Fields", "SCIRun"),
    op_gui_("op_gui", id, this)
{
}


ManageFieldSet::~ManageFieldSet()
{
}



void
ManageFieldSet::add_fieldset(clString widget, clString path,
			     FieldSetHandle fs)
{
  const clString newpath = path + " " + fs->get_string("name").c_str();
  clString res;

  TCL::eval(widget + " insert end \"" + newpath + "\"", res);
  string sres = res();
  fsidmap_[sres] = fs;
  TCL::eval(widget + " entry configure \"" + newpath + "\" -data { } ", res);

  vector<FieldSetHandle>::iterator fsi = fs->fieldset_begin();
  while (fsi != fs->fieldset_end())
  {
    add_fieldset(widget, newpath, *fsi);
    fsi++;
  }

  vector<FieldHandle>::iterator fi = fs->field_begin();
  while (fi != fs->field_end())
  {
    add_field(widget, newpath, *fi);
    fi++;
  }
}


void
ManageFieldSet::add_field(clString widget, clString path, FieldHandle f)
{
  clString res;

  TCL::eval(widget + " insert end \"" + path + " " +
		  f->get_string("name").c_str() + "\"", res);
  string sres = res();
  fidmap_[sres] = f;
  TCL::eval(widget + " entry configure \"" 
	    + path + " " + f->get_string("name").c_str() +
	    "\" -data { } ", res);
}


void
ManageFieldSet::update_hiertable()
{
  clString result;
  TCL::eval(".ui" + id + ".sel.h close -recursive 0", result);

  fsidmap_.clear();
  fidmap_.clear();

  dynamic_port_range range = get_iport("Input FieldSet");
  port_iter pi = range.first;
  while (pi != range.second)
  {
    FieldSetIPort *port = (FieldSetIPort *)get_iport((*pi).second);

    // Do something with port.
    FieldSetHandle h;
    port->get(h);

    if (h.get_rep())
    {
      add_fieldset(".ui" + id + ".sel.h", "", h);
    }

    ++pi;
  }

  range = get_iport("Input Field");
  pi = range.first;
  while (pi != range.second)
  {
    FieldIPort *port = (FieldIPort *)get_iport((*pi).second);

    // Do something with port.
    FieldHandle h;
    port->get(h);

    if (h.get_rep())
    {
      add_field(".ui" + id + ".sel.h", "", h);
    }

    ++pi;
  }
}



void
ManageFieldSet::connection(Module::ConnectionMode, int, int)
{
  update_hiertable();
}


static void
split(list<string> &result, const string vals)
{
  string::size_type index0 = 0;

  do {
    string::size_type index1 = vals.find(' ', index0);
    if (index1 > vals.size()) { index1 = vals.size(); }
    result.push_back(vals.substr(index0, index1 - index0));
    index0 = index1 + 1;
  } while (index0 < vals.size());
}


void
ManageFieldSet::execute()
{
  update_state(NeedData);

  clString op_gui = op_gui_.get();

  clString result;
  TCL::eval(".ui" + id + ".sel.h curselection", result);

  list<string> selected;
  split(selected, result());

  FieldSet *ofs = new FieldSet();
  ofs->set_string("name", "glomfield");

  list<string>::iterator si = selected.begin();
  while (si != selected.end())
  {
    const string &val = *si;
    ++si;

    FSHMap::iterator fsloc = fsidmap_.find(val);
    if (fsloc != fsidmap_.end())
    {
      ofs->push_back((*fsloc).second);
    }
    else
    {
      FHMap::iterator floc = fidmap_.find(val);
      if (floc != fidmap_.end())
      {
	ofs->push_back((*floc).second);
      }
      else
      {
	cerr << val << " not found in the iports.\n";
      }
    }
  }

  FieldSetHandle ofsh(ofs); 
  FieldSetOPort *ofsp = (FieldSetOPort *)get_oport(0);
  ofsp->send(ofsh);

  //FieldOPort *ofp = (FieldOPort *)get_oport(1);
}

} // End namespace SCIRun
