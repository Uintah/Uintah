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
 *  ModifyConductivities.cc:  Modify field conductivity tensor properties.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/FieldInterface.h>
#include <sci_hash_map.h>
#include <iostream>


namespace SCIRun {

class ModifyConductivities : public Module
{
private:
  int last_field_generation_;
  vector<pair<string, Tensor> > last_field_tensors_;
  vector<pair<string, Tensor> > last_gui_tensors_;
  bool addnew_;
  bool redo_gui_;

public:
  ModifyConductivities(GuiContext *context);
  virtual ~ModifyConductivities();

  void update_to_gui(const vector<pair<string, Tensor> > &tensors);
  bool update_from_gui(vector<pair<string, Tensor> > &tensors);
  bool different_tensors(const vector<pair<string, Tensor> > &a,
			 const vector<pair<string, Tensor> > &b);

  virtual void execute();
  virtual void tcl_command(GuiArgs &args, void *);
};


DECLARE_MAKER(ModifyConductivities)


ModifyConductivities::ModifyConductivities(GuiContext *context)
  : Module("ModifyConductivities", context, Filter, "Forward", "BioPSE"),
    last_field_generation_(0),
    addnew_(false),
    redo_gui_(false)
{
}



ModifyConductivities::~ModifyConductivities()
{
}


void
ModifyConductivities::update_to_gui(const vector<pair<string, Tensor> > &tensors)
{
  string result;
  gui->eval(id + " isopen", result);
  if (result != "open")
  {
    return;
  }
  gui->eval(id + " ui", result);
  gui->eval(id + " clear_all", result);

  for (unsigned int i=0; i < tensors.size(); i++)
  {
    string command = id + " set_item i" + to_string(i) +
      " { Material \"" + tensors[i].first + "\" Scale 1.0 C00 " +
      to_string(tensors[i].second.mat_[0][0]) + " C01 " +
      to_string(tensors[i].second.mat_[0][1]) + " C02 " +
      to_string(tensors[i].second.mat_[0][2]) + " C10 " +
      to_string(tensors[i].second.mat_[1][0]) + " C11 " +
      to_string(tensors[i].second.mat_[1][1]) + " C12 " +
      to_string(tensors[i].second.mat_[1][2]) + " C20 " +
      to_string(tensors[i].second.mat_[2][0]) + " C21 " +
      to_string(tensors[i].second.mat_[2][1]) + " C22 " +
      to_string(tensors[i].second.mat_[2][2]) + " }";

    gui->eval(command, result);
  }
}


static string
getafter(const string &after, const string &str)
{
  string::size_type start = str.find(after) + after.length() + 1;
  string::size_type end;
  if (str[start] != '{')
  {
    end = str.find(' ', start);
  }
  else
  {
    start++;
    end = str.find('}', start);
  }
  return str.substr(start, end - start);
}



bool
ModifyConductivities::update_from_gui(vector<pair<string, Tensor> > &tensors)
{
  string result;
  gui->eval(id + " isopen", result);
  if (result != "open")
  {
    return false;
  }

  for (unsigned int i = 0; i < tensors.size(); i++)
  {
    const string command = id + " get_item i" + to_string(i);
    gui->eval(command, result);

    tensors[i].first = getafter("Material", result);

    const string m00 = getafter("C00", result);
    if (m00 != to_string(tensors[i].second.mat_[0][0]))
    {
      tensors[i].second.mat_[0][0] = atof(m00.c_str());
    }

    const string m01 = getafter("C01", result);
    if (m01 != to_string(tensors[i].second.mat_[0][1]))
    {
      tensors[i].second.mat_[0][1] = atof(m01.c_str());
    }

    const string m02 = getafter("C02", result);
    if (m02 != to_string(tensors[i].second.mat_[0][2]))
    {
      tensors[i].second.mat_[0][2] = atof(m02.c_str());
    }

    const string m10 = getafter("C10", result);
    if (m10 != to_string(tensors[i].second.mat_[1][0]))
    {
      tensors[i].second.mat_[1][0] = atof(m10.c_str());
    }

    const string m11 = getafter("C11", result);
    if (m11 != to_string(tensors[i].second.mat_[1][1]))
    {
      tensors[i].second.mat_[1][1] = atof(m11.c_str());
    }

    const string m12 = getafter("C12", result);
    if (m12 != to_string(tensors[i].second.mat_[1][2]))
    {
      tensors[i].second.mat_[1][2] = atof(m12.c_str());
    }

    const string m20 = getafter("C20", result);
    if (m20 != to_string(tensors[i].second.mat_[2][0]))
    {
      tensors[i].second.mat_[2][0] = atof(m20.c_str());
    }

    const string m21 = getafter("C21", result);
    if (m21 != to_string(tensors[i].second.mat_[2][1]))
    {
      tensors[i].second.mat_[2][1] = atof(m21.c_str());
    }

    const string m22 = getafter("C22", result);
    if (m22 != to_string(tensors[i].second.mat_[2][2]))
    {
      tensors[i].second.mat_[2][2] = atof(m22.c_str());
    }

    const double scale = atof(getafter("Scale", result).c_str());
    if (scale != 1.0 && scale != 0.0)
    {
      tensors[i].second = tensors[i].second * scale;
    }
  }
  return true;
}


bool
ModifyConductivities::different_tensors(const vector<pair<string, Tensor> > &a,
					const vector<pair<string, Tensor> > &b)
{
  if (a.size() != b.size())
  {
    return true;
  }
  for (unsigned int i=0; i < a.size(); i++)
  {
    if (a[i].first != b[i].first)
    {
      return true;
    }
    for (int j = 0; j < 2; j++)
    {
      for (int k = 0; k < 2; k++)
      {
	if (a[i].second.mat_[j][k] != b[i].second.mat_[j][k])
	{
	  return true;
	}
      }
    }
  }
  return false;
}


void
ModifyConductivities::execute()
{
  // Read in the LatVolField<double>, clone it.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input");
  if (!ifp) {
    error("Unable to initialize iport 'Input'.");
    return;
  }
  FieldHandle field;
  if (!(ifp->get(field) && field.get_rep()))
  {
    return;
  }

  bool new_field_p = false;
  if (field->generation != last_field_generation_)
  {
    last_field_generation_ = field->generation;
    new_field_p = true;
  }

  // Get the tensors from the field.
  vector<pair<string, Tensor> > field_tensors;
  bool created_p = false;
  if (!field->get_property("conductivity_table", field_tensors))
  {
    //remark("Using identity conductivity tensors.");
    created_p = true;
    ScalarFieldInterface *sfi = field->query_scalar_interface(this);
    double minval, maxval;
    if (sfi)
    {
      sfi->compute_min_max(minval, maxval);
    }
    else
    {
      maxval = 1.0;
    }

    if (minval < 0 || maxval > 100)
    {
      error("Invalid number of tensors to create, no property to manage.");
      return;
    }

    field_tensors.resize((unsigned int)(maxval + 1.5));
    if (field_tensors.size() == last_gui_tensors_.size())
    {
      // Tensors in gui appear to work, just use those.
      field_tensors = last_gui_tensors_;
    }
    else
    {
      // Create some new ones.
      vector<double> t(6);
      t[0] = t[3] = t[5] = 1;
      t[1] = t[2] = t[4] = 0;

      Tensor tn(t);
      for (unsigned int i = 0; i < field_tensors.size(); i++)
      {
	field_tensors[i] =
	  pair<string, Tensor>("conductivity-" + to_string(i), tn);
      }
    }
    created_p = true;
  }
  
  // New input tensors, update the gui.
  if (different_tensors(field_tensors, last_field_tensors_))
  {
    update_to_gui(field_tensors);
    last_field_tensors_ = field_tensors;
    last_gui_tensors_ = field_tensors;
  }

  if (redo_gui_)
  {
    update_to_gui(last_gui_tensors_);
    redo_gui_ = false;
  }

  vector<pair<string, Tensor> > gui_tensors;
  gui_tensors.resize(last_gui_tensors_.size());
  if (!update_from_gui(gui_tensors))
  {
    gui_tensors = last_gui_tensors_;
  }
  if (addnew_)
  {
    addnew_ = false;

    string result;
    vector<double> t(6);
    t[0] = t[3] = t[5] = 1;
    t[1] = t[2] = t[4] = 0;

    const int i = gui_tensors.size();

    Tensor tn(t);
    gui_tensors.push_back(pair<string, Tensor>("conductivity-" + to_string(i),
					       tn));

    string command = id + " set_item i" + to_string(i) +
      " { Material \"" + gui_tensors[i].first + "\" Scale 1.0 C00 " +
      to_string(gui_tensors[i].second.mat_[0][0]) + " C01 " +
      to_string(gui_tensors[i].second.mat_[0][1]) + " C02 " +
      to_string(gui_tensors[i].second.mat_[0][2]) + " C10 " +
      to_string(gui_tensors[i].second.mat_[1][0]) + " C11 " +
      to_string(gui_tensors[i].second.mat_[1][1]) + " C12 " +
      to_string(gui_tensors[i].second.mat_[1][2]) + " C20 " +
      to_string(gui_tensors[i].second.mat_[2][0]) + " C21 " +
      to_string(gui_tensors[i].second.mat_[2][1]) + " C22 " +
      to_string(gui_tensors[i].second.mat_[2][2]) + " }";

    gui->eval(command, result);
  }
  
  bool changed_table_p = false;
  if (different_tensors(gui_tensors, last_gui_tensors_) ||
      different_tensors(gui_tensors, field_tensors) ||
      created_p)
  {
    field.detach();

    field->set_property("conductivity_table", gui_tensors, false);
    last_gui_tensors_ = gui_tensors;
    changed_table_p = true;
  }

  // Forward the results.
  FieldOPort *ofp = (FieldOPort *)get_oport("Output");
  if (!ofp) {
    error("Unable to initialize " + name + "'s Output port.");
    return;
  }
  ofp->send(field);
}


void
ModifyConductivities::tcl_command(GuiArgs &args, void *extra)
{
  if (args.count() == 2 && args[1] == "addnew")
  {
    addnew_ = true;
    if (!abort_flag)
    {
      abort_flag = 1;
      want_to_execute();
    }
  }
  else if (args.count() == 2 && args[1] == "redo_gui")
  {
    redo_gui_ = true;
    if (!abort_flag)
    {
      abort_flag = 1;
      want_to_execute();
    }
  }      
  else
  {
    Module::tcl_command(args, extra);
  }
}


} // End namespace SCIRun

