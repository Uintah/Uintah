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
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/LatticeVol.h>
#include <sci_hash_map.h>
#include <iostream>


namespace SCIRun {

class ModifyConductivities : public Module
{
private:
  unsigned int last_gui_hash_;
  unsigned int last_field_hash_;
  int last_field_gen_;

public:
  ModifyConductivities(const string& id);
  virtual ~ModifyConductivities();

  void update_gui(const vector<pair<string, Tensor> > &tensors);
  void push_changes(vector<pair<string, Tensor> > &tensors);
  unsigned int hash_tensors(vector<pair<string, Tensor> > &tensors);
  virtual void execute();
};


extern "C" Module* make_ModifyConductivities(const string& id) {
  return new ModifyConductivities(id);
}


ModifyConductivities::ModifyConductivities(const string& id)
  : Module("ModifyConductivities", id, Filter, "Forward", "BioPSE"),
    last_gui_hash_(0),
    last_field_hash_(0),
    last_field_gen_(0)
{
}



ModifyConductivities::~ModifyConductivities()
{
}


void
ModifyConductivities::update_gui(const vector<pair<string, Tensor> > &tensors)
{
  string result;
  TCL::eval(id + " ui", result);
  TCL::eval(id + " clear_all", result);

  for (unsigned int i=0; i < tensors.size(); i++)
  {
    string command = id + " set_item i" + to_string(i) +
      " { Material \"" + tensors[i].first + "\" Scale 1.0 M00 " +
      to_string(tensors[i].second.mat_[0][0]) + " M01 " +
      to_string(tensors[i].second.mat_[0][1]) + " M02 " +
      to_string(tensors[i].second.mat_[0][2]) + " M10 " +
      to_string(tensors[i].second.mat_[1][0]) + " M11 " +
      to_string(tensors[i].second.mat_[1][1]) + " M12 " +
      to_string(tensors[i].second.mat_[1][2]) + " M20 " +
      to_string(tensors[i].second.mat_[2][0]) + " M21 " +
      to_string(tensors[i].second.mat_[2][1]) + " M22 " +
      to_string(tensors[i].second.mat_[2][2]) + " }";

    TCL::eval(command, result);
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



void
ModifyConductivities::push_changes(vector<pair<string, Tensor> > &tensors)
{
  for (unsigned int i = 0; i < tensors.size(); i++)
  {
    string result;
    string command = id + " get_item i" + to_string(i);
    TCL::eval(command, result);

    tensors[i].first = getafter("Material", result);

    const string m00 = getafter("M00", result);
    if (m00 != to_string(tensors[i].second.mat_[0][0]))
    {
      tensors[i].second.mat_[0][0] = atof(m00.c_str());
    }

    const string m01 = getafter("M01", result);
    if (m01 != to_string(tensors[i].second.mat_[0][1]))
    {
      tensors[i].second.mat_[0][1] = atof(m01.c_str());
    }

    const string m02 = getafter("M02", result);
    if (m02 != to_string(tensors[i].second.mat_[0][2]))
    {
      tensors[i].second.mat_[0][2] = atof(m02.c_str());
    }

    const string m10 = getafter("M10", result);
    if (m10 != to_string(tensors[i].second.mat_[1][0]))
    {
      tensors[i].second.mat_[1][0] = atof(m10.c_str());
    }

    const string m11 = getafter("M11", result);
    if (m11 != to_string(tensors[i].second.mat_[1][1]))
    {
      tensors[i].second.mat_[1][1] = atof(m11.c_str());
    }

    const string m12 = getafter("M12", result);
    if (m12 != to_string(tensors[i].second.mat_[1][2]))
    {
      tensors[i].second.mat_[1][2] = atof(m12.c_str());
    }

    const string m20 = getafter("M20", result);
    if (m20 != to_string(tensors[i].second.mat_[2][0]))
    {
      tensors[i].second.mat_[2][0] = atof(m20.c_str());
    }

    const string m21 = getafter("M21", result);
    if (m21 != to_string(tensors[i].second.mat_[2][1]))
    {
      tensors[i].second.mat_[2][1] = atof(m21.c_str());
    }

    const string m22 = getafter("M22", result);
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
}


unsigned int
ModifyConductivities::hash_tensors(vector<pair<string, Tensor > > &tens)
{
  unsigned int h = 0;
  for (unsigned int i = 0; i < tens.size(); i++)
  {
    const char *s = tens[i].first.c_str();
    for ( ; *s; ++s)
    {
      h = 5 * h + *s;
    }

    const char *d = (const char *)(tens[i].second.mat_);
    for (unsigned int i = 0; i < sizeof(double) * 9; i++)
    {
      h = 5 * h + d[i];
    }
  }
  return h;
}


void
ModifyConductivities::execute()
{
  // Read in the LatticeVol<double>, clone it.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input");
  if (!ifp) {
    postMessage("Unable to initialize " + name + "'s Input iport\n");
    return;
  }
  FieldHandle field;
  if (!(ifp->get(field) && field.get_rep()))
  {
    return;
  }

  // Get the tensors from the field.
  bool created_p = false;
  vector<pair<string, Tensor> > tensors;
  if (!field->get("conductivity_table", tensors))
  {
    remark("Using identity conductivity tensors.");
    created_p = true;
    ScalarFieldInterface *sfi = field->query_scalar_interface();
    double maxval;
    if (sfi)
    {
      double minval;
      sfi->compute_min_max(minval, maxval);
    }
    else
    {
      maxval = 1.0;
    }

    if (maxval < 1.0 || maxval > 100)
    {
      error("Invalid number of tensors to create, no property to manage.");
      return;
    }

    tensors.resize((unsigned int)(maxval + 1.5));

    vector<double> t(6);
    t[0] = t[3] = t[5] = 1;
    t[1] = t[2] = t[4] = 0;

    Tensor tn(t);
    for (unsigned int i = 0; i < tensors.size(); i++)
    {
      tensors[i] = pair<string, Tensor>("conductivity-" + to_string(i), tn);
    }
  }

  // Update the GUI if the tensors are new.
  unsigned int hash1 = hash_tensors(tensors);
  if (hash1 != last_gui_hash_)
  {
    last_gui_hash_ = hash1;
    last_field_hash_ = hash1;
    update_gui(tensors);
  }

  bool was_new_field_p = false;
  if (last_field_gen_ != field->generation)
  {
    last_field_gen_ = field->generation;
    was_new_field_p = true;
  }

  push_changes(tensors);
  bool stored_p = false;
  unsigned int hash2 = hash_tensors(tensors);
  if (created_p ||
      (hash2 != last_field_hash_) ||
      (last_field_hash_ != last_gui_hash_ && was_new_field_p))
  {
    // Edits were made.
    last_field_hash_ = hash2;

    field.detach();

    vector<pair<string, Tensor> > *mtensors =
      new vector<pair<string, Tensor> >(tensors);
    field->store("conductivity_table", *mtensors, false);

    stored_p = true;
  }

  if (stored_p || was_new_field_p)
  {
    // Forward the results.
    FieldOPort *ofp = (FieldOPort *)get_oport("Output");
    if (!ofp) {
      error("Unable to initialize " + name + "'s Output port.");
      return;
    }
    ofp->send(field);
  }
}


} // End namespace SCIRun

