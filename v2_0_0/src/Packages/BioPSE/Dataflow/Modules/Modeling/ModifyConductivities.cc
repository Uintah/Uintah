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
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Geometry/Tensor.h>
#include <sci_hash_map.h>
#include <iostream>


namespace SCIRun {

class ModifyConductivities : public Module
{
private:
  GuiInt              gui_num_entries_;
  vector<GuiString *> gui_names_;
  vector<GuiString *> gui_sizes_;
  vector<GuiString *> gui_m00_;
  vector<GuiString *> gui_m01_;
  vector<GuiString *> gui_m02_;
  vector<GuiString *> gui_m10_;
  vector<GuiString *> gui_m11_;
  vector<GuiString *> gui_m12_;
  vector<GuiString *> gui_m20_;
  vector<GuiString *> gui_m21_;
  vector<GuiString *> gui_m22_;
  vector<pair<string, Tensor> > last_field_tensors_;
  vector<pair<string, Tensor> > last_gui_tensors_;
  int                           last_field_generation_;
  bool                          reset_gui_;

  void resize_gui(int num);
  void update_to_gui(vector<pair<string, Tensor> > &tensors);
  void update_from_gui(vector<pair<string, Tensor> > &tensors);
  bool different_tensors(const vector<pair<string, Tensor> > &a,
			 const vector<pair<string, Tensor> > &b);

public:
  ModifyConductivities(GuiContext *context);
  virtual ~ModifyConductivities();

  virtual void execute();
  virtual void tcl_command(GuiArgs &args, void *);
};


DECLARE_MAKER(ModifyConductivities)


ModifyConductivities::ModifyConductivities(GuiContext *context)
  : Module("ModifyConductivities", context, Filter, "Modeling", "BioPSE"),
    gui_num_entries_(context->subVar("num-entries")),
    last_field_generation_(0),
    reset_gui_(false)
{
  resize_gui(0);
}



ModifyConductivities::~ModifyConductivities()
{
}



void
ModifyConductivities::resize_gui(int num)
{
  gui_num_entries_.set(num);
  unsigned int i;
  // Expand the gui elements.
  for (i = gui_names_.size(); i < (unsigned int)gui_num_entries_.get(); i++)
  {
    ostringstream oss;
    oss << i;
    gui_names_.push_back(new GuiString(ctx->subVar("names-" + oss.str())));
    gui_sizes_.push_back(new GuiString(ctx->subVar("sizes-" + oss.str())));
    gui_m00_.push_back(new GuiString(ctx->subVar("m00-" + oss.str())));
    gui_m01_.push_back(new GuiString(ctx->subVar("m01-" + oss.str())));
    gui_m02_.push_back(new GuiString(ctx->subVar("m02-" + oss.str())));
    gui_m10_.push_back(new GuiString(ctx->subVar("m10-" + oss.str())));
    gui_m11_.push_back(new GuiString(ctx->subVar("m11-" + oss.str())));
    gui_m12_.push_back(new GuiString(ctx->subVar("m12-" + oss.str())));
    gui_m20_.push_back(new GuiString(ctx->subVar("m20-" + oss.str())));
    gui_m21_.push_back(new GuiString(ctx->subVar("m21-" + oss.str())));
    gui_m22_.push_back(new GuiString(ctx->subVar("m22-" + oss.str())));
  }
}


void
ModifyConductivities::update_to_gui(vector<pair<string, Tensor> > &tensors)
{
  // Update GUI
  resize_gui(tensors.size());
  for (unsigned int i = 0; i <tensors.size(); i++)
  {
    gui_names_[i]->set(tensors[i].first);
    gui_sizes_[i]->set("1.0");

    gui_m00_[i]->set(to_string(tensors[i].second.mat_[0][0]));
    gui_m01_[i]->set(to_string(tensors[i].second.mat_[0][1]));
    gui_m02_[i]->set(to_string(tensors[i].second.mat_[0][2]));

    gui_m10_[i]->set(to_string(tensors[i].second.mat_[1][0]));
    gui_m11_[i]->set(to_string(tensors[i].second.mat_[1][1]));
    gui_m12_[i]->set(to_string(tensors[i].second.mat_[1][2]));

    gui_m20_[i]->set(to_string(tensors[i].second.mat_[2][0]));
    gui_m21_[i]->set(to_string(tensors[i].second.mat_[2][1]));
    gui_m22_[i]->set(to_string(tensors[i].second.mat_[2][2]));

    string result;
    gui->eval(id + " create_entries", result);
  }
}



void
ModifyConductivities::update_from_gui(vector<pair<string, Tensor> > &tensors)
{
  gui_num_entries_.reset();
  resize_gui(gui_num_entries_.get());
  tensors.resize(gui_names_.size());
  for (unsigned int i = 0; i <tensors.size(); i++)
  {
    tensors[i].first = gui_names_[i]->get();
    tensors[i].second.mat_[0][0] = atof(gui_m00_[i]->get().c_str());
    tensors[i].second.mat_[0][1] = atof(gui_m01_[i]->get().c_str());
    tensors[i].second.mat_[0][2] = atof(gui_m02_[i]->get().c_str());
    tensors[i].second.mat_[1][0] = atof(gui_m10_[i]->get().c_str());
    tensors[i].second.mat_[1][1] = atof(gui_m11_[i]->get().c_str());
    tensors[i].second.mat_[1][2] = atof(gui_m12_[i]->get().c_str());
    tensors[i].second.mat_[2][0] = atof(gui_m20_[i]->get().c_str());
    tensors[i].second.mat_[2][1] = atof(gui_m21_[i]->get().c_str());
    tensors[i].second.mat_[2][2] = atof(gui_m22_[i]->get().c_str());
    const double scale = atof(gui_sizes_[i]->get().c_str());
    if (scale != 1.0 && scale != 0.0)
    {
      tensors[i].second = tensors[i].second * scale;
    }
  }
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
    if (last_field_generation_ == 0)
    {
      update_from_gui(last_gui_tensors_);
    }
    last_field_generation_ = field->generation;
    new_field_p = true;
  }

  // Get the tensors from the field.
  vector<pair<string, Tensor> > field_tensors;
  bool created_p = false;

  MatrixIPort *imp = (MatrixIPort *)get_iport("Tensor Matrix");
  if (!imp) {
    error("Unable ti initialize iport 'Tensor Matrix'.");
    return;
  }
  MatrixHandle imatrix;
  if (imp->get(imatrix) && imatrix.get_rep())
  {
    ScalarFieldInterfaceHandle sfi = field->query_scalar_interface(this);
    double minval, maxval;
    sfi->compute_min_max(minval, maxval);
    if (imatrix->nrows() > maxval && imatrix->ncols() == 9)
    {
      for (int i = 0; i < imatrix->nrows(); i++)
      {
	Tensor t;
	t.mat_[0][0] = imatrix->get(i, 0);
	t.mat_[0][1] = imatrix->get(i, 1);
	t.mat_[0][2] = imatrix->get(i, 2);

	t.mat_[1][0] = imatrix->get(i, 3);
	t.mat_[1][1] = imatrix->get(i, 4);
	t.mat_[1][2] = imatrix->get(i, 5);

	t.mat_[2][0] = imatrix->get(i, 6);
	t.mat_[2][1] = imatrix->get(i, 7);
	t.mat_[2][2] = imatrix->get(i, 8);
	const string s = "matrix-row-" + to_string(i+1);
	field_tensors.push_back(pair<string, Tensor>(s, t));
      }
      created_p = true;
    }
    else if (imatrix->nrows() == 9 && imatrix->ncols() > maxval)
    {
      for (int i = 0; i < imatrix->ncols(); i++)
      {
	Tensor t;
	t.mat_[0][0] = imatrix->get(0, i);
	t.mat_[0][1] = imatrix->get(1, i);
	t.mat_[0][2] = imatrix->get(2, i);

	t.mat_[1][0] = imatrix->get(3, i);
	t.mat_[1][1] = imatrix->get(4, i);
	t.mat_[1][2] = imatrix->get(5, i);

	t.mat_[2][0] = imatrix->get(6, i);
	t.mat_[2][1] = imatrix->get(7, i);
	t.mat_[2][2] = imatrix->get(8, i);
	const string s = "matrix-column-" + to_string(i+1);
	field_tensors.push_back(pair<string, Tensor>(s, t));
      }
      created_p = true;
    }
    else
    {
      warning("Bad input matrix.");
      warning("It should be of size Nx9 or 9xN where N is greater than the field data range.");
    }
  }

  if (!(created_p || field->get_property("conductivity_table", field_tensors)))
  {
    created_p = true;
    ScalarFieldInterfaceHandle sfi = field->query_scalar_interface(this);
    double minval, maxval;
    if (sfi.get_rep())
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
    else
    {
      remark("No tensors found, using default identity tensors.");
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

  if (reset_gui_)
  {
    update_to_gui(last_gui_tensors_);
    reset_gui_ = false;
  }

  vector<pair<string, Tensor> > gui_tensors;
  gui_tensors.resize(last_gui_tensors_.size());
  update_from_gui(gui_tensors);

#if 0  
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
#endif

  bool changed_table_p = false;
  if (different_tensors(gui_tensors, last_gui_tensors_) ||
      different_tensors(gui_tensors, field_tensors) ||
      created_p)
  {
    field.detach();

    field->set_property("conductivity_table", gui_tensors, false);
    changed_table_p = true;
  }

  DenseMatrix *omatrix = scinew DenseMatrix(gui_tensors.size(), 9);
  for (unsigned int j = 0; j < gui_tensors.size(); j++)
  {
    omatrix->put(j, 0, gui_tensors[j].second.mat_[0][0]);
    omatrix->put(j, 1, gui_tensors[j].second.mat_[0][1]);
    omatrix->put(j, 2, gui_tensors[j].second.mat_[0][2]);
				  
    omatrix->put(j, 3, gui_tensors[j].second.mat_[1][0]);
    omatrix->put(j, 4, gui_tensors[j].second.mat_[1][1]);
    omatrix->put(j, 5, gui_tensors[j].second.mat_[1][2]);
				  
    omatrix->put(j, 6, gui_tensors[j].second.mat_[2][0]);
    omatrix->put(j, 7, gui_tensors[j].second.mat_[2][1]);
    omatrix->put(j, 8, gui_tensors[j].second.mat_[2][2]);
  }

  // Forward the matrix results.
  MatrixOPort *omp = (MatrixOPort *)get_oport("Output Matrix");
  if (!omp) {
    error("Unable to initialize " + name + "'s Output Matrix port.");
    return;
  }
  omp->send(omatrix);

  // Forward the field results.
  FieldOPort *ofp = (FieldOPort *)get_oport("Output Field");
  if (!ofp) {
    error("Unable to initialize " + name + "'s Output Field port.");
    return;
  }
  ofp->send(field);
}


void
ModifyConductivities::tcl_command(GuiArgs &args, void *extra)
{
  if (args.count() == 2 && args[1] == "reset_gui")
  {
    reset_gui_ = true;
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

