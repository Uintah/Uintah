/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

const unsigned int GUI_LIMIT = 250; // Number of entries before we
                                    // require force flag to generate
                                    // gui table.

class ModifyConductivities : public Module
{
private:
  GuiInt			gui_num_entries_;
  GuiInt			gui_use_gui_values_;
  GuiInt                        gui_force_gui_update_;
  vector<GuiString *>		gui_names_;
  vector<GuiDouble *>		gui_sizes_;
  vector<GuiDouble *>		gui_m00_;
  vector<GuiDouble *>		gui_m01_;
  vector<GuiDouble *>		gui_m02_;
  vector<GuiDouble *>		gui_m10_;
  vector<GuiDouble *>		gui_m11_;
  vector<GuiDouble *>		gui_m12_;
  vector<GuiDouble *>		gui_m20_;
  vector<GuiDouble *>		gui_m21_;
  vector<GuiDouble *>		gui_m22_;
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
    gui_use_gui_values_(context->subVar("use-gui-values")),
    gui_force_gui_update_(context->subVar("force-gui-update")),
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
    const string num = to_string(i);
    gui_names_.push_back(new GuiString(ctx->subVar("names-" + num)));
    gui_sizes_.push_back(new GuiDouble(ctx->subVar("sizes-" + num)));
    gui_m00_.push_back(new GuiDouble(ctx->subVar("m00-" + num)));
    gui_m01_.push_back(new GuiDouble(ctx->subVar("m01-" + num)));
    gui_m02_.push_back(new GuiDouble(ctx->subVar("m02-" + num)));
    gui_m10_.push_back(new GuiDouble(ctx->subVar("m10-" + num)));
    gui_m11_.push_back(new GuiDouble(ctx->subVar("m11-" + num)));
    gui_m12_.push_back(new GuiDouble(ctx->subVar("m12-" + num)));
    gui_m20_.push_back(new GuiDouble(ctx->subVar("m20-" + num)));
    gui_m21_.push_back(new GuiDouble(ctx->subVar("m21-" + num)));
    gui_m22_.push_back(new GuiDouble(ctx->subVar("m22-" + num)));
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
    gui_sizes_[i]->set(1.0);

    gui_m00_[i]->set(tensors[i].second.mat_[0][0]);
    gui_m01_[i]->set(tensors[i].second.mat_[0][1]);
    gui_m02_[i]->set(tensors[i].second.mat_[0][2]);

    gui_m10_[i]->set(tensors[i].second.mat_[1][0]);
    gui_m11_[i]->set(tensors[i].second.mat_[1][1]);
    gui_m12_[i]->set(tensors[i].second.mat_[1][2]);

    gui_m20_[i]->set(tensors[i].second.mat_[2][0]);
    gui_m21_[i]->set(tensors[i].second.mat_[2][1]);
    gui_m22_[i]->set(tensors[i].second.mat_[2][2]);
  }
  gui->execute(id + " create_entries");
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
    tensors[i].second.mat_[0][0] = gui_m00_[i]->get();
    tensors[i].second.mat_[0][1] = gui_m01_[i]->get();
    tensors[i].second.mat_[0][2] = gui_m02_[i]->get();
    tensors[i].second.mat_[1][0] = gui_m10_[i]->get();
    tensors[i].second.mat_[1][1] = gui_m11_[i]->get();
    tensors[i].second.mat_[1][2] = gui_m12_[i]->get();
    tensors[i].second.mat_[2][0] = gui_m20_[i]->get();
    tensors[i].second.mat_[2][1] = gui_m21_[i]->get();
    tensors[i].second.mat_[2][2] = gui_m22_[i]->get();
    const double scale = gui_sizes_[i]->get();
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
  MatrixHandle imatrix;
  vector<string> tensor_names;

  if (imp->get(imatrix) && imatrix.get_rep())
  {
    imatrix->get_property("tensor-names", tensor_names);
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
	if ((unsigned int)i < tensor_names.size()) {
	  field_tensors.push_back(pair<string, Tensor>(tensor_names[i], t));
	} else {
	  const string s = "matrix-row-" + to_string(i+1);
	  field_tensors.push_back(pair<string, Tensor>(s, t));
	}
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
	if ((unsigned int)i < tensor_names.size()) {
	  field_tensors.push_back(pair<string, Tensor>(tensor_names[i], t));
	} else {
	  const string s = "matrix-column-" + to_string(i+1);
	  field_tensors.push_back(pair<string, Tensor>(s, t));
	}
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
  if ((!gui_use_gui_values_.get() || created_p) &&
      different_tensors(field_tensors, last_field_tensors_))
  {
    if (field_tensors.size() < GUI_LIMIT || gui_force_gui_update_.get())
    {
      update_to_gui(field_tensors);
    }
    else
    {
      vector<pair<string, Tensor> > blank;
      blank.push_back(pair<string, Tensor>("Too many entries", Tensor(0)));
      update_to_gui(blank);
    }
    last_field_tensors_ = field_tensors;
    last_gui_tensors_ = field_tensors;
  }

  if (reset_gui_)
  {
    if (last_gui_tensors_.size() < GUI_LIMIT || gui_force_gui_update_.get())
    {
      update_to_gui(last_gui_tensors_);
    }
    else
    {
      vector<pair<string, Tensor> > blank;
      blank.push_back(pair<string, Tensor>("Too many entries", Tensor(0)));
      update_to_gui(blank);
    }
    reset_gui_ = false;
  }
  
  vector<pair<string, Tensor> > gui_tensors;
  gui_tensors.resize(last_gui_tensors_.size());
  if (last_gui_tensors_.size() < GUI_LIMIT || gui_force_gui_update_.get())
  {
    update_from_gui(gui_tensors);
  }
  else
  {
    gui_tensors = last_gui_tensors_;
  }

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
  tensor_names.clear();
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
    tensor_names.push_back(gui_tensors[j].first);
  }
  omatrix->set_property("tensor-names", tensor_names, false);

  // Forward the matrix results.
  MatrixOPort *omp = (MatrixOPort *)get_oport("Output Matrix");
  omp->send(omatrix);

  // Forward the field results.
  FieldOPort *ofp = (FieldOPort *)get_oport("Output Field");
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

