
/*
 *  MatrixSelectVector: Select a row or column of a matrix
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class MatrixSelectVector : public Module {
  GuiString row_or_col_;
  GuiDouble selectable_min_;
  GuiDouble selectable_max_;
  GuiInt    selectable_inc_;
  GuiString selectable_units_;
  GuiInt    range_min_;
  GuiInt    range_max_;

public:
  MatrixSelectVector(const string& id);
  virtual ~MatrixSelectVector();
  virtual void execute();
};


extern "C" Module* make_MatrixSelectVector(const string& id)
{
  return new MatrixSelectVector(id);
}


MatrixSelectVector::MatrixSelectVector(const string& id)
  : Module("MatrixSelectVector", id, Filter,"Math", "SCIRun"),
    row_or_col_("row_or_col", id, this),
    selectable_min_("selectable_min", id, this),
    selectable_max_("selectable_max", id, this),
    selectable_inc_("selectable_inc", id, this),
    selectable_units_("selectable_units", id, this),
    range_min_("range_min", id, this),
    range_max_("range_max", id, this)
{
}


MatrixSelectVector::~MatrixSelectVector()
{
}


void
MatrixSelectVector::execute()
{
  update_state(NeedData);

  MatrixIPort *imat = (MatrixIPort *)get_iport("Matrix");
  if (!imat) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  MatrixHandle mh;
  if (!(imat->get(mh) && mh.get_rep()))
  {
    warning("Empty input matrix.");
    return;
  }
  
  MatrixOPort *ovec = (MatrixOPort *)get_oport("Vector");
  if (!ovec) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  MatrixOPort *osel = (MatrixOPort *)get_oport("Selected Index");
  if (!osel) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  
  update_state(JustStarted);
  
  const bool use_row = (row_or_col_.get() == "row");
  bool changed_p = false;
  PropertyManager *mh_prop = mh.get_rep();
  if (use_row)
  {
    string units;
    if (!mh_prop->get("row_units", units))
    {
      units = "Units";
    }
    if (units != selectable_units_.get())
    {
      selectable_units_.set(units);
      changed_p = true;
    }

    double minlabel;
    if (!mh_prop->get("row_min", minlabel))
    {
      minlabel = 0.0;
    }
    if (minlabel != selectable_min_.get())
    {
      selectable_min_.set(minlabel);
      changed_p = true;
    }

    double maxlabel;
    if (!mh_prop->get("row_max", maxlabel))
    {
      maxlabel = mh->nrows() - 1.0;
    }
    if (maxlabel != selectable_max_.get())
    {
      selectable_max_.set(maxlabel);
      changed_p = true;
    }

    int increments = mh->nrows();
    if (increments != selectable_inc_.get())
    {
      selectable_inc_.set(increments);
      changed_p = true;
    }
  }
  else
  {
    string units;
    if (!mh_prop->get("col_units", units))
    {
      units = "Units";
    }
    if (units != selectable_units_.get())
    {
      selectable_units_.set(units);
      changed_p = true;
    }

    double minlabel;
    if (!mh_prop->get("col_min", minlabel))
    {
      minlabel = 0.0;
    }
    if (minlabel != selectable_min_.get())
    {
      selectable_min_.set(minlabel);
      changed_p = true;
    }

    double maxlabel;
    if (!mh_prop->get("col_max", maxlabel))
    {
      maxlabel = mh->ncols() - 1.0;
    }
    if (maxlabel != selectable_max_.get())
    {
      selectable_max_.set(maxlabel);
      changed_p = true;
    }

    int increments = mh->ncols();
    if (increments != selectable_inc_.get())
    {
      selectable_inc_.set(increments);
      changed_p = true;
    }
  }

  if (changed_p)
  {
    std::ostringstream str;
    str << id << " update";
    TCL::execute(str.str().c_str());
  }
  
  reset_vars();

#if 1
  // Specialized matrix multiply, with Weight Vector given as a sparse
  // matrix.  It's not clear what this has to do with MatrixSelectVector.
  MatrixIPort *ivec = (MatrixIPort *)get_iport("Weight Vector");
  if (!ivec) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  MatrixHandle weightsH;
  if (ivec->get(weightsH) && weightsH.get_rep())
  {
    ColumnMatrix *w = dynamic_cast<ColumnMatrix*>(weightsH.get_rep());
    ColumnMatrix *cm;
    if (use_row) 
    {
      cm = scinew ColumnMatrix(mh->ncols());
      cm->zero();
      double *data = cm->get_data();
      for (int i = 0; i<w->nrows()/2; i++)
      {
	const int idx = (int)((*w)[i*2]);
	double wt = (*w)[i*2+1];
	for (int j = 0; j<mh->ncols(); j++)
	{
	  data[j]+=mh->get(idx, j)*wt;
	}
      }
    }
    else
    {
      cm = scinew ColumnMatrix(mh->nrows());
      cm->zero();
      double *data = cm->get_data();
      for (int i = 0; i<w->nrows()/2; i++)
      {
	const int idx = (int)((*w)[i*2]);
	double wt = (*w)[i*2+1];
	for (int j = 0; j<mh->nrows(); j++)
	{
	  data[j]+=mh->get(j, idx)*wt;
	}
      }
    }
    ovec->send(MatrixHandle(cm));
    return;
  }
#endif

  const int start = range_min_.get();
  const int end = range_max_.get();
  const int inc = (start>end)?-1:1;
  int which;
  for (which = start; ; which += inc)
  {
    ColumnMatrix *cm;
    if (use_row)
    {
      cm = scinew ColumnMatrix(mh->ncols());
      double *data = cm->get_data();
      for (int c = 0; c<mh->ncols(); c++)
      {
	data[c] = mh->get(which, c);
      }
    }
    else
    {
      cm = scinew ColumnMatrix(mh->nrows());
      double *data = cm->get_data();
      for (int r = 0; r<mh->nrows(); r++)
      {
	data[r] = mh->get(r, which);
      }
    }	    

    // Attempt to copy no-transient properties.
    // TODO: update min/max to be the current value:  min + (max - min) * inc
    //PropertyManager *cmp = cm;
    //*cmp = *mh_prop;

    ColumnMatrix *selected = scinew ColumnMatrix(1);
    selected->put(0, 0, (double)which);

    if (which == end)
    {
      ovec->send(MatrixHandle(cm));
      osel->send(MatrixHandle(selected));
      break;
    }
    else
    {
      ovec->send_intermediate(MatrixHandle(cm));
      osel->send_intermediate(MatrixHandle(selected));
    }
  }
}


} // End namespace SCIRun
