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
 *  BuildMisfitField.cc: Build the lead field matrix through reciprocity
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Math/Mat.h>

#include <iostream>
#include <stdio.h>
#include <math.h>

// Take in a ColumnMatrix, c,  and a LeadField, L
//   Scan c through all of the elements (three column groupings) in L
//   For each "element", find the misfit for the optimal moment weights
//   Send out a ColumnMatrix of the misfits

namespace BioPSE {

using std::cerr;
using std::endl;
using std::pair;

using namespace SCIRun;


class BuildMisfitField : public Module {    
  int last_leadfield_generation_;
  int last_measurements_generation_;
  MatrixHandle last_misfit_;
  string last_metric_;
  double last_pvalue_;
  GuiString metric_;
  GuiString pvalue_;
public:
  BuildMisfitField(GuiContext *context);
  double compute_misfit(double *b, double *bprime, int nr);
  virtual ~BuildMisfitField();
  virtual void execute();
};


DECLARE_MAKER(BuildMisfitField)


//---------------------------------------------------------------
BuildMisfitField::BuildMisfitField(GuiContext *context)
  : Module("BuildMisfitField", context, Filter, "LeadField", "BioPSE"), 
  last_leadfield_generation_(-1),
  last_measurements_generation_(-1),
  last_misfit_(0), last_metric_(""), last_pvalue_(1),
  metric_(context->subVar("metric")), pvalue_(context->subVar("pvalue"))
{
}

BuildMisfitField::~BuildMisfitField(){}

double BuildMisfitField::compute_misfit(double *b, double *bprime, int nr) {
  double avg1=0, avg2=0;
  int r;
  for (r=0; r<nr; r++) {
    avg1+=b[r];
    avg2+=bprime[r];
  }
  avg1/=nr;
  avg2/=nr;
  
  double ccNum=0;
  double ccDenom1=0;
  double ccDenom2=0;
  double rms=0;
  
  for (r=0; r<nr; r++) {
    double shift1=(b[r]-avg1);
    double shift2=(bprime[r]-avg2);
    ccNum+=shift1*shift2;
    ccDenom1+=shift1*shift1;
    ccDenom2+=shift2*shift2;
    double tmp=fabs(shift1-shift2);
    if (last_pvalue_==1) rms+=tmp;
    else if (last_pvalue_==2) rms+=tmp*tmp; 
    else rms+=pow(tmp,last_pvalue_);
  }

  rms = pow(rms/nr,1/last_pvalue_);
  double ccDenom=Sqrt(ccDenom1*ccDenom2);
  double cc=Min(ccNum/ccDenom, 1000000.);
  double ccInv=Min(1.0-fabs(ccNum/ccDenom), 1000000.);
  double rmsRel=Min(rms/ccDenom1, 1000000.);
  if (last_metric_ == "rms") {
    return rms;
  } else if (last_metric_ == "rmsRel") {
    return rmsRel;
  } else if (last_metric_ == "invCC") {
    return ccInv;
  } else if (last_metric_ == "CC") {
    return cc;
  } else {
    cerr << "BuildMisfitField: error - unknown metric "<<last_metric_<<endl;
    return 0;
  }
}

void BuildMisfitField::execute() {
  MatrixIPort *leadfield_iport = 
    (MatrixIPort *)get_iport("Leadfield (nelecs x nelemsx3)");
  MatrixIPort *measurements_iport = 
    (MatrixIPort *)get_iport("Measurement Vector");
  MatrixOPort *misfit_oport = 
    (MatrixOPort *)get_oport("Misfit Vector");

  if (!leadfield_iport) {
    error("Unable to initialize iport 'Leadfield (nelecs x nelemsx3)'.");
    return;
  }
  if (!measurements_iport) {
    error("Unable to initialize iport 'Measurement Vector'.");
    return;
  }
  if (!misfit_oport) {
    error("Unable to initialize oport 'Misfit Vector'.");
    return;
  }

  MatrixHandle leadfield_in;
  if (!leadfield_iport->get(leadfield_in) || !leadfield_in.get_rep()) {
    cerr << "BuildMisfitField -- couldn't get leadfield.  Returning.\n";
    return;
  }
  DenseMatrix *dm = dynamic_cast<DenseMatrix*>(leadfield_in.get_rep());
  if (!dm) {
    cerr << "BuildMisfitField -- error, leadfield wasn't a DenseMatrix.\n";
    return;
  }

  MatrixHandle measurements_in;
  if (!measurements_iport->get(measurements_in) || !measurements_in.get_rep()) {
    cerr << "BuildMisfitField -- couldn't get measurement vector.  Returning.\n";
    return;
  }
  ColumnMatrix *cm = dynamic_cast<ColumnMatrix*>(measurements_in.get_rep());
  if (!cm) {
    cerr << "BuildMisfitField -- error, measurement vectors wasn't a ColumnMatrix.\n";
    return;
  }
  if (cm->nrows() != dm->nrows()) {
    cerr << "BuildMisfitField -- error, leadfield ("<<dm->nrows()<<") and measurements ("<<cm->nrows()<<") have different numbers of rows.\n";
    return;
  }
  int nr = cm->nrows();
  int nelems = dm->ncols()/3;

  string metric=metric_.get();
  double pvalue;
  string_to_double(pvalue_.get(), pvalue);
 
  if (last_leadfield_generation_ == dm->generation &&
      last_measurements_generation_ == cm->generation &&
      last_metric_ == metric && last_pvalue_ == pvalue) {
    cerr << "BuildMisfitField -- sending same data again.\n";
    misfit_oport->send(last_misfit_);
    return;
  }

  last_leadfield_generation_ = dm->generation;
  last_measurements_generation_ = cm->generation;
  last_metric_ = metric;
  last_pvalue_ = pvalue;

  double best_val;
  int best_idx;

  ColumnMatrix *misfit=new ColumnMatrix(nelems);
  last_misfit_ = misfit;

  double *b = &((*cm)[0]);
  double *bprime = new double[nr];
  double *x = new double[3];
  double *A[3];
  int r, c;
  for (c=0; c<3; c++)
    A[c] = new double[nr];

  for (int i=0; i<nelems; i++) {
    if (i%100 == 0) update_progress(i*1./nelems);

    for (r=0; r<nr; r++)
      for (c=0; c<3; c++)
	A[c][r]=(*dm)[r][i*3+c];
	
    min_norm_least_sq_3(A, b, x, bprime, nr);

    (*misfit)[i]=compute_misfit(b, bprime, nr);

    if (i==0 || (*misfit)[i]<best_val) {
      best_val = (*misfit)[i];
      best_idx = i;
    }
  }

  delete[] bprime;
  delete[] x;
  for (c=0; c<3; c++) {
    delete[] A[c];
  }

  misfit_oport->send(last_misfit_);
  cerr << "Best misfit was "<<best_val<<", which was cell "<<best_idx<<"\n";
}
} // End namespace BioPSE
