
/*
 *  BuildNoise: Add BuildNoise to a matrix
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
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Math/Expon.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Math/MusilRNG.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class BuildNoise : public Module {
  MusilRNG musil;
  GuiDouble snr_;
public:
  BuildNoise(const string& id);
  virtual ~BuildNoise();
  virtual void execute();
};

extern "C" Module* make_BuildNoise(const string& id)
{
    return new BuildNoise(id);
}

BuildNoise::BuildNoise(const string& id)
: Module("BuildNoise", id, Filter,"Math", "SCIRun"),
  snr_("snr", id, this)
{
}

BuildNoise::~BuildNoise()
{
}

void BuildNoise::execute() {
  MatrixIPort *isignal = (MatrixIPort *)get_iport("Signal");
  MatrixOPort *onoise = (MatrixOPort *)get_oport("Noise");

  if (!isignal) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!onoise) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  
  update_state(NeedData);
  MatrixHandle matH;
  if (!isignal->get(matH))
    return;
  if (!matH.get_rep()) {
    warning("Empty input matrix.");
    return;
  }

  // gotta make sure we have a Dense or Column matrix...
  // ...if it's Sparse, change it to Dense

  SparseRowMatrix *sm = dynamic_cast<SparseRowMatrix *>(matH.get_rep());
  if (sm) matH = matH->dense();
  else matH.detach();

  double mean, power, sigma;
  mean=power=sigma=0;
  int r, c;
  int nr = matH->nrows();
  int nc = matH->ncols();
  double curr;
  double snr = snr_.get();
  cerr << "signal-to-noise ratio = "<<snr<<"\n";
  for (r=0; r<nr; r++)
    for (c=0; c<nc; c++) {
      curr = matH->get(r, c);
      mean += curr;
    }
  mean /= nr*nc;
  for (r=0; r<nr; r++)
    for (c=0; c<nc; c++) {
      curr = matH->get(r, c);
      power += (curr-mean)*(curr-mean);
    }
  power /= nr*nc;
  
  sigma = sqrt(power)/(snr*Sqrt(2*M_PI));
  cerr << "sigma = "<<sigma<<"\n";
  
  for (r=0; r<nr; r++) 
    for (c=0; c<nc; c++) {
      // gaussian distribution about this percentage
      double rnd = 2.0 * musil() - 1.0;
      double perturb = rnd * sigma * sqrt((-2.0 * log(rnd*rnd)) / (rnd*rnd));
      (*(matH.get_rep()))[r][c] = perturb;
    }
  onoise->send(matH);
}
} // End namespace SCIRun
