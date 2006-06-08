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
 *  SamplerInterface.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Deinterfacement of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Interface_h
#define SCI_Interface_h 

#include <Core/Parts/PartInterface.h>

namespace MIT {

using namespace SCIRun;

class Sampler;

class SamplerInterface : public PartInterface {
protected:
  Sampler *sampler_;

  int num_iterations_;
  int current_iter_;
  int subsample_;
  double kappa_;
  
public:
  SamplerInterface( Sampler *, PartInterface *parent );
  virtual ~SamplerInterface();
  
  // get values
  int iterations() { return num_iterations_; }
  int subsample() { return subsample_; }
  double kappa() { return kappa_; }
  
  // set values
  void num_iterations( int i ) { num_iterations_ = i; } 
  void current_iter( int i ) { current_iter_ = i; current_iter_changed(i); }
  void go( int );
  void subsample( int i ) { subsample_ = i ; }
  void kappa( double k ) { if ( kappa_ != k ) {kappa_ = k; kappa_changed(k); }}
  void theta( vector<double> *t );
  void sigma( vector<vector<double> > *s);
  // Signals
  Signal1<int> current_iter_changed;
  Signal1<double> kappa_changed;
  Signal done;
};

} // namespace MIT

#endif // SCI_Interface_h

