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
 *  SamplerGui.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Deinterfacement of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_SamplerGui_h
#define SCI_SamplerGui_h 

#include <Core/PartsGui/PartGui.h>
#include <Core/Util/Signals.h>

namespace MIT {

using namespace SCIRun;

class SamplerGui : public PartGui {
public:
  // Signals
  Signal1<int> num_iter;
  Signal1<int> subsample;
  Signal1<double> kappa;
  Signal1<int> nparms;
  Signal1<int> go;
  Signal1<vector <double> *> theta;
  Signal1<vector <vector <double> > *> sigma;

  // Slots
  void set_iter( int );
  void done();
  void set_kappa(double );
  void set_nparms( int );
  void set_theta( vector<double> *);
  void set_sigma( vector<vector<double> > *);
  
public:
  SamplerGui( const string &name, const string &script = "SamplerGui"); 
  virtual ~SamplerGui();
  
  void attach( PartInterface *) {}
  virtual void tcl_command( TCLArgs &, void *);
};

} // namespace MIT

#endif // SCI_SamplerGui_h

