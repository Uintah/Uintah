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
 *  GraphPart.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_GraphPart_h
#define SCI_GraphPart_h 

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/Util/Signals.h>
#include <Core/Parts/Part.h>
#include <Core/Parts/PartInterface.h>
#include <Core/2d/DrawObj.h>

namespace SCIRun {
  
class SCICORESHARE GraphPart : public Part, public PartInterface  {
private:
  vector< vector<double> > data_;

public:
  GraphPart( PartInterface *parent = 0, const string &name="GraphPart",
	     bool=true);
  virtual ~GraphPart();

  void set_num_lines( int );

#ifdef CHRIS
  void add_values( unsigned , const vector<double> &);

  Signal1< const vector<DrawObj*> & > reset;
  Signal2< unsigned, const vector<double> & > new_values;
#else
  void add_values( const vector<double> &);

  Signal1< int > reset;
  Signal1< const vector<double> & > new_values;
#endif
};

} // namespace SCIRun

#endif // SCI_GraphPart_h
