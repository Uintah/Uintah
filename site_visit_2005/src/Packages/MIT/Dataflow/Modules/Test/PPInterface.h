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
 *  PPInterface.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Deinterfacement of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef MIT_PPInterface_h
#define MIT_PPInterface_h 

#include <Core/Parts/PartInterface.h>

namespace MIT {

using namespace SCIRun;

class PPexample;

class PPInterface : public PartInterface {
protected:
  PPexample *example_;

  int burning_;
  int monitor_;
  int thin_;

public:
  PPInterface( Part *, PartInterface *parent );
  virtual ~PPInterface();

  // get values
  int burning() { return burning_; }
  int monitor() { return monitor_; }
  int thin() { return thin_; }

  // set values
  void burning( int b ) { burning_ = b; }
  void monitor( int m ) { monitor_ = m; }
  void thin( int t) { thin_ = t; }
  void go();
};

} // namespace MIT

#endif // SCI_Interface_h

