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
 *  RtPDSimGui.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Deinterfacement of Computer Science
 *   University of Utah
 *   Oct 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_RtPDSimGui_h
#define SCI_RtPDSimGui_h 

#include <Core/Util/Signals.h>
#include <Core/PartsGui/PartGui.h>

namespace MIT {

using namespace SCIRun;

class RtPDSimPartGui : public PartGui {
public:
  Signal1<float> df;

  void set_df( float );

public:
  RtPDSimPartGui( const string &name, const string &script ="RtPDSimPartGui"); 
  virtual ~RtPDSimPartGui();

  void attach( PartInterface *);
  virtual void tcl_command( TCLArgs &, void *);
};

} // namespace SCIRun

#endif // SCI_RtPDSimGui_h

