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
  Moduleions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  ProgressReporter.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2003
 *
 *  Copyright (C) 2003 SCI Group
 */


#ifndef SCIRun_Core_Util_ProgressReporter_h
#define SCIRun_Core_Util_ProgressReporter_h

#include <Core/share/share.h>
#include <string>
#include <iostream>

namespace SCIRun {

  //using namespace std;

class SCICORESHARE ProgressReporter 
{
public:
  typedef enum {Starting, Compiling, CompilationDone, Done } ProgressState;
public:
  virtual ~ProgressReporter();

  virtual void error(const std::string& msg)      { std::cerr << "Error: " << msg << std::endl; }
  virtual void warning(const std::string& msg)    { std::cerr << "Warning: " << msg << std::endl; }
  virtual void remark(const std::string& msg)     { std::cerr << "Remark: " << msg << std::endl; }
  virtual void postMessage(const std::string&msg) { std::cerr << "Msg: " << msg << std::endl; }

  virtual std::ostream &msgStream() { return std::cerr; }
  virtual void msgStream_flush() {}


  virtual void report_progress( ProgressState ) {}
};


} // Namespace SCIRun

#endif
