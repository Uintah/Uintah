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
 *  FieldIEPlugin:  Data structure needed to make a SCIRun FieldIE Plugin
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   May 2004
 *
 *  Copyright (C) 2004 SCI Institute
 */

#ifndef SCI_project_FieldIEPlugin_h
#define SCI_project_FieldIEPlugin_h 1

#include <Core/Util/Assert.h>
#include <Core/share/share.h>
#include <Core/Datatypes/Field.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;
using std::map;

//----------------------------------------------------------------------
class SCICORESHARE FieldIEPlugin {
public:
  const string pluginname;

  const string fileextension;
  const string filemagic;

  FieldHandle (*filereader)(const char *filename);
  void (*filewriter)(FieldHandle f, const char *filename);

  FieldIEPlugin(const string &name,
		const string &fileextension,
		const string &filemagic,
		FieldHandle (*fieldreader)(const char *filename) = 0,
		void (*fieldwriter)(FieldHandle f,
				    const char *filename) = 0);

  ~FieldIEPlugin();

  bool operator==(const FieldIEPlugin &other) const;

  static map<string, FieldIEPlugin *> *table;
};


} // End namespace SCIRun

#endif
