/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Datatypes/Field.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;
using std::map;


//----------------------------------------------------------------------
class FieldIEPlugin {
public:
  const string pluginname;

  const string fileextension;
  const string filemagic;

  FieldHandle (*filereader)(ProgressReporter *pr, const char *filename);
  bool (*filewriter)(ProgressReporter *pr,
		     FieldHandle f, const char *filename);

  FieldIEPlugin(const string &name,
		const string &fileextension,
		const string &filemagic,
		FieldHandle (*freader)(ProgressReporter *pr,
				       const char *filename) = 0,
		bool (*fwriter)(ProgressReporter *pr, FieldHandle f,
				const char *filename) = 0);

  ~FieldIEPlugin();

  bool operator==(const FieldIEPlugin &other) const;
};



class FieldIEPluginManager {
public:
  void get_importer_list(vector<string> &results);
  void get_exporter_list(vector<string> &results);
  FieldIEPlugin *get_plugin(const string &name);
};


} // End namespace SCIRun

#endif
