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
 *  Persistent.h: Base class for persistent objects...
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   May 2004
 *
 *  Copyright (C) 2004 SCI Institute
 */

#include <Core/ImportExport/Field/FieldIEPlugin.h>
#include <Core/Containers/StringUtil.h>

#ifdef __APPLE__
  // Part of the hack to get static constructors from this library to
  // load (under Mac OSX)
#  include <Core/ImportExport/Matrix/MatrixIEPlugin.h>
#  include <Core/ImportExport/ColorMap/ColorMapIEPlugin.h>
#endif

#include <sgi_stl_warnings_off.h>
#include <map>
#include <sgi_stl_warnings_on.h>

using namespace std;
using namespace SCIRun;

#define DEBUG 0

#ifdef __APPLE__
  // We need a symbol from the matrices to force the instantiation of the static
  // constructors (under Mac OSX).
  extern MatrixHandle TextColumnMatrix_reader(ProgressReporter *pr, const char *filename);
  extern bool TextColumnMatrix_writer(ProgressReporter *pr,
                                      MatrixHandle matrix, const char *filename);
  extern ColorMapHandle TextColorMap_reader(ProgressReporter *pr, const char *filename);
  extern bool TextColorMap_writer(ProgressReporter *pr,
                                  ColorMapHandle colormap, const char *filename);
#endif

namespace SCIRun {

#ifdef _WIN32
  #define SHARE __declspec(dllimport)
#else
  #define SHARE
#endif


static map<string, FieldIEPlugin *> *field_plugin_table = 0;
extern SHARE Mutex fieldIEPluginMutex; // From Core/Util/DynamicLoader.cc

#ifdef __APPLE__
void
macImportExportForceLoad()
{
  // On non-Mac computers, the following function comes from TextPointCloudString_plugin.cc
  extern FieldHandle TextPointCloudString_reader(ProgressReporter *pr, const char *filename);
  extern bool TextPointCloudString_writer(ProgressReporter *pr,
                                          FieldHandle field, const char *filename);

  static FieldIEPlugin TextPointCloudString_plugin("TextPointCloudString",
                                                   ".pcs.txt", "",
                                                   TextPointCloudString_reader,
                                                   TextPointCloudString_writer);
  
  static MatrixIEPlugin TextColumnMatrix_plugin("TextColumnMatrix",
                                                "", "",
                                                TextColumnMatrix_reader,
                                                TextColumnMatrix_writer);

  static ColorMapIEPlugin TextColorMap_plugin("TextColorMap",
                                              "", "",
                                              TextColorMap_reader,
                                              TextColorMap_writer);
}
#endif

//----------------------------------------------------------------------
FieldIEPlugin::FieldIEPlugin(const string& pname,
			     const string& fextension,
			     const string& fmagic,
			     FieldHandle (*freader)(ProgressReporter *pr,
						    const char *filename),
			     bool (*fwriter)(ProgressReporter *pr,
					     FieldHandle f,
					     const char *filename))
  : pluginname(pname),
    fileextension(fextension),
    filemagic(fmagic),
    filereader(freader),
    filewriter(fwriter)
{
  fieldIEPluginMutex.lock();

  if (!field_plugin_table)
  {
    field_plugin_table = scinew map<string, FieldIEPlugin *>();
  }

  string tmppname = pluginname;
  int counter = 2;
  for (;;)
  {
    map<string, FieldIEPlugin *>::iterator loc = field_plugin_table->find(tmppname);
    if (loc == field_plugin_table->end())
    {
      if (tmppname != pluginname) { ((string)pluginname) = tmppname; }
      (*field_plugin_table)[pluginname] = this;
      break;
    }
    if (*(*loc).second == *this)
    {
      cerr << "WARNING: FieldIEPlugin '" << tmppname << "' duplicated.\n";
      break;
    }

    cout << "WARNING: Multiple FieldIEPlugins with '" << pluginname
	 << "' name.\n";
    tmppname = pluginname + "(" + to_string(counter) + ")";
    counter++;
  }

  fieldIEPluginMutex.unlock();
}



FieldIEPlugin::~FieldIEPlugin()
{
  if (field_plugin_table == NULL)
  {
    cerr << "WARNING: FieldIEPlugin.cc: ~FieldIEPlugin(): field_plugin_table is NULL\n";
    cerr << "         For: " << pluginname << "\n";
    return;
  }

  fieldIEPluginMutex.lock();

  map<string, FieldIEPlugin *>::iterator iter = field_plugin_table->find(pluginname);
  if (iter == field_plugin_table->end())
  {
    cerr << "WARNING: FieldIEPlugin " << pluginname << 
      " not found in database for removal.\n";
  }
  else
  {
    field_plugin_table->erase(iter);
  }

  if (field_plugin_table->size() == 0)
  {
    delete field_plugin_table;
    field_plugin_table = 0;
  }

  fieldIEPluginMutex.unlock();
}


bool
FieldIEPlugin::operator==(const FieldIEPlugin &other) const
{
  return (pluginname == other.pluginname &&
	  fileextension == other.fileextension &&
	  filemagic == other.filemagic &&
	  filereader == other.filereader &&
	  filewriter == other.filewriter);
}



void
FieldIEPluginManager::get_importer_list(vector<string> &results)
{
  if (field_plugin_table == 0) return;

  fieldIEPluginMutex.lock();
  map<string, FieldIEPlugin *>::const_iterator itr = field_plugin_table->begin();
  while (itr != field_plugin_table->end())
  {
    if ((*itr).second->filereader != NULL)
    {
      results.push_back((*itr).first);
    }
    ++itr;
  }
  fieldIEPluginMutex.unlock();
}


void
FieldIEPluginManager::get_exporter_list(vector<string> &results)
{
  if (field_plugin_table == 0) return;

  fieldIEPluginMutex.lock();
  map<string, FieldIEPlugin *>::const_iterator itr = field_plugin_table->begin();
  while (itr != field_plugin_table->end())
  {
    if ((*itr).second->filewriter != NULL)
    {
      results.push_back((*itr).first);
    }
    ++itr;
  }
  fieldIEPluginMutex.unlock();
}

 
FieldIEPlugin *
FieldIEPluginManager::get_plugin(const string &name)
{
  if (field_plugin_table == 0) return NULL;

  // Should check for invalid name.
  map<string, FieldIEPlugin *>::iterator loc = field_plugin_table->find(name);
  if (loc == field_plugin_table->end())
  {
    return NULL;
  }
  else
  {
    return (*loc).second;
  }
}


} // End namespace SCIRun


