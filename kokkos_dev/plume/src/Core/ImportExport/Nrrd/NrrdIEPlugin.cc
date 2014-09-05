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

#include <Core/ImportExport/Nrrd/NrrdIEPlugin.h>
#include <Core/Containers/StringUtil.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <sgi_stl_warnings_on.h>

using namespace std;

#define DEBUG 0

namespace SCIRun {


#ifdef _WIN32
#define SHARE __declspec(dllimport)
#else
#define SHARE
#endif


static map<string, NrrdIEPlugin *> *matrix_plugin_table = 0;
extern SHARE Mutex matrixIEPluginMutex; // From Core/Util/DynamicLoader.cc

//----------------------------------------------------------------------

NrrdIEPlugin::NrrdIEPlugin(const string& pname,
                           const string& fextension,
                           const string& fmagic,
                           NrrdDataHandle (*freader)(ProgressReporter *pr,
                                                 const char *filename),
                           bool (*fwriter)(ProgressReporter *pr,
                                           NrrdDataHandle f,
                                           const char *filename))
  : pluginname_(pname),
    fileExtension_(fextension),
    fileMagic_(fmagic),
    fileReader_(freader),
    fileWriter_(fwriter)
{
  matrixIEPluginMutex.lock();

  if (!matrix_plugin_table)
  {
    matrix_plugin_table = scinew map<string, NrrdIEPlugin *>();
  }

  string tmppname = pluginname_;
  int counter = 2;
  for (;;)
  {
    map<string, NrrdIEPlugin *>::iterator loc = matrix_plugin_table->find(tmppname);
    if (loc == matrix_plugin_table->end())
    {
      if (tmppname != pluginname_) { ((string)pluginname_) = tmppname; }

      (*matrix_plugin_table)[pluginname_] = this;
      break;
    }
    if (*(*loc).second == *this)
    {
      cerr << "WARNING: NrrdIEPlugin '" << tmppname << "' duplicated.\n";
      break;
    }

    cout << "WARNING: Multiple NrrdIEPlugins with '" << pluginname_
	 << "' name.\n";
    tmppname = pluginname_ + "(" + to_string(counter) + ")";
    counter++;
  }

  matrixIEPluginMutex.unlock();
}



NrrdIEPlugin::~NrrdIEPlugin()
{
  if (matrix_plugin_table == NULL)
  {
    cerr << "WARNING: NrrdIEPlugin.cc: ~NrrdIEPlugin(): matrix_plugin_table is NULL\n";
    cerr << "         For: " << pluginname_ << "\n";
    return;
  }

  matrixIEPluginMutex.lock();

  map<string, NrrdIEPlugin *>::iterator iter = matrix_plugin_table->find(pluginname_);
  if (iter == matrix_plugin_table->end())
  {
    cerr << "WARNING: NrrdIEPlugin " << pluginname_ << 
      " not found in database for removal.\n";
  }
  else
  {
    matrix_plugin_table->erase(iter);
  }

  if (matrix_plugin_table->size() == 0)
  {
    delete matrix_plugin_table;
    matrix_plugin_table = 0;
  }

  matrixIEPluginMutex.unlock();
}


bool
NrrdIEPlugin::operator==(const NrrdIEPlugin &other) const
{
  return (pluginname_ == other.pluginname_ &&
	  fileExtension_ == other.fileExtension_ &&
	  fileMagic_ == other.fileMagic_ &&
	  fileReader_ == other.fileReader_ &&
	  fileWriter_ == other.fileWriter_);
}



void
NrrdIEPluginManager::get_importer_list(vector<string> &results)
{
  if (matrix_plugin_table == NULL) return;

  matrixIEPluginMutex.lock();
  map<string, NrrdIEPlugin *>::const_iterator itr = matrix_plugin_table->begin();
  while (itr != matrix_plugin_table->end())
  {
    if ((*itr).second->fileReader_ != NULL)
    {
      results.push_back((*itr).first);
    }
    ++itr;
  }
  matrixIEPluginMutex.unlock();
}


void
NrrdIEPluginManager::get_exporter_list(vector<string> &results)
{
  if (matrix_plugin_table == NULL) return;

  matrixIEPluginMutex.lock();
  map<string, NrrdIEPlugin *>::const_iterator itr = matrix_plugin_table->begin();
  while (itr != matrix_plugin_table->end())
  {
    if ((*itr).second->fileWriter_ != NULL)
    {
      results.push_back((*itr).first);
    }
    ++itr;
  }
  matrixIEPluginMutex.unlock();
}

 
NrrdIEPlugin *
NrrdIEPluginManager::get_plugin(const string &name)
{
  if (matrix_plugin_table == NULL) return NULL;

  // Should check for invalid name.
  map<string, NrrdIEPlugin *>::iterator loc = matrix_plugin_table->find(name);
  if (loc == matrix_plugin_table->end())
  {
    return NULL;
  }
  else
  {
    return (*loc).second;
  }
}


} // End namespace SCIRun


