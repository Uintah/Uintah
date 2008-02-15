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


// Resources.h - 

#ifndef SCIRun_Resources_h
#define SCIRun_Resources_h 

#include <string>
#include <vector>
#include <map>

namespace SCIRun {
using std::string;
using std::vector;
using std::map;


/*
 * Port
 */

class PortInfo {
public:
  string type_;
  string package_;
  string datatype_;
  string imaker_;
  string omaker_;
  vector<string> libs_;
};

/*
 * Package
 */

class PackageInfo {
public:
  string name_;
  string path_;
  string lib_path_;
  string ui_path_;
  int level_;
};


/*
 * Module
 */


class ModulePortInfo {
public:
  string name_;
  string type_;
};

class ModuleInfo {
public:
  string package_;
  string name_;
  string id_;
  vector<string> categories_;
  string maker_;
  string ui_;
  vector<ModulePortInfo*> iports_;
  vector<ModulePortInfo*> oports_;
  bool has_dynamic_port_;

  vector<string> libs_;
};


/* 
 * Resources
 */ 

class Resources {
public:
  Resources(void);
  ~Resources(void);
  
  // General Info
  // for compatibility with current NetworkEditor
  vector<string> get_packages_names();
  vector<string> get_categories_names( const string & );
  vector<string> get_modules_names( const string &, const string & );

  // Packages
  PackageInfo *get_package_info( const string & );
  string get_package_ui_path( const string & );

  // Modules
  ModuleInfo *get_module_info( const string & );
  const vector<string> &get_module_libs( const string & );
  string get_module_maker( const string & );
  string get_module_ui( const string &);

  // Ports
  PortInfo *get_port_info( const string & );
  const vector<string> &get_port_libs( const string & );
  string get_iport_maker( const string & );
  string get_oport_maker( const string & );

 
  void read( const string & );


private:
  typedef map<string, PackageInfo *> PackagesList;
  typedef map<string,ModuleInfo *> ModulesList;
  typedef map<string,PortInfo *> PortsList;

  PackagesList packages_;
  ModulesList modules_;
  PortsList ports_;

  string data_path_;

  friend class ResourcesParser;
  friend class PackageParser;
  friend class ModuleParser;
  friend class PortParser;
};

// Resources is intended to be a singleton class, but nothing will break
// if you instantiate it many times.  This is the singleton instance,
// on which you should invoke operations:

extern Resources resources;

} // End namespace SCIRun

#endif // SCIRun_Resources_h
