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


// PackageDB.h - Interface to module-finding and loading mechanisms

#ifndef PSE_Dataflow_PackageDB_h
#define PSE_Dataflow_PackageDB_h 1

#include <Core/Containers/AVLTree.h>
#include <Core/Util/soloader.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/share.h>

namespace SCIRun {

  class GuiInterface;
  using namespace std;
    typedef struct {
      string name;
      string datatype;
      iport_maker maker;
    } IPortInfo;

    typedef struct {
      string name;
      string datatype;
      oport_maker maker;
    } OPortInfo;

    typedef struct {
      string packageName;
      string categoryName;
      string moduleName;
      string optional;
      string help_description;
      ModuleMaker maker;
      string uiFile;
      std::map<int,IPortInfo*>* iports;
      std::map<int,OPortInfo*>* oports;
      char lastportdynamic;
    } ModuleInfo;

    typedef AVLTree<string,ModuleInfo*> Category;
    typedef AVLTree<string,Category*> Package;
    typedef AVLTree<string,Package*> Packages;
    
    typedef AVLTreeIter<string,ModuleInfo*> CategoryIter;
    typedef AVLTreeIter<string,Category*> PackageIter;
    typedef AVLTreeIter<string,Package*> PackagesIter;

    class SHARE PackageDB {
    public:
      PackageDB(GuiInterface* gui);
      ~PackageDB();

      void		loadPackage(bool resolve=true);
      void		setGui(GuiInterface* gui);
      Module*		instantiateModule(const string& packageName,
					  const string& categoryName,
					  const string& moduleName,
					  const string& instanceName);
      bool		haveModule(const string& packageName,
				   const string& categoryName,
				   const string& moduleName) const;
      vector<string>	packageNames () const;
      vector<string>	categoryNames(const string& packageName) const;
      vector<string>	moduleNames  (const string& packageName,
				      const string& categoryName) const;
      ModuleInfo*	GetModuleInfo(const string& name,
				      const string& catname,
				      const string& packname);
      // Used if the module has changed categories.
      string		getCategoryName(const string &packName,
					const string &catName,
					const string &modName);
    private:

      bool		findMaker(ModuleInfo* info);
      void		registerModule(ModuleInfo* info);
      void		gui_exec(const string&);
      void		printMessage(const string&);
      vector<string>	delayed_commands_;
      Packages *        db_;
      vector<string>    packageList_;
      GuiInterface *	gui_;
    };

    // PackageDB is intended to be a singleton class, but nothing will break
    // if you instantiate it many times.  This is the singleton instance,
    // on which you should invoke operations:
#if defined(_WIN32) && !defined(BUILD_Dataflow_Network)
     __declspec(dllimport) PackageDB* packageDB;
#else
     extern PackageDB* packageDB;
#endif

} // End namespace SCIRun

#endif // PSE_Dataflow_PackageDB_h
