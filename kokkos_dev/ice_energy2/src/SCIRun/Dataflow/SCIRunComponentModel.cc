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
 *  SCIRunComponentModel.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Dataflow/SCIRunComponentModel.h>
#include <SCIRun/Dataflow/SCIRunTCLThread.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Init/init.h>
#include <Core/Util/Environment.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/Scheduler.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <SCIRun/Dataflow/SCIRunComponentDescription.h>
#include <SCIRun/Dataflow/SCIRunComponentInstance.h>
#include <SCIRun/SCIRunFramework.h>
#include <iostream>

//#include <Core/Util/sci_system.h>
#include <main/sci_version.h>

namespace SCIRun {

TCLInterface*
SCIRunComponentModel::gui(0);

Network*
SCIRunComponentModel::net(0);

static bool
split_name(const std::string &type,
           std::string &package,
           std::string &category,
           std::string &module)
{
  unsigned int dot = type.find('.');
  if (dot >= type.size()) { return false; }

  package = type.substr(0, dot);
  std::string rest = type.substr(dot+1);
  dot = rest.find('.');
  if (dot >= rest.size()) { return false; }

  category = rest.substr(0, dot);
  module = rest.substr(dot+1);
  if (module.size() < 1) { return false; }

  return true;
}

SCIRunComponentModel::SCIRunComponentModel(SCIRunFramework* framework)
  : ComponentModel("scirun", framework)
{
  create_sci_environment(0,0);
  packageDB = new PackageDB(0);
  // load the packages
  packageDB->loadPackage();
}

SCIRunComponentModel::~SCIRunComponentModel()
{
}

bool
SCIRunComponentModel::haveComponent(const std::string& type)
{
    std::string package, category, module;
    if (! split_name(type, package, category, module) ) { return false; }

    return packageDB->haveModule(package, category, module);
}

ComponentInstance*
SCIRunComponentModel::createInstance(const std::string& name,
                                     const std::string& type,
                                     const sci::cca::TypeMap::pointer& tm)
{
std::cerr << "SCIRunComponentModel::createInstance: type=" << type << std::endl;
    std::string package, category, module;
    if (! split_name(type, package, category, module) ) { return 0; }

    initGuiInterface();
    Module* m = net->add_module2(package, category, module);
    tm->putBool("dataflow", true);
    SCIRunComponentInstance* ci =
        new SCIRunComponentInstance(framework, name, type, tm, m);
    return ci;
}

void
SCIRunComponentModel::initGuiInterface()
{
  if (SCIRunComponentModel::gui) { return; }

std::cerr << "SCIRunComponentModel::initGuiInterface" << std::endl;

  sci_putenv("SCIRUN_NOSPLASH", "1");
  sci_putenv("SCIRUN_HIDE_PROGRESS", "1");

  SCIRunInit();

  net = new Network();

  SCIRunTCLThread *thr = new SCIRunTCLThread(net);
  Thread* t = new Thread(thr, "SR2 TCL event loop", 0, Thread::NotActivated);

  t->setStackSize(1024*1024);
  t->activate(false);
  t->detach();

  thr->tclWait();

  // Create user interface link
  gui = thr->getTclInterface();

  Scheduler* sched_task=new Scheduler(net);

  // Activate the scheduler.  Arguments and return
  // values are meaningless
  Thread* t2 = new Thread(sched_task, "Scheduler");
  t2->setDaemon(true);
  t2->detach();
}

bool
SCIRunComponentModel::destroyInstance(ComponentInstance * ic)
{
  std::cerr<<"Warning:I don't know how to destroy a SCIRun component instance"
           << std::endl;
  return true;
}

const std::string
SCIRunComponentModel::getName() const
{
  return "Dataflow";
}

void
SCIRunComponentModel::destroyComponentList()
{
  std::cerr << "Error: SCIRunComponentModel does not implement destroyComponentList"
            << std::endl;
}

void
SCIRunComponentModel::buildComponentList(const StringVector& files)
{
  std::cerr << "Error: SCIRunComponentModel does not implement buildComponentList"
            << std::endl;
}

void
SCIRunComponentModel::setComponentDescription(const std::string& type, const std::string& library)
{
  std::cerr << "Error: SCIRunComponentModel does not implement setComponentDescription"
            << std::endl;
}

void
SCIRunComponentModel::listAllComponentTypes(std::vector<ComponentDescription*>& list, bool /*listInternal*/)
{
    std::vector<std::string> packages = packageDB->packageNames();
    typedef std::vector<std::string>::iterator striter;

    for (striter iter = packages.begin(); iter != packages.end(); ++iter) {
        std::string package = *iter;
        std::vector<std::string> categories =
            packageDB->categoryNames(package);
        for (striter iter = categories.begin(); iter != categories.end(); ++iter) {
            std::string category = *iter;
            std::vector<std::string> modules =
                packageDB->moduleNames(package, category);
            for (striter iter = modules.begin(); iter != modules.end(); ++iter) {
                std::string module = *iter;
                list.push_back(
                    new SCIRunComponentDescription(this, package, category, module));
            }
        }
    }
}

}// end namespace SCIRun
