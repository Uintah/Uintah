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
 *  SCIRunComponentInstance.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Dataflow/SCIRunComponentInstance.h>
#include <SCIRun/Dataflow/SCIRunPortInstance.h>
#include <SCIRun/Dataflow/SCIRunUIPort.h>
#include <SCIRun/Dataflow/SCIRunGoPort.h>
#include <SCIRun/CCA/CCAPortInstance.h>
#include <SCIRun/TypeMap.h>

#include <Dataflow/Network/Module.h>

namespace SCIRun {

SCIRunComponentInstance::SCIRunComponentInstance(
    SCIRunFramework* framework,
    const std::string& instanceName,
    const std::string& className,
    const sci::cca::TypeMap::pointer& tm,
    Module* module)
 : ComponentInstance(framework, instanceName, className, tm), module(module)
{
    // See if we have a user-interface...
    if (module->have_ui()) {
        specialPorts.push_back(
            new CCAPortInstance("ui", "sci.cca.ports.UIPort",
                                sci::cca::TypeMap::pointer(0),
                                sci::cca::Port::pointer(new SCIRunUIPort(this)),
                                CCAPortInstance::Provides));
    }

    // map Module execute function to CCA Go port
    specialPorts.push_back(
        new CCAPortInstance("go", "sci.cca.ports.GoPort",
                            sci::cca::TypeMap::pointer(0),
                            sci::cca::Port::pointer(new SCIRunGoPort(this)),
                            CCAPortInstance::Provides));
}

SCIRunComponentInstance::~SCIRunComponentInstance()
{
}

PortInstance* SCIRunComponentInstance::getPortInstance(const std::string& name)
{
    static const int INPUT_LEN = 7;
    static const int OUTPUT_LEN = 8;

    // SCIRun ports can potentially have the same name for both, so
    // SCIRunPortInstance tags them with a prefix of "Input: " or
    // "Output: ", so we need to check that first.
    if (name.substr(0, INPUT_LEN) == "Input: ") {
        IPort* port = module->get_input_port(name.substr(7));
        if (!port) {
            return 0;
        }
	sci::cca::TypeMap::pointer tm(new TypeMap);
        return new SCIRunPortInstance(this, port, tm, SCIRunPortInstance::Input);
    } else if (name.substr(0, OUTPUT_LEN) == "Output: ") {
        OPort* port = module->get_output_port(name.substr(OUTPUT_LEN));
        if (!port) {
            return 0;
        }
	sci::cca::TypeMap::pointer tm(new TypeMap);
        return new SCIRunPortInstance(this, port, tm, SCIRunPortInstance::Output);
    } else {
        for (unsigned int i = 0; i < specialPorts.size(); i++) {
            if (specialPorts[i]->getName() == name) {
                return specialPorts[i];
            }
        }
        return 0;
    }
}

sci::cca::TypeMap::pointer SCIRunComponentInstance::getPortProperties(const std::string& /*portName*/)
{
  return sci::cca::TypeMap::pointer(new TypeMap);
}

PortInstanceIterator* SCIRunComponentInstance::getPorts()
{
  return new Iterator(this);
}

SCIRunComponentInstance::Iterator::Iterator(SCIRunComponentInstance* component)
  : component(component), idx(0)
{
}

SCIRunComponentInstance::Iterator::~Iterator()
{
}

bool SCIRunComponentInstance::Iterator::done()
{
  return idx >= (int)component->specialPorts.size()
    +component->module->num_output_ports()
    +component->module->num_input_ports();
}

PortInstance* SCIRunComponentInstance::Iterator::get()
{
    Module* module = component->module;
    int spsize = static_cast<int>(component->specialPorts.size());
    if (idx < spsize) {
        return component->specialPorts[idx];
    } else if (idx < spsize + module->num_output_ports()) {
	sci::cca::TypeMap::pointer tm(new TypeMap);
	// TODO: check memory leak
        return new SCIRunPortInstance(component,
                                      module->get_output_port(idx - spsize),
				      tm,
                                      SCIRunPortInstance::Output);
    } else if (idx < spsize + module->num_output_ports() + module->num_input_ports()) {
	sci::cca::TypeMap::pointer tm(new TypeMap);
	// TODO: check memory leak
        return new SCIRunPortInstance(component,
                                      module->get_input_port(idx - spsize - module->num_output_ports()),
				      tm,
                                      SCIRunPortInstance::Input);
    } else {
        return 0; // Illegal
    }
}

} // end namespace SCIRun
