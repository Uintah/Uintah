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
 * VergilApplication.cc
 * 
 */

#include <Dataflow/Network/Module.h>     // module base class
#include <Core/GuiInterface/GuiVar.h>    // GUI data interface
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/Assert.h>
#include <sci_defs/ptolemy_defs.h>

#include <Packages/Ptolemy/Core/jni/JNIUtil.h>
#include <Packages/Ptolemy/Core/jni/VergilWrapper.h>

#include <string>

using std::string;

namespace Ptolemy {

using namespace SCIRun;

class VergilApplication : public Module
{
public:
    VergilApplication(GuiContext* ctx);

    // virtual destructor
    virtual ~VergilApplication();

    // virtual execute function
    virtual void execute();

protected:
    GuiString filename_;

private:
    JavaVM *jvm;
    std::string defaultConfig;
    std::string defaultModel;
    std::string file;
};

DECLARE_MAKER(VergilApplication)
VergilApplication::VergilApplication(GuiContext* ctx)
    : Module("VergilApplication", ctx, Source, "Vergil", "Ptolemy"), filename_(ctx->subVar("filename"))
{
    std::cerr << "VergilApplication::VergilApplication" << std::endl;
    jvm = JNIUtil::getJavaVM();
    ASSERT(jvm != 0);
    // hardcoded for now
    std::string path(PTOLEMY_PATH);
    defaultConfig = path + "/ptolemy/configs";
    defaultModel = path + "/ptolemy/moml/demo/modulation.xml";
}

VergilApplication::~VergilApplication()
{
    std::cerr << "VergilApplication::~VergilApplication" << std::endl;
    // detach current thread from JVM?
    JNIUtil::destroyJavaVM();
}

void VergilApplication::execute()
{
    std::cerr << "VergilApplication::execute" << std::endl;

    try {
        file = filename_.get();
        // use defaults if a file name was not set
        if (file.empty()) {
            file = defaultModel;
        }
        Thread *t = new Thread(scinew VergilThread(defaultConfig, file),
            "Ptolemy Thread", 0, Thread::NotActivated);
        t->setStackSize(1024*1024);
        t->activate(false);
        t->detach();
    }
    catch (const Exception& e) {
        std::cerr << "Caught exception:\n";
        std::cerr << e.message() << std::endl;
        abort();
    }
    catch (...) {
        std::cerr << "Caught unexpected exception!\n";
        abort();
    }
}

}
