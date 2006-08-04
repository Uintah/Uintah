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
 *  QtUtils.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <CCA/Components/Builder/QtUtils.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <iostream>

#include <qapplication.h>


using namespace SCIRun;

static QApplication* theApp;
static Semaphore* startup;

class QtThread : public Runnable {
public:
    QtThread() {}
    ~QtThread() {}
    void run();
};


QApplication* QtUtils::getApplication()
{
    if ( !theApp ) {
        startup = new Semaphore("Qt Thread startup wait", 0);
        Thread* t = new Thread(new QtThread(), "SCIRun Builder",
                               0, Thread::NotActivated);
        t->setStackSize(8*256*1024);
        t->activate(false);
        t->detach();
        startup->down();
    }
    return theApp;
}

void QtThread::run()
{
    std::cerr << "******************QtThread::run()**********************" << std::endl;
    int argc = 3;
    char* argv[3];
    argv[0] = "SCIRun2";
    argv[1] = "-im";
    argv[2] = "-iconic";

    theApp = new QApplication(argc, argv);
    startup->up();

    theApp->exec();
}
