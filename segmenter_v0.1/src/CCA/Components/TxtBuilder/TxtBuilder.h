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
 *  TxtBuilder.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#ifndef CCA_Components_TxtBuilder_TxtBuilder_h
#define CCA_Components_TxtBuilder_TxtBuilder_h
#include <Core/Thread/Runnable.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <CCA/Components/TxtBuilder/TxtNetwork.h>

using namespace std;
namespace SCIRun {

  class TxtMessage;

  class TxtBuilder : public sci::cca::Component, public Runnable{
  public:
    TxtBuilder();
    ~TxtBuilder();
    void setServices(const sci::cca::Services::pointer& svc);
    static sci::cca::Services::pointer getService();
    static sci::cca::ports::BuilderService::pointer getBuilderService();
    static void displayMessage(const string &msg);
    static string getString(string prompt);
  private:
    WINDOW *win_menu;
    WINDOW *win_main;
    WINDOW *win_msg;
    static WINDOW *win_cmd;

    TxtBuilder(const TxtBuilder&);
    TxtBuilder& operator=(const TxtBuilder&);
    static TxtMessage *tm;
    TxtNetwork network;
    static sci::cca::Services::pointer svc;
    static sci::cca::ports::BuilderService::pointer bs;
    sci::cca::ports::ComponentRepository::pointer cr;
    void run();
    bool exec_command(const char *cmdline);
    int parse(string cmdline, string args[]);
    void use_cluster(string args[]);
    void create(string args[]);
    void connect(string args[]);
    void go(string args[]);
    void disconnect(string args[]);
    void destroy(string args[]);
    void execute(string args[]);
    void load(string args[]);
    void save_as(string args[]);
    void save(string args[]);
    void insert(string args[]);
    void clear(string args[]);
    void quit(string args[]);
    void help(string args[]);
    void bad_command(string args[]);
  };
} //namespace SCIRun

#endif

