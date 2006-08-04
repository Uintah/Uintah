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

   The above copyright notice and this permission noice shall be included
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
 *  pingpong.cc
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 U of U
 */

#include <iostream>
#include <unistd.h>
#include <Core/CCA/PIDL/PIDL.h>

#include <Core/CCA/PIDL/MalformedURL.h>

#include <testprograms/Component/pingpong/PingPong_impl.h>
#include <testprograms/Component/pingpong/PingPong_sidl.h>
#include <Core/Thread/Time.h>

using std::cerr;
using std::cout;

using namespace SCIRun;

void usage(char* progname)
{
    cerr << "usage: " << progname << " [options]\n";
    cerr << "valid options are:\n";
    cerr << "  -server  - server process\n";
    cerr << "  -client URL  - client process\n";
    cerr << "  -reps n  - repeat n times\n";
    cerr << "\n";
    exit(0);
}

int main(int argc, char* argv[])
{
    using std::string;

    using PingPong_ns::PingPong_impl;
    using PingPong_ns::PingPong;

    try {
        bool client=false;
        bool server=false;
        string client_url;
        int reps=10;

        for (int i = 1; i < argc; i++) {
            string arg(argv[i]);
            if (arg == "-server") {
                if (client) {
                    usage(argv[0]);
                }
                server=true;
            } else if (arg == "-client") {
                if (server) {
                    usage(argv[0]);
                }
                if (++i >= argc) {
                    usage(argv[0]);
                }
                client_url=argv[i];
                client=true;
            } else if (arg == "-reps") {
                if (++i >= argc) {
                    usage(argv[0]);
                }
                reps=atoi(argv[i]);
            } else {
                usage(argv[0]);
            }
        }
        if (!client && !server) {
            usage(argv[0]);
        }

        PIDL::initialize();

        if (server) {
            PingPong::pointer pp(new PingPong_impl);
            pp->addReference();
            cerr << "Waiting for pingpong connections...\n";
            cerr << pp->getURL().getString() << '\n';
        } else {
            //cerr << "calling objectFrom\n";
            Object::pointer obj=PIDL::objectFrom(client_url);
            //cerr << "objectFrom completed\n";
            //cerr << "calling pidl_cast\n";
            PingPong::pointer pp=  pidl_cast<PingPong::pointer>(obj);
            //cerr << "pidl_case completed\n";
            if (pp.isNull()) {
                cerr << "pp_isnull\n";
                abort();
            }
            double stime=Time::currentSeconds();
            for (int i = 0; i < reps; i++) {
                int j = pp->pingpong(i);
                if (i != j) {
                    cerr << "BAD data: " << i << " vs. " << j << '\n';
                }
            }

            double dt=Time::currentSeconds()-stime;
            cerr << reps << " reps in " << dt << " seconds\n";
            double us = dt / reps * 1000 * 1000;
            cerr << us << " us/rep\n";
        }
        PIDL::serveObjects();
        cout << "serveObjects done!\n";
        PIDL::finalize();
    } catch(const MalformedURL& e) {
        cerr << "pingpong.cc: Caught MalformedURL exception:\n";
        cerr << e.message() << '\n';
    } catch(const Exception& e) {
        cerr << "pingpong.cc: Caught exception:\n";
        cerr << e.message() << '\n';
        abort();
    } catch(...) {
        cerr << "Caught unexpected exception!\n";
        abort();
    }
    return 0;
}

