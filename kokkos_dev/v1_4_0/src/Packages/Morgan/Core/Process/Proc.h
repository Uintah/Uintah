/*
 *  Proc.h: A process
 *
 *  Written by:
 *   Jason V. Morgan
 *   Department of Computer Science
 *   University of Utah
 *   December 2000
 *
 *  Updated by:
 *   Jason V. Morgan
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Morgan_Process_Proc_h
#define Morgan_Process_Proc_h

#include <Packages/Morgan/Core/Process/ProcManagerException.h>
#include <Packages/Morgan/Core/FdStream/FdStreambuf.h>
#include <stdio.h>
#include <string>
#include <iostream>

namespace Morgan {
    namespace Process {

    using Morgan::fdstreambuf;
    using std::istream;
    using std::ostream;

    class Proc {
    public:
        Proc(int pid, int input_fd, int output_fd);
        ~Proc();

        istream& ifs() { return ifs_; }
        ostream& ofs() { return ofs_; }

    private:
        fdstreambuf ifsb; // the input stream buffer, must come before ifs_
        fdstreambuf ofsb; // the output stream buffer, must come before ofs_
        istream ifs_; // the input stream
        ostream ofs_; // the output stream
        int pid; // the process id
    };

    }
}

#endif
