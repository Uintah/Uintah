/*
 *  ProcDbd.cc:
 *
 *  Written by:
 *   Jason V. Morgan
 *   December 27, 2000
 *
 */

#include <Packages/Morgan/Core/Process/Proc.h>
#include <Packages/Morgan/Core/Process/ProcManager.h>
#include <Packages/Morgan/Core/Dbi/ProcDbd.h>
#include <Packages/Morgan/Core/PackagePathIterator/PackagePathIterator.h>
#include <Packages/Morgan/share/share.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <strstream>

using Morgan::Dbi::Dbd;
using Morgan::Dbi::ProcDbd;
using Morgan::Process::Proc;
using Morgan::Process::ProcManager;
using Morgan::Process::ProcManagerException;
using namespace std;


ProcDbd::ProcDbd(Morgan::Process::Proc* iproc) : proc(iproc) {
}

ProcDbd::~ProcDbd() {
    delete proc; // kill the process that was opened
}

bool ProcDbd::execute(const char* statement) {
    string buf;

    proc->ofs() << "do: " << statement << endl;
    getline(proc->ifs(), buf);
    return buf == "ok:";
}

bool ProcDbd::fetch(string& buf, int col) {
    string ibuf;
    proc->ofs() << "fetch: " << col << endl; 
    getline(proc->ifs(), ibuf);

    // see if the fetch failed
    if(strncmp(ibuf.c_str(), "error:", 6) == 0) {
        last_error = ibuf;
        last_error = string(ibuf.c_str() + 6);
        return false;
    }

    buf = string(ibuf.c_str() + 4);

    return true;
}

bool ProcDbd::is_null(int col) {
    ostrstream str;
    str << "is_null: " << col << ends;
    return bool_query(str.str());
}

void ProcDbd::next_row() {
    proc->ofs() << "next_row: " << endl;
}

bool ProcDbd::at_end() {
    return bool_query("at_end:");
} 

int ProcDbd::cols() {
    return int_query("cols:");
}

Dbd* ProcDbd::connect(const char* database, const char* hostname, int port,
                      const char* username, const char* password) {
    try {
        string buf;
        Proc* proc = 0;
        for(PackagePathIterator i ; !proc && i ; ++i) {
            string command(*i);
            command += "/sql";
            cout << "Trying to execute: " << command << endl;
            proc = ProcManager::start_proc(command.c_str(), NULL);
        }
        if(!proc) {
            return NULL;
        }
        proc->ofs() << "database: " << database << endl;
        if(hostname) {
            proc->ofs() << "hostname: " << hostname << endl;
            proc->ofs() << "port: " << port << endl;
        }
        if(username) {
            proc->ofs() << "username: " << username << endl;
        }
        if(password) {
            proc->ofs() << "password: " << password << endl;
        }
        proc->ofs() << "connect:" << endl;
        if(!proc->ofs()) {
            perror("Error writing to process");
            return NULL;
        }

        fprintf(stderr, "Flushed output\n");

        getline(proc->ifs(), buf);

        if(buf == "error:") {
            proc->ifs() >> ws;
            getline(proc->ifs(), buf); // get the error string
            fprintf(stderr, "Unable to connect to database: %s", buf.c_str());
            delete proc;
            return NULL;
        }

        if(buf != "ok:") {
            fprintf(stderr, "%s", buf.c_str());
            delete proc;
            return NULL;
        }

        return new ProcDbd(proc);
    } catch(ProcManagerException&) {
        return NULL; // could not start up process
    }
}

bool ProcDbd::bool_query(const char* query) {
    string buf;
    proc->ofs() << query << endl;
    getline(proc->ifs(),  buf);
    if(buf == "ok: true") {
        return true;
    } else {
        return false;
    }
}

int ProcDbd::int_query(const char* query) {
    string buf;
    proc->ofs() << query << endl;
    getline(proc->ifs(), buf);
    if(strncmp(buf.c_str(), "ok:", 3) == 0) {
        return atoi(buf.c_str() + 3);
    } else {
        return 0;
    }
}
