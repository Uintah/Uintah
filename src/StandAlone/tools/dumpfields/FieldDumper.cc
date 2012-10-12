/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "FieldDumper.h"
#include <Core/OS/Dir.h>
#include <iomanip>

using namespace std;

namespace Uintah {
  using namespace SCIRun;

  FieldDumper::FieldDumper(DataArchive * da, string basedir)
    : da_(da), basedir_(basedir)
  {
  }

  string
  FieldDumper::createDirectory()
  {
    // Create a directory if it's not already there.
    // The exception occurs when the directory is already there
    // and the Dir.create fails.  This exception is ignored. 
    string dirname = this->basedir_+"_"+this->directoryExt();
    Dir dumpdir;
    try {
      cout << "creating " << dirname << endl;
      dumpdir.create(dirname);
    } catch (ErrnoException & e) {
      cout << "creating failed for reason " << e.getErrno() << endl;
      cout << "ignore on " << EEXIST << endl;
      // only allow exists as a reason for failure
      if(e.getErrno()!= EEXIST) {
        cout << "unknown error - rethrowing" << endl;
        throw;
      }
    }
    return dirname;
  }

  string
  FieldDumper::time_string(double tval)
  {
    ostringstream b;
    b.setf(ios::fixed,ios::floatfield);
    b.precision(8);
    b << setw(12) << setfill('0') << tval;
    string btxt(b.str());
    string r;
    for(string::const_iterator bit(btxt.begin());bit!=btxt.end();bit++) {
      char c = *bit;
      if(c=='.' || c==' ')
        r += '_';
      else
        r += c;
    }
    return r;
  }

  string
  FieldDumper::step_string(int istep)
  {
    ostringstream b;
    b.setf(ios::fixed,ios::floatfield);
    b.precision(8);
    b << setw(8) << setfill('0') << istep;
    return b.str();
  }

  string
  FieldDumper::mat_string(int mval)
  {
    ostringstream b;
    b << setw(4) << setfill('0') << mval;
    return b.str();
  }

  string 
  FieldDumper::dirName(double time_val, int istep) const
  {
    return this->basedir_+"_"+this->directoryExt()+"/"+step_string(istep);
    // return this->basedir_+"_"+this->directoryExt()+"/"+time_string(time_val);
  }

  FieldDumper::~FieldDumper() {}

  FieldDumper::Step::Step(string tsdir, int timestep, double time, int index, bool nocreate)
    : tsdir_(tsdir), timestep_(timestep), time_(time), index_(index)
  {
    // create directory to store output files 
    if(!nocreate) {
      Dir stepdir;
      try {
        stepdir.create(tsdir_);
      } catch (ErrnoException & e) {
        // only allow exists as a reason for failure
        if(e.getErrno()!= EEXIST)
          throw;
      }}
  }

  string
  FieldDumper::Step::fileName(string variable_name, string ext)  const
  {
    return fileName(variable_name, -1, ext);
  }
  
  string
  FieldDumper::Step::fileName(string variable_name, int imat, string ext)  const
  {
    string datafile = tsdir_+"/";
  
    datafile += string("VAR_") + variable_name + string(".");
  
    if (imat>=0)
      datafile+= string("MT_") + mat_string(imat) + string(".");
  
    if (ext!="")
      datafile+=ext;
  
    return datafile;
  }
  
  FieldDumper::Step::~Step() {}
}
