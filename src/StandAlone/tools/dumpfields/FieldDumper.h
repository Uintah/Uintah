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

#ifndef DUMPFIELDS_FIELD_DUMPER_H
#define DUMPFIELDS_FIELD_DUMPER_H

#include <Core/DataArchive/DataArchive.h>
#include <Core/Disclosure/TypeDescription.h>
#include <string>

namespace Uintah {
  
  class FieldDumper 
  {
  protected:
    FieldDumper(DataArchive * da, string basedir);
    
    string dirName(double tval, int iset) const;
    string createDirectory();
    
  public:
    virtual ~FieldDumper();
    
    // extension for the output directory
    virtual string directoryExt() const = 0;
    
    // add a new field
    virtual void addField(string fieldname, const Uintah::TypeDescription * type) = 0;
    
    // dump a single step
    class Step {
    protected:
      Step(string tsdir, int timestep, double time, int index, bool nocreate=false);
      
      string fileName(string variable_name, string extension="") const;
      string fileName(string variable_name, int materialNum, string extension="") const;
      
    public:
      virtual ~Step();
      
      virtual string infostr() const = 0;
      virtual void storeGrid() = 0;
      virtual void storeField(string fieldname, const Uintah::TypeDescription * type) = 0;
      
    public: // FIXME: 
      string tsdir_;
      int    timestep_;
      double time_;
      int index_;
    };
    virtual Step * addStep(int timestep, double time, int index) = 0;
    virtual void finishStep(Step * step) = 0;
    
    DataArchive * archive() const { return this->da_; }
    
  private:
    static string mat_string(int mval);
    static string time_string(double tval);
    static string step_string(int istep);
    
  private:
    DataArchive* da_;
    string       basedir_;
  };
}

#endif

