/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef DUMPFIELDS_TEXTDUMPER_H
#define DUMPFIELDS_TEXTDUMPER_H

#include <StandAlone/tools/dumpfields/FieldDumper.h>
#include <StandAlone/tools/dumpfields/Args.h>
#include <StandAlone/tools/dumpfields/FieldSelection.h>

#include <fstream>

namespace Uintah {

  class TextOpts {
  public:
    TextOpts(Args & args);
    
    bool onedim;
    bool tseries;
    
    static std::string options() { 
      return \
        std::string("      -onedim                generate one dim plots") + "\n" +
        std::string("      -tseries               generate single time series file");
    }
  };
  
  class TextDumper : public FieldDumper 
  {
  public:
    TextDumper(DataArchive * da, string basedir, const TextOpts & opts, const FieldSelection & flds);
    
    string directoryExt() const { return "text"; }
    void addField(string fieldname, const Uintah::TypeDescription * /*type*/);
  
    class Step : public FieldDumper::Step {
    public:
      Step(DataArchive * da, string basedir, int timestep, double time, int index, 
           const TextOpts & opts, const FieldSelection & flds);
      
      string infostr() const { return tsdir_; }
      void   storeGrid () {}
      void   storeField(string fieldname, const Uintah::TypeDescription * type);
    
    private:
      DataArchive *        da_;
      string               outdir_;
      TextOpts             opts_;
    const FieldSelection & flds_;
    };
  
    //
    Step * addStep(int timestep, double time, int index);
    void   finishStep(FieldDumper::Step * s);
    
  private:
    std::ofstream               idxos_;
    TextOpts               opts_;
    const FieldSelection & flds_;
    FILE*                  filelist_;
  };
}

#endif
