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

#ifndef DUMPFIELDS_HIST_DUMPER_H
#define DUMPFIELDS_HIST_DUMPER_H

#include <StandAlone/tools/dumpfields/FieldDumper.h>
#include <StandAlone/tools/dumpfields/Args.h>
#include <StandAlone/tools/dumpfields/FieldSelection.h>

#include <fstream>

namespace Uintah {
  
  class HistogramOpts {
  public:
    HistogramOpts(Args & args);
    
    int    nbins;
    double minval;
    double maxval;
    bool   normalize_by_bins;
    double xscale;
    
    static std::string options() { 
      return \
        std::string("      -nbins  value          number of histogram bins") + "\n" +
        std::string("      -minval value          value of minimum bin") + "\n" +
        std::string("      -maxval value          value of maximum bin") + "\n" +
        std::string("      -normalize             normalize by bin count") + "\n" +
        std::string("      -normscale             normalize range by this value");
    }
  };
  
  class HistogramDumper : public FieldDumper 
  {
  public:
    HistogramDumper(DataArchive* da, string basedir, HistogramOpts opts,
                    const FieldSelection & fselect);
    
    string directoryExt() const { return "hist"; }
    void addField(string /*fieldname*/, const Uintah::TypeDescription * /*theType*/) {}
  
    class Step : public FieldDumper::Step {
    public:
      Step(DataArchive * da, string outdir, int timestep, double time, int index, 
           const HistogramOpts & opts, const FieldSelection & fselect);
    
      void storeGrid () {}
      void storeField(string fieldname, const Uintah::TypeDescription * type);
    
      string infostr() const { return stepdname_; }
    
    private:
      string stepdname_;
      
      DataArchive* da_;
      string       basedir_;
      
      const HistogramOpts &  opts_;
      const FieldSelection & fselect_;
    };
  
    //
    Step * addStep(int index, double time, int iset);
    void   finishStep(FieldDumper::Step * s);
  
  private:
    std::ofstream idxos_;
    FILE*         filelist_;
    
    HistogramOpts opts_;
    const FieldSelection & fselect_;
  };

}

#endif
