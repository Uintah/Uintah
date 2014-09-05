#ifndef DUMPFIELDS_HIST_DUMPER_H
#define DUMPFIELDS_HIST_DUMPER_H

#include "FieldDumper.h"
#include "Args.h"
#include "FieldSelection.h"

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
      Step(DataArchive * da, string outdir, int index, double time, 
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
    ofstream idxos_;
    FILE*    filelist_;
    
    HistogramOpts opts_;
    const FieldSelection & fselect_;
  };

}

#endif
