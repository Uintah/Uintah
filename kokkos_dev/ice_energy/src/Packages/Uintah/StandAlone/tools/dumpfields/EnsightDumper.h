#ifndef DUMPFIELDS_ENSIGHT_DUMPER_H
#define DUMPFIELDS_ENSIGHT_DUMPER_H

#include "FieldDumper.h"
#include "Args.h"
#include "FieldSelection.h"

#include <iomanip>

namespace Uintah {
  
  class EnsightOpts {
  public:
    EnsightOpts(Args & args);
    
    bool withpart;
    bool onemesh;
    bool binary;
    
    static std::string options() { 
      return \
        std::string("      -withpart              include particle fields") + "\n" +
        std::string("      -onemesh               store single mesh for all time steps") + "\n" +
        std::string("      -bin                   store in binary format");
    }
  };
  
  class EnsightDumper : public FieldDumper 
  {
  private:
    // dump field as text or binary
    struct FldDumper { 
      // ensight is very fussy about the format of the text, so it's nice
      // to have this all in one place
      FldDumper(bool bin) : bin_(bin), os_(0) {}
    
      void setstrm(ostream * os) { os_ = os; }
      void unsetstrm() { os_ = 0; }
    
      void textfld(string v, int width=80) {
        *os_ << setw(width) << setiosflags(ios::left) << v;
      }
      
      void textfld(string v, int width, int binwidth) {
        if(bin_)
          *os_ << setw(binwidth)  << setiosflags(ios::left) << v;
        else
          *os_ << setw(width)  << setiosflags(ios::left) << v;
      }
    
      void textfld(int v, int width=10) {
        *os_ << setiosflags(ios::left) << setw(width) << v;
      }
    
      void textfld(float v) { 
        char b[13];
        snprintf(b, 13, "%12.5e", v);
        *os_ << b;
      }
    
      void textfld(double v) { 
        char b[13];
        snprintf(b, 13, "%12.5e", v);
        *os_ << b;
      }
    
      void numfld(int v, int wid=10) { 
        if(!bin_) {
          textfld(v, wid);
        } else {
          os_->write((char*)&v, sizeof(int));
        }
      }
    
      void endl() { 
        if(!bin_) {
          *os_ << std::endl;
        }
      }
    
      void numfld(float v) { 
        v = (fabs(v)<FLT_MIN)?0.:v;
        if(!bin_) {
          textfld(v);
        } else {
          os_->write((char*)&v, sizeof(float));
        }
      }
    
      void numfld(double v) { 
        numfld((float)v);
      }
    
      bool      bin_;
      ostream * os_;
    };

    struct Data {
      Data(DataArchive * da, FldDumper * dumper, 
           const EnsightOpts & opts, const FieldSelection & fselect)
        : 
        da_(da), dumper_(dumper), onemesh_(opts.onemesh), withpart_(opts.withpart), fselect_(fselect)
      {}
      
      DataArchive * da_;
      string        dir_;
      FldDumper   * dumper_;
      bool          onemesh_, withpart_;
      const FieldSelection & fselect_;
    };
  
  public:
    EnsightDumper(DataArchive* da, string basedir, 
                  const EnsightOpts & opts, const FieldSelection & fselect);
    ~EnsightDumper();
    
    string directoryExt() const { return "ensight"; }
    void   addField(string fieldname, const Uintah::TypeDescription * type);
  
    class Step : public FieldDumper::Step {
      friend class EnsightDumper;
    private:
      Step(Data * data, string tsdir, int index, double time, int fileindex);
      
    public:
      string infostr() const { return stepdesc_; }
      void   storeGrid ();
      void   storeField(string filename, const Uintah::TypeDescription * type);
    
    private:
      void storePartField(string filename, const Uintah::TypeDescription * type);
      void storeGridField(string filename, const Uintah::TypeDescription * type);
    
    private:
      int    fileindex_;
      string stepdname_;
      string stepdesc_;
    
    private:
      Data    * data_;
      bool      needmesh_;
      IntVector vshape_, minind_;
    };
    friend class Step;
  
    //
    Step * addStep(int index, double time, int iset);
    void finishStep(FieldDumper::Step * step);
    
  private:
    int           nsteps_;
    ofstream      casestrm_;
    int           tscol_;
    ostringstream tsstrm_;
    FldDumper     flddumper_;
    Data          data_;
  };

}

#endif

