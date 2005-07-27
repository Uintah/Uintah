#ifndef DUMPFIELDS_TEXTDUMPER_H
#define DUMPFIELDS_TEXTDUMPER_H

#include "FieldDumper.h"

namespace Uintah {
  
  class DXDumper : public Dumper 
  {
  public:
    DXDumper(DataArchive* da, string basedir, bool binmode=false, bool onedim=false);
    ~DXDumper();
  
    string directoryExt() const { return "dx"; }
    void addField(string fieldname, const Uintah::TypeDescription * type);
  
    struct FldWriter { 
      FldWriter(string outdir, string fieldname);
      ~FldWriter();
    
      int      dxobj_;
      ofstream strm_;
      list< pair<float,int> > timesteps_;
    };
  
    class Step : public Dumper::Step {
    public:
      Step(DataArchive * da, string outdir, int index, double time, int fileindex, 
           const map<string,FldWriter*> & fldwriters, bool bin, bool onedim);
    
      void storeGrid ();
      void storeField(string filename, const Uintah::TypeDescription * type);
    
      string infostr() const { return dxstrm_.str()+"\n"+fldstrm_.str()+"\n"; }
    
    private:
      DataArchive*  da_;
      int           fileindex_;
      string        outdir_;
      ostringstream dxstrm_, fldstrm_;
      const map<string,FldWriter*> & fldwriters_;
      bool          bin_, onedim_;
    };
    friend class Step;
  
    //
    Step * addStep(int index, double time, int iset);
    void finishStep(Dumper::Step * step);
  
  private:  
    int           nsteps_;
    int           dxobj_;
    ostringstream timestrm_;
    ofstream      dxstrm_;
    map<string,FldWriter*> fldwriters_;
    bool          bin_;
    bool          onedim_;
    string        dirname_;
  };

}

#endif

