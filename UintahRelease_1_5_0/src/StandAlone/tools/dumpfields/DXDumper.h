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
      Step(DataArchive * da, string outdir, int timestep, double time, int index, int fileindex, 
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
    Step * addStep(int timestep, double time, int index);
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

