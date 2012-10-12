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

#include <StandAlone/tools/dumpfields/HistogramDumper.h>
#include <StandAlone/tools/dumpfields/ScalarDiags.h>
#include <StandAlone/tools/dumpfields/TensorDiags.h>

#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <iomanip>
#include <sstream>
#include <fstream>

#include <cfloat>

using namespace std;

namespace Uintah {
  
  HistogramOpts::HistogramOpts(Args & args)
  {
    nbins  = args.getInteger("nbins",   256);
    minval = args.getReal("minval", +FLT_MAX);
    maxval = args.getReal("maxval", -FLT_MAX);
    xscale = args.getReal("normscale", 1.);
    normalize_by_bins = args.getLogical("normalize");
  }
  
  HistogramDumper::HistogramDumper(DataArchive* da, string basedir, 
                                   HistogramOpts opts, const FieldSelection & fselect)
    : FieldDumper(da, basedir), opts_(opts), fselect_(fselect)
  {
    // set defaults for cout
    cout.setf(ios::scientific,ios::floatfield);
    cout.precision(8);
  
    string outdir = this->createDirectory();
  
    // set up the file that contains a list of all the files
    string filelistname = outdir + string("/") + string("timelist");
    filelist_ = fopen(filelistname.c_str(),"w");
    if (!filelist_) {
      cerr << "Can't open output file " << filelistname << endl;
      abort();
    }
  }

  HistogramDumper::Step * 
  HistogramDumper::addStep(int timestep, double time, int index)
  {
    return scinew Step(this->archive(), this->dirName(time, index), timestep, time, index, opts_, fselect_);
  }

  void
  HistogramDumper::finishStep(FieldDumper::Step * s)
  {
    fprintf(filelist_, "%10d %16.8g  %20s\n", s->timestep_, s->time_, s->infostr().c_str());
  }

  HistogramDumper::Step::Step(DataArchive * da, string tsdir, int timestep, double time, int index,  
                              const HistogramOpts & opts, const FieldSelection & fselect)
    : 
    FieldDumper::Step(tsdir, timestep, time, index), da_(da), 
    opts_(opts), fselect_(fselect)
  {
  }
  
  static inline double MIN(double a, double b) { if(a<b) return a; return b; }
  static inline double MAX(double a, double b) { if(a>b) return a; return b; }
  
  static void _binSingleVal(vector<int> & bins, double minval, double maxval, double val)
  {
    if(val<minval || val>maxval) return;
    int nbins = (int)bins.size();
    int ibin = (val==maxval)?nbins-1:((int)(nbins*(val-minval)/(maxval-minval)));
    bins[ibin]++;
  }

  void
  HistogramDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
  {
    if(!fselect_.wantField(fieldname)) return;
    
    GridP grid = da_->queryGrid(index_);
    
    // need to count materials before we start, since we want material loop outside
    // of patch loop
    vector<int> mats;
    for(int l=0;l<=0;l++) {
      LevelP level = grid->getLevel(l);
      for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
        const Patch* patch = *iter;
        ConsecutiveRangeSet matls= da_->queryMaterials(fieldname, patch, index_);
        for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
          int matl = *matlIter;
          if(fselect_.wantMaterial(matl) && !count(mats.begin(), mats.end(), matl)) 
            mats.push_back(matl);
        }
      }
    }
    
    // loop through requested diagnostics
    TensorDiag const * tensor_preop = createTensorOp(fselect_);
    list<ScalarDiag const *> scalardiaggens = createScalarDiags(td, fselect_, tensor_preop);
    
    if(scalardiaggens.size())
      {
        cout << "   " << fieldname << endl;
      }
    else
      {
        static int noisecount = 0;
        if(++noisecount<=20)
          {
            cout << "   WARNING: Field '" << fieldname << "' has no scalar diagnostics specified" << endl;
            cout << "            You probably want a '-diagnostic magnitude' option for vectors" << endl;
            cout << "            or the '-diagnostic norm' option for tensors." << endl;
            cout << endl;
            cout << "            Try running with '-showdiags' to get the available diagnostics" << endl;
            cout << endl;
          }
      }
    
    for(list<ScalarDiag const *>::const_iterator diagit(scalardiaggens.begin());
        diagit!=scalardiaggens.end();diagit++) 
      {
        vector<int> bins(opts_.nbins);
        double minval=opts_.minval;
        double maxval=opts_.maxval;
        
        int ipass0 = 1;
        if(minval>maxval) ipass0 = 0;
        
        for(int ipass=ipass0;ipass<2;ipass++) {
          
          // find range
          // for each point, find scalar val and bin it
          for(int ibin=0;ibin<opts_.nbins;ibin++) bins[ibin] = 0;
          
          for(vector<int>::const_iterator mit(mats.begin());mit!=mats.end();mit++) {
            int matl = *mit;
            
            LevelP level = grid->getLevel(0);
            for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
              const Patch* patch = *iter;
              
              if(td->getType()==Uintah::TypeDescription::CCVariable) {
                
                CCVariable<double> svals;
                (**diagit)(da_, patch, fieldname, matl, index_, svals);
                
                for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
                  double val = svals[*iter];
                  if(ipass==1)
                    _binSingleVal(bins, minval, maxval, val);
                  else {
                    minval = MIN(minval, val);
                    maxval = MAX(maxval, val);
                  }
                }
              } else if(td->getType()==Uintah::TypeDescription::NCVariable) {

                NCVariable<double> svals;
                (**diagit)(da_, patch, fieldname, matl, index_, svals);
                
                for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
                  double val = svals[*iter];
                  
                  if(ipass==1)
                    _binSingleVal(bins, minval, maxval, val);
                  else {
                    minval = MIN(minval, val);
                    maxval = MAX(maxval, val);
                  }
                }
              } else if (td->getType()==Uintah::TypeDescription::ParticleVariable) {
                ParticleVariable<Point> posns;
                da_->query(posns, "p.x", matl, patch, index_);
                ParticleSubset* pset = posns.getParticleSubset();
                
                ParticleVariable<double> svals;
                (**diagit)(da_, patch, fieldname, matl, index_, pset, svals);
                
                for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++) {
                  double val = svals[*iter];
                  if(ipass==1)
                    _binSingleVal(bins, minval, maxval, val);
                  else {
                    minval = MIN(minval, val);
                    maxval = MAX(maxval, val);
                  }
                }
              }
            }  // patches
          } // materials
          
          if(ipass==0)
            {
              if( fabs(minval-maxval)<1.e-16 )
                {
                  if(fabs(minval)<2.e-16)
                    {
                      cout << "   WARNING: all your data seems to be at zero, adjusting the range" << endl;
                      // both zero
                      minval = -1;
                      maxval =  1;
                    }
                  else
                    {
                      double midval = (minval+maxval)/2; 
                      cout << "   WARNING: all your data seems to be at " << midval << ", adjusting the range" << endl;
                      minval = 0.9 * midval;
                      maxval = 1.1 * midval;
                    }
                }
            }
          
        } // pass
        
        string ext = (*diagit)->name();
        if(ext=="norm" || ext=="value") ext = "";
        string fname = this->fileName(fieldname+ext, "hist");
        cout << "     " << fname << endl;
        cout << "     range = " << minval << "," << maxval
             << endl;
        
        ofstream os(fname.c_str());
        os << "# time = " << time_ << ", field = " 
           << fieldname << endl;
        os << "# min = " << minval << endl;
        os << "# max = " << maxval << endl;
        
        double totcount = 1.;
        if(opts_.normalize_by_bins) {
          totcount = 0;
          for(int ibin=0;ibin<opts_.nbins;ibin++) {
            totcount += bins[ibin];
          }
        }
        
        for(int ibin=0;ibin<opts_.nbins;ibin++) {
          double xmid = minval+(ibin+0.5)*(maxval-minval)/opts_.nbins;
          os << xmid*opts_.xscale << " " << bins[ibin]/totcount << endl;
        }
        
        if(!os)
          throw InternalError("Failed to write histogram file '"+fname+"'",__FILE__,__LINE__);
        
      } // diag
  }
  
}
