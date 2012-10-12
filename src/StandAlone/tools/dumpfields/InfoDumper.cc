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

#include "InfoDumper.h"
#include "ScalarDiags.h"
#include "TensorDiags.h"

#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <iomanip>
#include <sstream>
using namespace std;

static string _rngtxt(double v1, double v2)
{
  ostringstream b;
  b << "(" 
    << setw(12) << setprecision(6) << v1 << ","
    << setw(12) << setprecision(6) << v2 << ")";
  return b.str();
}

namespace Uintah {
  
  InfoOpts::InfoOpts(Args & args)
  {
    showeachmat = args.getLogical("allmats");
  }
  
  InfoDumper::InfoDumper(DataArchive* da, string basedir,
                         const InfoOpts & opts, const FieldSelection & fselect)
    : FieldDumper(da, basedir), opts_(opts), fselect_(fselect)
  {
  }

  InfoDumper::Step * 
  InfoDumper::addStep(int timestep, double time, int index)
  {
    return scinew Step(this->archive(), this->dirName(time, index), timestep, time, index, opts_, fselect_);
  }

  void
  InfoDumper::finishStep(FieldDumper::Step * s)
  {
  }

  InfoDumper::Step::Step(DataArchive * da, string tsdir, int timestep, double time, int index, 
                         const InfoOpts & opts, const FieldSelection & fselect)
    : 
    FieldDumper::Step(tsdir, timestep, time, index, true), da_(da), opts_(opts), fselect_(fselect)
  {
  }
  
  void
  InfoDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
  {
    if(!fselect_.wantField(fieldname)) return;
    cout << "  " << fieldname << endl;
    
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
    
    TensorDiag const * tensor_preop = createTensorOp(fselect_);
    list<ScalarDiag const *> scalardiaggens = createScalarDiags(td, fselect_, tensor_preop);
    for(list<ScalarDiag const *>::const_iterator diagit(scalardiaggens.begin());
        diagit!=scalardiaggens.end();diagit++) 
      {
        double avgval    = 0.;
        
        LevelP level = grid->getLevel(0);
        
        // calculating the standard deviation doubles the processing time :-/
        for(int ipass=0;ipass<2;ipass++) {
          
        // junk being collected for this diagnostic
        double minval   =+FLT_MAX;
        double maxval   =-FLT_MAX;
        double minabsval=+FLT_MAX;
        double maxabsval=-FLT_MAX;
        double sumval   =0.;
        double valcount =0.;
        double sumdiffsq = 0.;
        
        for(vector<int>::const_iterator mit(mats.begin());mit!=mats.end();mit++) {
          int matl = *mit;
          
          // junk being collected for this material
          double mat_minval   =+FLT_MAX;
          double mat_maxval   =-FLT_MAX;
          double mat_minabsval=+FLT_MAX;
          double mat_maxabsval=-FLT_MAX;
          double mat_sumval   =0.;
          double mat_valcount =0.;
          
          for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
            const Patch* patch = *iter;
            
            if(td->getType()==Uintah::TypeDescription::CCVariable) {
              
              CCVariable<double> svals;
              (**diagit)(da_, patch, fieldname, matl, index_, svals);
              
              for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
                double val = svals[*iter];
                
                if(val<mat_minval) mat_minval = val;
                if(val>mat_maxval) mat_maxval = val;
                if(fabs(val)<mat_minabsval) mat_minabsval = fabs(val);
                if(fabs(val)>mat_maxabsval) mat_maxabsval = fabs(val);
                
                mat_sumval   += val;
                mat_valcount += 1;
                
                sumdiffsq += pow(val-avgval, 2.0);
              }
            } else if(td->getType()==Uintah::TypeDescription::NCVariable) {
              
              NCVariable<double> svals;
              (**diagit)(da_, patch, fieldname, matl, index_, svals);
              
              for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
                double val = svals[*iter];
                
                if(val<mat_minval) mat_minval = val;
                if(val>mat_maxval) mat_maxval = val;
                if(fabs(val)<mat_minabsval) mat_minabsval = fabs(val);
                if(fabs(val)>mat_maxabsval) mat_maxabsval = fabs(val);
                
                mat_sumval   += val;
                mat_valcount += 1;
                
                sumdiffsq += pow(val-avgval, 2.0);
              }
            } else if (td->getType()==Uintah::TypeDescription::ParticleVariable) {
              ParticleVariable<Point> posns;
              da_->query(posns, "p.x", matl, patch, index_);
              ParticleSubset* pset = posns.getParticleSubset();
              
              ParticleVariable<double> svals;
              (**diagit)(da_, patch, fieldname, matl, index_, pset, svals);
              
              for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++) {
                double val = svals[*iter];
                
                if(val<mat_minval) mat_minval = val;
                if(val>mat_maxval) mat_maxval = val;
                if(fabs(val)<mat_minabsval) mat_minabsval = fabs(val);
                if(fabs(val)>mat_maxabsval) mat_maxabsval = fabs(val);
                
                mat_sumval   += val;
                mat_valcount += 1;
                
                sumdiffsq += pow(val-avgval, 2.0);
              }
            }
          } // patches
          
          // if no points found for this material, dont write empty information
          if(mat_valcount<0.5) continue;
          
          // may want more column formatted output here ...
          if(opts_.showeachmat && ipass==1) {
            cout << "      " << (*diagit)->name() <<  " mat " << matl << ", "
                 << "range = " << _rngtxt(mat_minval,mat_maxval) << ", "
                 << "absolute range = " << _rngtxt(mat_minabsval,mat_maxabsval) << ", "
                 << "average = " << mat_sumval/mat_valcount
                 << endl;
          }
          
          if(mat_minval<minval) minval = mat_minval;
          if(mat_maxval>maxval) maxval = mat_maxval;
          if(mat_minabsval<minabsval) minabsval = mat_minabsval;
          if(mat_maxabsval>maxabsval) maxabsval = mat_maxabsval;
          
          sumval   += mat_sumval;
          valcount += mat_valcount;
          
        } // materials
        
        if(valcount<.5) continue; // dont write if never found a value
        
        // may want more column formatted output here ...
        
        if(ipass==0)
          avgval = sumval/valcount;
        else
          cout << "      " << (*diagit)->name() << ",       "
               << "range = " << _rngtxt(minval,maxval) << ", "
               << "absolute range = " << _rngtxt(minabsval,maxabsval) << ", "
               << "average = " << sumval/valcount << ", "
               << "standard deviation = " << sqrt( 1./(valcount-1)*sumdiffsq )
             << endl;
        } // pass
        
      } // diagnostic
  }
  
}
