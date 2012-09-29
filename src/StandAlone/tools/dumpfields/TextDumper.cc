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

#include "TextDumper.h"
#include "ScalarDiags.h"
#include "VectorDiags.h"
#include "TensorDiags.h"
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>

using namespace std;

#define ONEDIM_DIM 2

namespace Uintah {
  using namespace SCIRun;
  
  TextOpts::TextOpts(Args & args)
  {
    onedim  = args.getLogical("onedim");
    tseries = args.getLogical("tseries");
  }
  
  TextDumper::TextDumper(DataArchive * da, string basedir, 
                         const TextOpts & opts, const FieldSelection & flds)
    : FieldDumper(da, basedir), opts_(opts), flds_(flds)
  {
    // set defaults for cout
    cout.setf(ios::scientific,ios::floatfield);
    cout.precision(8);
  
    // set up a file that contains a list of all the files
    string dirname = this->createDirectory();
    string filelistname = dirname+"/timelist";
  
    filelist_ = fopen(filelistname.c_str(),"w");
    if (!filelist_) {
      cerr << "Can't open output file " << filelistname << endl;
      abort();
    }
  }
  
  TextDumper::Step * 
  TextDumper::addStep(int timestep, double time, int index)
  {
    return scinew Step(this->archive(), this->dirName(time, index), 
                       timestep, time, index, opts_, flds_);
  }  

  void
  TextDumper::addField(string fieldname, const Uintah::TypeDescription * type)
  {
  }

  void
  TextDumper::finishStep(FieldDumper::Step * s)
  {
    fprintf(filelist_, "%10d %16.8g  %20s\n", s->timestep_, s->time_, s->infostr().c_str());
  }

  TextDumper::Step::Step(DataArchive * da, string tsdir,
                         int timestep, double time, int index,  
                         const TextOpts & opts, const FieldSelection & flds)
    : 
    FieldDumper::Step(tsdir, timestep, time, index),
    da_(da), opts_(opts), flds_(flds)
  {
    //    GridP grid = da_->queryGrid(time);
    GridP grid = da_->queryGrid(timestep);
  }

  static
  bool
  outside(IntVector p, IntVector mn, IntVector mx)
  {
    return  ( p[0]<mn[0] || p[0]>=mx[0] ||
              p[1]<mn[1] || p[1]>=mx[1] ||
              p[2]<mn[2] || p[2]>=mx[2] );
  }

  void
  TextDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
  {
    GridP grid = da_->queryGrid(index_);
  
    cout << "   " << fieldname << endl;
    const Uintah::TypeDescription* subtype = td->getSubType();
  
    int nmats = 0;
    // count the materials
    for(int l=0;l<=0;l++) { // FIXME: only first level
      LevelP level = grid->getLevel(l);
      for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
        const Patch* patch = *iter;
        ConsecutiveRangeSet matls= da_->queryMaterials(fieldname, patch, index_);
        for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
          int matl = *matlIter;
          if(matl>=nmats) nmats = matl+1;
        }
      }
    }
    
    TensorDiag const * tensor_preop = createTensorOp(flds_);
    
    // only support level 0 for now
    for(int l=0;l<=0;l++) {
      LevelP level = grid->getLevel(l);
    
      IntVector minind, maxind;
      level->findNodeIndexRange(minind, maxind);
      if(opts_.onedim) {
        IntVector ghostl(-minind);
        minind[0] += ghostl[0];
        maxind[0] -= ghostl[0];
        for(int id=0;id<3;id++) {
          if(id!=ONEDIM_DIM) {
            minind[id] = (maxind[id]+minind[id])/2;
            maxind[id] = minind[id]+1;
          }
        }
      }
    
      for(Level::const_patchIterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
      
        ConsecutiveRangeSet matls = da_->queryMaterials(fieldname, patch, index_);
      
        // loop over materials
        for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
            matlIter != matls.end(); matlIter++) {
          const int matl = *matlIter;
          if(!flds_.wantMaterial(matl)) continue;
          
          list<ScalarDiag const *> scalardiaggens = createScalarDiags(td, flds_, tensor_preop);
          list<VectorDiag const *> vectordiaggens = createVectorDiags(td, flds_, tensor_preop);
          list<TensorDiag const *> tensordiaggens = createTensorDiags(td, flds_, tensor_preop);
          /*
          cout << " have " << scalardiaggens.size() << " scalar diagnostics" << endl;
          cout << " have " << vectordiaggens.size() << " vector diagnostics" << endl;
          cout << " have " << tensordiaggens.size() << " tensor diagnostics" << endl;
          */
          
          // loop through requested diagnostics
          list<string> outdiags;
          for(list<ScalarDiag const *>::const_iterator diagit(scalardiaggens.begin());diagit!=scalardiaggens.end();diagit++) 
            outdiags.push_back( (*diagit)->name() );
          for(list<VectorDiag const *>::const_iterator diagit(vectordiaggens.begin());diagit!=vectordiaggens.end();diagit++) 
            outdiags.push_back( (*diagit)->name() );
          for(list<TensorDiag const *>::const_iterator diagit(tensordiaggens.begin());diagit!=tensordiaggens.end();diagit++) 
            outdiags.push_back( (*diagit)->name() );
          
          map<string, ofstream *> outfiles;
          map<string, string>     outfieldnames;
          for(list<string>::const_iterator dit(outdiags.begin());dit!=outdiags.end();dit++)
            {
              string outfieldname = fieldname;
              if(*dit!="value") {
                outfieldname += "_";
                outfieldname += *dit;
              }
              outfieldnames[*dit] = outfieldname;
              
              string fname = fileName(outfieldname, matl, "txt");
              
              outfiles[*dit];
              if(opts_.tseries || timestep_==1) {
                outfiles[*dit] = new ofstream( fname.c_str() );
                *outfiles[*dit] 
                  << "# time = " << time_ << ", field = " << fieldname << ", mat " << matl << " of " << nmats << endl;
              } else {
                outfiles[*dit] = new ofstream( fname.c_str(), ios::app);
                *outfiles[*dit] << endl;
              }
            }
          
          bool no_match = false;
          
          for(list<ScalarDiag const *>::const_iterator diagit(scalardiaggens.begin());diagit!=scalardiaggens.end();diagit++) 
            {
              ofstream & outfile = *outfiles[(*diagit)->name()];
              cout << "   " << fileName(outfieldnames[(*diagit)->name()], matl, "txt") << endl;
              
              switch(td->getType()){
              case Uintah::TypeDescription::CCVariable:
                {
                  CCVariable<double> svals;
                  (**diagit)(da_, patch, fieldname, matl, index_, svals);
                  
                  for(NodeIterator iter = patch->getNodeIterator();
                      !iter.done(); iter++){
                    if(outside(*iter, minind, maxind)) continue;
                    Point xpt = patch->nodePosition(*iter);
                    if(opts_.tseries) outfile << time_ << " ";
                    outfile << xpt(0) << " " 
                            << xpt(1) << " " 
                            << xpt(2) << " ";
                    outfile << svals[*iter] << " "
                            << endl;
                  }
                } break;
              case Uintah::TypeDescription::NCVariable:
                {
                  NCVariable<double> svals;
                  (**diagit)(da_, patch, fieldname, matl, index_, svals);
                  
                  for(NodeIterator iter = patch->getNodeIterator();
                      !iter.done(); iter++){
                    if(outside(*iter, minind, maxind)) continue;
                    Point xpt = patch->nodePosition(*iter);
                    if(opts_.tseries) outfile << time_ << " ";
                    outfile << xpt(0) << " " 
                            << xpt(1) << " " 
                            << xpt(2) << " ";
                    outfile << svals[*iter] << " "
                            << endl;
                  }
                } break;
              case Uintah::TypeDescription::ParticleVariable:
                {
                  ParticleVariable<Point> posns;
                  da_->query(posns, "p.x", matl, patch, index_);
                  
                  ParticleSubset* pset = posns.getParticleSubset();
                  
                  ParticleVariable<double> svals;
                  (**diagit)(da_, patch, fieldname, matl, index_, pset, svals);
                  
                  for(ParticleSubset::iterator iter = pset->begin();
                      iter != pset->end(); iter++) {
                    Point xpt = posns[*iter];
                    if(opts_.tseries) outfile << time_ << " ";
                    outfile << xpt(0) << " " 
                            << xpt(1) << " " 
                            << xpt(2) << " ";
                    outfile << (svals[*iter]) << " " 
                            << endl;
                  }
                } break;
              default:
                no_match = true;
              } // td->getType() switch
              
            } // scalar diags
          
          for(list<VectorDiag const *>::const_iterator diagit(vectordiaggens.begin());diagit!=vectordiaggens.end();diagit++) 
            {
              ofstream & outfile = *outfiles[(*diagit)->name()];
              cout << "   " << fileName(outfieldnames[(*diagit)->name()], matl, "txt") << endl;
              
              switch(td->getType()){
              case Uintah::TypeDescription::CCVariable:
                {
                  CCVariable<Vector> vvals;
                  (**diagit)(da_, patch, fieldname, matl, index_, vvals);
                  
                  for(NodeIterator iter = patch->getNodeIterator();
                      !iter.done(); iter++){
                    if(outside(*iter, minind, maxind)) continue;
                    Point xpt = patch->nodePosition(*iter);
                    if(opts_.tseries) outfile << time_ << " ";
                    outfile << xpt(0) << " " 
                            << xpt(1) << " " 
                            << xpt(2) << " ";
                    outfile << vvals[*iter][0] << " "
                            << vvals[*iter][1] << " "
                            << vvals[*iter][2] << " "
                            << endl;
                  }
                } break;
              case Uintah::TypeDescription::NCVariable:
                {
                  NCVariable<Vector> vvals;
                  (**diagit)(da_, patch, fieldname, matl, index_, vvals);
                  
                  for(NodeIterator iter = patch->getNodeIterator();
                      !iter.done(); iter++){
                    if(outside(*iter, minind, maxind)) continue;
                    Point xpt = patch->nodePosition(*iter);
                    if(opts_.tseries) outfile << time_ << " ";
                    outfile << xpt(0) << " " 
                            << xpt(1) << " " 
                            << xpt(2) << " ";
                    outfile << vvals[*iter][0] << " "
                            << vvals[*iter][1] << " "
                            << vvals[*iter][2] << " "
                            << endl;
                  }
                } break;
              case Uintah::TypeDescription::ParticleVariable:
                {
                  ParticleVariable<Point> posns;
                  da_->query(posns, "p.x", matl, patch, index_);
                  
                  ParticleSubset* pset = posns.getParticleSubset();
                  
                  ParticleVariable<Vector> vvals;
                  (**diagit)(da_, patch, fieldname, matl, index_, pset, vvals);
                  
                  for(ParticleSubset::iterator iter = pset->begin();
                      iter != pset->end(); iter++) {
                    Point xpt = posns[*iter];
                    if(opts_.tseries) outfile << time_ << " ";
                    outfile << xpt(0) << " " 
                            << xpt(1) << " " 
                            << xpt(2) << " ";
                    outfile << vvals[*iter][0] << " " 
                            << vvals[*iter][1] << " " 
                            << vvals[*iter][2] << " " 
                            << endl;
                  }
                } break;
              default:
                no_match = true;
              } // td->getType() switch
              
            } // vector diag
          
          for(list<TensorDiag const *>::const_iterator diagit(tensordiaggens.begin());diagit!=tensordiaggens.end();diagit++) 
            {
              ofstream & outfile = *outfiles[(*diagit)->name()];
              cout << "   " << fileName(outfieldnames[(*diagit)->name()], matl, "txt") << endl;
              
              switch(td->getType()){
              case Uintah::TypeDescription::CCVariable:
                {
                  CCVariable<Matrix3> tvals;
                  (**diagit)(da_, patch, fieldname, matl, index_, tvals);
                  
                  for(NodeIterator iter = patch->getNodeIterator();
                      !iter.done(); iter++){
                    if(outside(*iter, minind, maxind)) continue;
                    Point xpt = patch->nodePosition(*iter);
                    if(opts_.tseries) outfile << time_ << " ";
                    outfile << xpt(0) << " " 
                            << xpt(1) << " " 
                            << xpt(2) << " ";
                    outfile << tvals[*iter](0,0) << " "
                            << tvals[*iter](0,1) << " "
                            << tvals[*iter](0,2) << " "
                            << tvals[*iter](1,0) << " "
                            << tvals[*iter](1,1) << " "
                            << tvals[*iter](1,2) << " "
                            << tvals[*iter](2,0) << " "
                            << tvals[*iter](2,1) << " "
                            << tvals[*iter](2,2) << " "
                            << endl;
                  }
                } break;
              case Uintah::TypeDescription::NCVariable:
                {
                  NCVariable<Matrix3> tvals;
                  (**diagit)(da_, patch, fieldname, matl, index_, tvals);
                  
                  for(NodeIterator iter = patch->getNodeIterator();
                      !iter.done(); iter++){
                    if(outside(*iter, minind, maxind)) continue;
                    Point xpt = patch->nodePosition(*iter);
                    if(opts_.tseries) outfile << time_ << " ";
                    outfile << xpt(0) << " " 
                            << xpt(1) << " " 
                            << xpt(2) << " ";
                    outfile << tvals[*iter](0,0) << " "
                            << tvals[*iter](0,1) << " "
                            << tvals[*iter](0,2) << " "
                            << tvals[*iter](1,0) << " "
                            << tvals[*iter](1,1) << " "
                            << tvals[*iter](1,2) << " "
                            << tvals[*iter](2,0) << " "
                            << tvals[*iter](2,1) << " "
                            << tvals[*iter](2,2) << " "
                            << endl;
                  }
                } break;
              case Uintah::TypeDescription::ParticleVariable:
                {
                  ParticleVariable<Point> posns;
                  da_->query(posns, "p.x", matl, patch, index_);
                  
                  ParticleSubset* pset = posns.getParticleSubset();
                  
                  ParticleVariable<Matrix3> tvals;
                  (**diagit)(da_, patch, fieldname, matl, index_, pset, tvals);
                  
                  for(ParticleSubset::iterator iter = pset->begin();
                      iter != pset->end(); iter++) {
                    Point xpt = posns[*iter];
                    if(opts_.tseries) outfile << time_ << " ";
                    outfile << xpt(0) << " " 
                            << xpt(1) << " " 
                            << xpt(2) << " ";
                    outfile << tvals[*iter](0,0) << " "
                            << tvals[*iter](0,1) << " "
                            << tvals[*iter](0,2) << " "
                            << tvals[*iter](1,0) << " "
                            << tvals[*iter](1,1) << " "
                            << tvals[*iter](1,2) << " "
                            << tvals[*iter](2,0) << " "
                            << tvals[*iter](2,1) << " "
                            << tvals[*iter](2,2) << " "
                            << endl;
                  }
                } break;
              default:
                no_match = true;
              } // td->getType() switch
              
            } // vectort diag
          
          for(map<string, ofstream *>::iterator fit(outfiles.begin());fit!=outfiles.end();fit++)
            {
              delete fit->second;
            }
          
          if (no_match)
            cerr << "WARNING: Unexpected type for " << td->getName() << " of " << subtype->getName() << endl;
        
        } // materials
      } // patches
    } // levels 
  }    

  
}
