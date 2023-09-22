/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#include "TextDumper.h"
#include "ScalarDiags.h"
#include "VectorDiags.h"
#include "TensorDiags.h"
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>

using namespace std;

#define ONEDIM_DIM 2

namespace Uintah {

  TextOpts::TextOpts(Args & args)
  {
    onedim  = args.getLogical("onedim");
    tseries = args.getLogical("tseries");
  }
  //______________________________________________________________________
  //
  TextDumper::TextDumper(DataArchive          * da,
                         string basedir,
                         const TextOpts       & opts,
                         const FieldSelection & flds)
    : FieldDumper(da, basedir), opts_(opts), flds_(flds)
  {
    // set defaults for cout
    cout.setf(ios::scientific,ios::floatfield);
    cout.precision(16);

    // set up a file that contains a list of all the files
    string dirname = this->createDirectory();
    string filelistname = dirname+"/timelist";

    filelist_ = fopen(filelistname.c_str(),"w");
    fprintf(filelist_, "%10s : %16s  : %20s\n","Timestep", "Time[sec]", "Directory");
    if (!filelist_) {
      cerr << "Can't open output file " << filelistname << endl;
      abort();
    }
  }
  //______________________________________________________________________
  //
  TextDumper::~TextDumper()
  {
    fclose(filelist_);
  }
  //______________________________________________________________________
  //
  TextDumper::Step *
  TextDumper::addStep(int timestep, double time, int index)
  {
    const string dirName   = this->dirName(time, timestep);
    DataArchive * da = this->archive();

    cout << " dirName: " << dirName  << endl;

    return scinew Step( da, dirName,
                       timestep, time, index, opts_, flds_);
  }

  //______________________________________________________________________
  //
  void
  TextDumper::addField(string fieldname, const Uintah::TypeDescription * type)
  {
  }
  //______________________________________________________________________
  //
  void
  TextDumper::finishStep(FieldDumper::Step * s)
  {
    fprintf(filelist_, "%10d : %16.8g  : %20s\n", s->timestep_, s->time_, s->infostr().c_str());
    fflush(filelist_);
  }

  //______________________________________________________________________
  //
  TextDumper::Step::Step(DataArchive * da, string tsdir,
                         int timestep, double time, int index,
                         const TextOpts & opts, const FieldSelection & flds)
    :
    FieldDumper::Step(tsdir, timestep, time, index),
    da_(da), opts_(opts), flds_(flds)
  {
    //    GridP grid = da_->queryGrid(time);
    GridP grid = da_->queryGrid( index_ );
  }

  //______________________________________________________________________
  //
  static
  bool
  outside(IntVector p, IntVector mn, IntVector mx)
  {
    return  ( p[0]<mn[0] || p[0]>=mx[0] ||
              p[1]<mn[1] || p[1]>=mx[1] ||
              p[2]<mn[2] || p[2]>=mx[2] );
  }

  //______________________________________________________________________
  //
  void
  TextDumper::Step::storeGrid()
  {
    //    GridP grid = da_->queryGrid(time);
    GridP grid = da_->queryGrid( index_ );

    string fname = tsdir_+"/"+outdir_ + "grid.txt";
    ofstream outfile = ofstream( fname.c_str(), ios::app);
    outfile << " Number of levels     : " << grid->numLevels() << endl;

    BBox b;
    grid->getInteriorSpatialRange(b);
    outfile << " Computational Domain : " << b << endl;

    for(int l=0;l<=0;l++) { // FIXME: only first level
      LevelP level = grid->getLevel(l);
      level->getInteriorSpatialRange(b);

      outfile << "#__________________________________L-" << l << endl;
      outfile << " Has coarser level  : " << level->hasCoarserLevel() << endl;
      outfile << " Has finer level    : " << level->hasFinerLevel() << endl;
      outfile << " Is non-cubic level : " << level->isNonCubic() << endl;
      outfile << " NumPatches         : " << level->numPatches() << endl;
      outfile << " TotalCells         : " << level->totalCells() << endl;
      outfile << " Level spatial range: " << b << endl;
      outfile << " Cell spacing       : " << level->dCell() << endl;

      IntVector low;
      IntVector high;
      level->findInteriorCellIndexRange(low, high);
      outfile << " Interior cell range: " << low << "," << high << endl;

      level->findInteriorNodeIndexRange(low, high);
      outfile << " Interior node range: " << low << "," << high << endl;
    }
  }

  //______________________________________________________________________
  //
  bool
  TextDumper::Step::isValidType (const Uintah::TypeDescription * td)
  {


    bool value = true;
    switch( td->getType() ){
      case Uintah::TypeDescription::CCVariable:
        break;
      case Uintah::TypeDescription::NCVariable:
        break;
      case Uintah::TypeDescription::ParticleVariable:
        break;
      default:
        {
        value = false;
        return value;
        }
    }

    const Uintah::TypeDescription* subtype = td->getSubType();
    switch( subtype->getType() ){
      case Uintah::TypeDescription::double_type:
        break;
      case Uintah::TypeDescription::Vector:
        break;
      case Uintah::TypeDescription::Matrix3:
        break;
      default:
        {
        value = false;
        return value;
        }
    }
    return value;
  }
  //______________________________________________________________________
  //
  void
  TextDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
  {

    if( !isValidType(td) ){
      cout << "    Invalid variable type, not processing ("<< fieldname << ")\n";
      return;
    }
    static int count=1;

    GridP grid = da_->queryGrid(index_);

    cout << "   " << fieldname << endl;
    const Uintah::TypeDescription* subtype = td->getSubType();

    //__________________________________
    //  count the number of matls
    int nmats = 0;

    for(int l=0;l<=0;l++) { // FIXME: only first level
      LevelP level = grid->getLevel(l);

      for(Level::const_patch_iterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
        const Patch* patch = *iter;
        ConsecutiveRangeSet matls= da_->queryMaterials(fieldname, patch, index_);

        for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
          int matl = *matlIter;
          if(matl>=nmats){
            nmats = matl+1;
          }
        }
      }
    }

    TensorDiag const * tensor_preop = createTensorOp(flds_);

    //__________________________________
    // find level extents
    // only support level 0 for now
    for(int l=0;l<=0;l++) {
      LevelP level = grid->getLevel(l);

      IntVector levelLowIndx;
      IntVector levelHighIndx;
      level->findNodeIndexRange(levelLowIndx, levelHighIndx);

      if(opts_.onedim) {
        IntVector ghostl(-levelLowIndx);
        levelLowIndx[0] += ghostl[0];
        levelHighIndx[0] -= ghostl[0];

        for(int id=0;id<3;id++) {
          if(id!=ONEDIM_DIM) {
            levelLowIndx[id] = (levelHighIndx[id] + levelLowIndx[id])/2;
            levelHighIndx[id] = levelLowIndx[id]+1;
          }
        }
      }

      //__________________________________
      //  Patch loop
      for(Level::const_patch_iterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;

        const int patchID = patch->getID();
        //__________________________________
        // loop over materials
        ConsecutiveRangeSet matls = da_->queryMaterials(fieldname, patch, index_);

        for(ConsecutiveRangeSet::iterator matlIter = matls.begin(); matlIter != matls.end(); matlIter++) {
          const int matl = *matlIter;

          if( !flds_.wantMaterial(matl) ){
           continue;
          }

          list<ScalarDiag const *> scalarDiagGens = createScalarDiags(td, flds_, tensor_preop);
          list<VectorDiag const *> vectorDiagGens = createVectorDiags(td, flds_, tensor_preop);
          list<TensorDiag const *> tensorDiagGens = createTensorDiags(td, flds_, tensor_preop);
          /*
          cout << " have " << scalarDiagGens.size() << " scalar diagnostics" << endl;
          cout << " have " << vectorDiagGens.size() << " vector diagnostics" << endl;
          cout << " have " << tensorDiagGens.size() << " tensor diagnostics" << endl;
          */

          // loop through requested diagnostics
          list<string> outdiags;
          for(list<ScalarDiag const *>::const_iterator iter=scalarDiagGens.begin();iter!=scalarDiagGens.end();iter++){
            outdiags.push_back( (*iter)->name() );
          }

          for(list<VectorDiag const *>::const_iterator iter=vectorDiagGens.begin();iter!=vectorDiagGens.end();iter++) {
            outdiags.push_back( (*iter)->name() );
          }

          for(list<TensorDiag const *>::const_iterator iter=tensorDiagGens.begin();iter!=tensorDiagGens.end();iter++){
            outdiags.push_back( (*iter)->name() );
          }

          //__________________________________
          //  Open all the files and write a header
          map<string, ofstream *> outfiles;
          map<string, string>     outFileNames;
          map<string, string>     outfieldnames;

          for(list<string>::const_iterator iter=outdiags.begin();iter!=outdiags.end();iter++){
            string diagName     = *iter;
            string outfieldname = fieldname;

            if( diagName!="value" ) {
              outfieldname += "_";
              outfieldname += diagName;
            }

            outfieldnames[diagName] = outfieldname;

            string fname = fileName(outfieldname, matl, "txt");

            outFileNames[diagName] = fname;

            outfiles[diagName];
            outfiles[diagName] = new ofstream( fname.c_str(), ios::app);

            if( patchID == 0 ){
              *outfiles[diagName] << "# time = " << time_ << ", field = " << fieldname << ", mat " << matl << " of " << nmats << endl;
            }
          }

          bool no_match = false;

          //__________________________________
          //                         DOUBLE
          for(list<ScalarDiag const *>::const_iterator diagit=scalarDiagGens.begin();diagit!=scalarDiagGens.end();diagit++) {

              string diagName = (*diagit)->name();

              ofstream & outfile = *outfiles[diagName];
              string outFileName = outFileNames[diagName];

              outfile.setf(ios::scientific,ios::floatfield);
              outfile.precision(16);

              switch(td->getType()){
              case Uintah::TypeDescription::CCVariable:
                {
                  CCVariable<double> svals;
                  (**diagit)(da_, patch, fieldname, matl, index_, svals);

                  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){

                    IntVector c = *iter;
                    if( outside(c, levelLowIndx, levelHighIndx) ){
                      continue;
                    }

                    Point pos = patch->cellPosition(c);

                    if(opts_.tseries) {
                      outfile << time_ << ",";
                    }

                    outfile << pos(0) << ","
                            << pos(1) << ","
                            << pos(2) << ",";
                    outfile << svals[c] << endl;
                  }
                } break;
              case Uintah::TypeDescription::NCVariable:
                {
                  NCVariable<double> svals;
                  (**diagit)(da_, patch, fieldname, matl, index_, svals);

                  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){

                    IntVector n = *iter;
                    if( outside(n, levelLowIndx, levelHighIndx) ){
                     continue;
                    }

                    Point pos = patch->nodePosition(n);

                    if(opts_.tseries){
                      outfile << time_ << ",";
                    }

                    outfile << pos(0) << ","
                            << pos(1) << ","
                            << pos(2) << ",";
                    outfile << svals[n] << endl;
                  }
                } break;
              case Uintah::TypeDescription::ParticleVariable:
                {
                  ParticleVariable<Point> pos;
                  da_->query(pos, "p.x", matl, patch, index_);

                  ParticleSubset* pset = pos.getParticleSubset();

                  ParticleVariable<double> svals;
                  (**diagit)(da_, patch, fieldname, matl, index_, pset, svals);
                  
                  bool pID_exists {false};
                  ParticleVariable<long64> pID;
                  pID_exists = da_->query( pID,"p.particleID", matl, patch, index_ );

                  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
                    particleIndex idx = *iter;

                    Point p = pos[idx];

                    if(opts_.tseries){
                      outfile << time_ << ",";
                    }

                    outfile << p(0) << ","
                            << p(1) << ","
                            << p(2) << ",";
                    
                    if( pID_exists ) {
                      outfile << (pID[idx]) << ",";
                    }
                    
                    outfile << (svals[idx]) << endl;

                  }
                } break;
              default:
                {
                  no_match = true;
                  std::remove( outFileName.c_str() );
                }

              } // td->getType() switch

            } // scalar diags

            //__________________________________
            //                       VECTOR
            for(list<VectorDiag const *>::const_iterator diagit=vectorDiagGens.begin();diagit!=vectorDiagGens.end();diagit++) {

              string diagName = (*diagit)->name();

              ofstream & outfile = *outfiles[diagName];
              string outFileName = outFileNames[diagName];

              outfile.setf(ios::scientific,ios::floatfield);
              outfile.precision(16);

              switch(td->getType()){
              case Uintah::TypeDescription::CCVariable:
                {
                  CCVariable<Vector> vvals;
                  (**diagit)(da_, patch, fieldname, matl, index_, vvals);

                  for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){

                    IntVector c = *iter;
                    if( outside(c, levelLowIndx, levelHighIndx) ){
                      continue;
                    }

                    Point pos = patch->cellPosition(c);
                    if(opts_.tseries){
                      outfile << time_ << ",";
                    }
                    outfile << pos(0) << ","
                            << pos(1) << ","
                            << pos(2) << ",";
                    outfile << vvals[c][0] << ","
                            << vvals[c][1] << ","
                            << vvals[c][2] << endl;
                  }
                } break;
              case Uintah::TypeDescription::NCVariable:
                {
                  NCVariable<Vector> vvals;
                  (**diagit)(da_, patch, fieldname, matl, index_, vvals);

                  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){

                    IntVector n = *iter;
                    if( outside(n, levelLowIndx, levelHighIndx)  ){
                      continue;
                    }

                    Point pos = patch->nodePosition(n);

                    if(opts_.tseries){
                      outfile << time_ << ",";
                    }

                    outfile << pos(0) << ","
                            << pos(1) << ","
                            << pos(2) << ",";
                    outfile << vvals[n][0] << ","
                            << vvals[n][1] << ","
                            << vvals[n][2] << endl;
                  }
                } break;
              case Uintah::TypeDescription::ParticleVariable:
                {
                  ParticleVariable<Point> pos;
                  da_->query(pos, "p.x", matl, patch, index_);

                  ParticleSubset* pset = pos.getParticleSubset();

                  ParticleVariable<Vector> vvals;
                  (**diagit)(da_, patch, fieldname, matl, index_, pset, vvals);

                  bool pID_exists {false};
                  ParticleVariable<long64> pID;
                  pID_exists = da_->query( pID,"p.particleID", matl, patch, index_ );

                  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {

                    particleIndex idx = *iter;

                    Point p = pos[idx];

                    if(opts_.tseries){
                      outfile << time_ << ",";
                    }
                    outfile << p(0) << ","
                            << p(1) << ","
                            << p(2) << ",";
                    
                    if( pID_exists ) {
                      outfile << (pID[idx]) << ",";
                    }
                    
                    outfile << vvals[idx][0] << ","
                            << vvals[idx][1] << ","
                            << vvals[idx][2] << endl;
                  }
                } break;
              default:
                {
                  no_match = true;
                  std::remove( outFileName.c_str() );
                }
              } // td->getType() switch

            } // vector diag


            //__________________________________
            //                         MATRIX3

            for(list<TensorDiag const *>::const_iterator diagit=tensorDiagGens.begin();diagit!=tensorDiagGens.end();diagit++){
              string diagName = (*diagit)->name();

              ofstream & outfile = *outfiles[diagName];
              string outFileName = outFileNames[diagName];

              outfile.setf(ios::scientific,ios::floatfield);
              outfile.precision(16);

              switch(td->getType()){
              case Uintah::TypeDescription::CCVariable:
                {
                  CCVariable<Matrix3> tvals;
                  (**diagit)(da_, patch, fieldname, matl, index_, tvals);

                  for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
                    IntVector n = *iter;

                    if( outside(n, levelLowIndx, levelHighIndx) ){
                      continue;
                    }

                    Point pos = patch->cellPosition(n);

                    if(opts_.tseries){
                      outfile << time_ << ",";
                    }
                    outfile << pos(0) << ","
                            << pos(1) << ","
                            << pos(2) << ",";
                    outfile << tvals[n](0,0) << ","
                            << tvals[n](0,1) << ","
                            << tvals[n](0,2) << ","
                            << tvals[n](1,0) << ","
                            << tvals[n](1,1) << ","
                            << tvals[n](1,2) << ","
                            << tvals[n](2,0) << ","
                            << tvals[n](2,1) << ","
                            << tvals[n](2,2) << endl;
                  }
                } break;
              case Uintah::TypeDescription::NCVariable:
                {
                  NCVariable<Matrix3> tvals;
                  (**diagit)(da_, patch, fieldname, matl, index_, tvals);

                  for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
                    IntVector n = *iter;

                    if( outside(n, levelLowIndx, levelHighIndx) ){
                      continue;
                    }

                    Point pos = patch->nodePosition(n);

                    if(opts_.tseries){
                      outfile << time_ << ",";
                    }

                    outfile << pos(0) << ","
                            << pos(1) << ","
                            << pos(2) << ",";
                    outfile << tvals[n](0,0) << ","
                            << tvals[n](0,1) << ","
                            << tvals[n](0,2) << ","
                            << tvals[n](1,0) << ","
                            << tvals[n](1,1) << ","
                            << tvals[n](1,2) << ","
                            << tvals[n](2,0) << ","
                            << tvals[n](2,1) << ","
                            << tvals[n](2,2) << endl;
                  }
                } break;
              case Uintah::TypeDescription::ParticleVariable:
                {
                  ParticleVariable<Point> pos;
                  da_->query(pos, "p.x", matl, patch, index_);

                  ParticleSubset* pset = pos.getParticleSubset();

                  ParticleVariable<Matrix3> tvals;
                  (**diagit)(da_, patch, fieldname, matl, index_, pset, tvals);

                  bool pID_exists {false};
                  ParticleVariable<long64> pID;
                  pID_exists = da_->query( pID,"p.particleID", matl, patch, index_ );

                  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
                    particleIndex idx = *iter;

                    Point p = pos[idx];

                    if(opts_.tseries){
                      outfile << time_ << ",";
                    }

                    outfile << p(0) << ","
                            << p(1) << ","
                            << p(2) << ",";
                    
                    if( pID_exists ) {
                      outfile << (pID[idx]) << ",";
                    }
                    
                    outfile << tvals[idx](0,0) << ","
                            << tvals[idx](0,1) << ","
                            << tvals[idx](0,2) << ","
                            << tvals[idx](1,0) << ","
                            << tvals[idx](1,1) << ","
                            << tvals[idx](1,2) << ","
                            << tvals[idx](2,0) << ","
                            << tvals[idx](2,1) << ","
                            << tvals[idx](2,2) << endl;
                  }
                } break;
              default:
                {
                  no_match = true;
                  std::remove( outFileName.c_str() );
                }
              } // td->getType() switch
            } // tensor diag

            for(map<string, ofstream *>::iterator iter=outfiles.begin();iter!=outfiles.end();iter++){
              delete iter->second;
            }

          if (no_match && count == 1){
            cerr << "WARNING: Unexpected type for " << td->getName() << " of " << subtype->getName() << endl;
            count ++;
          }
        } // materials
      } // patches
    } // levels
  }


}
