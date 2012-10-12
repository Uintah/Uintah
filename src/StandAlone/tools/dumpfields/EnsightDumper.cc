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

#include "EnsightDumper.h"
#include "ScalarDiags.h"
#include "VectorDiags.h"

#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>

using namespace std;

namespace Uintah {

  
  EnsightOpts::EnsightOpts(Args & args)
  {
    withpart = args.getLogical("withpart");
    onemesh  = args.getLogical("onemesh");
    binary   = args.getLogical("bin");
  }
  
  EnsightDumper::EnsightDumper(DataArchive* da_, string basedir_, 
                               const EnsightOpts & opts,
                               const FieldSelection & fselect)
    : 
    FieldDumper(da_, basedir_), 
    nsteps_(0), flddumper_(opts.binary), 
    data_(da_,&flddumper_, opts, fselect)
  {
    data_.dir_ = this->createDirectory();
    
    // set up the file that contains a list of all the files
    string casefilename = data_.dir_ + string("/") + string("ensight.case");
    casestrm_.open(casefilename.c_str());
    if (!casestrm_) {
      cerr << "Can't open output file " << casefilename << endl;
      abort();
    }
    cout << "     " << casefilename << endl;
  
    // header
    casestrm_ << "FORMAT" << endl;
    casestrm_ << "type: ensight gold" << endl;
    casestrm_ << endl;
    casestrm_ << "GEOMETRY" << endl;
    if(data_.onemesh_)
      casestrm_ << "model:       gridgeo" << endl;
    else
      casestrm_ << "model:   1   gridgeo_****" << endl;
    casestrm_ << endl;
    casestrm_ << "VARIABLE" << endl;
  
    // time step data
    tscol_  = 0;
    tsstrm_ << "time values: ";;
  }

  EnsightDumper::~EnsightDumper()
  {
    casestrm_ << endl;
    casestrm_ << "TIME" << endl;
    casestrm_ << "time set: 1 " << endl;
    casestrm_ << "number of steps: " << nsteps_ << endl;
    casestrm_ << "filename start number: 0" << endl;
    casestrm_ << "filename increment: 1" << endl;
    casestrm_ << tsstrm_.str() << endl;
  }

  static string nodots(string n)
  {
    string r;
    for(string::const_iterator nit(n.begin());nit!=n.end();nit++)
      if(*nit!='.')
        r += *nit;
    return r;
  }

  void
  EnsightDumper::addField(string fieldname, const Uintah::TypeDescription * td)
  {
    string subtypestr, typestr;
    if(td->getType()==Uintah::TypeDescription::NCVariable)
      typestr = "node";
    else if(td->getType()==Uintah::TypeDescription::CCVariable)
      typestr = "cell";
    else
      return; // FIXME: dont dump particle data
    
    list<ScalarDiag const *> scalardiaggens = createScalarDiags(td, data_.fselect_);
    for(list<ScalarDiag const *>::const_iterator diagit(scalardiaggens.begin());
        diagit!=scalardiaggens.end();diagit++) {
      // NOTE: this must match the file name format used by the actual writer
      casestrm_ << "scalar per " << typestr << ": 1 " 
                << fieldname << "_" << (*diagit)->name()
                << " " << nodots(fieldname) << "_" << (*diagit)->name() << "_" << "****" << endl;
    }
  }
  
  EnsightDumper::Step * 
  EnsightDumper::addStep(int timestep, double time, int index)
  {
    Step * res = scinew Step(&data_, this->dirName(time, index), timestep, time, index, nsteps_);
    nsteps_++;
    return res;
  }

  void
  EnsightDumper::finishStep(FieldDumper::Step * step)
  {
    tsstrm_ << step->time_ << " ";
    if(++tscol_==10) {
      tsstrm_ << endl << "  ";
      tscol_ = 0;
    }
  }

  EnsightDumper::Step::Step(Data * data, string tsdir, int timestep, double time, int index, int fileindex)
    :
    FieldDumper::Step(tsdir, timestep, time, index, true),
    fileindex_(fileindex),
    data_(data),
    needmesh_(!data->onemesh_||(fileindex==0))
  {
  }

  void
  EnsightDumper::Step::storeGrid()
  {
    GridP grid = data_->da_->queryGrid(index_);
    FldDumper * fd = data_->dumper_;
    
    // only support level 0 for now
    int lnum = 0;
    LevelP level = grid->getLevel(lnum);
  
    // store to basename/grid.geo****
    char goutname[1024];
    char poutname[1024];
    // NOTE: this must match the file name format used in EnsightDumper::addField
    if(data_->onemesh_) {
      snprintf(goutname, 1024, "%s/gridgeo", data_->dir_.c_str());
    } else {
      snprintf(goutname, 1024, "%s/gridgeo_%04d", data_->dir_.c_str(), fileindex_);
    }
    snprintf(poutname, 1024, "%s/partgeo_%04d", data_->dir_.c_str(), fileindex_);
  
    // find ranges
    // dont bother dumping ghost stuff to ensight
    IntVector minind, maxind;
    level->findNodeIndexRange(minind, maxind);
    minind_ = minind;
    vshape_ = (maxind-minind);
    
    /*
    cout << "  " << goutname << endl;
    cout << "   minind = " << minind_ << endl;
    cout << "   maxind = " << maxind << endl;
    cout << "   vshape = " << vshape_ << endl;
    cout << endl;
    */
    
    if(needmesh_) {
    
      ofstream gstrm(goutname);
      fd->setstrm(&gstrm);
  
      if(fd->bin_) fd->textfld("C Binary",79,80);
      fd->textfld("grid description",79,80) ; fd->endl();
      fd->textfld("grid description",79,80) ; fd->endl();
      fd->textfld("node id off",     79,80) ; fd->endl();
      fd->textfld("element id off",  79,80) ; fd->endl();
  
      fd->textfld("part",            79,80) ; fd->endl();
      fd->numfld(1); fd->endl();
      fd->textfld("3d regular block",79,80); fd->endl();
      fd->textfld("block",79,80); fd->endl();
      fd->numfld(vshape_(0),10);
      fd->numfld(vshape_(1),10);
      fd->numfld(vshape_(2),10);
      fd->endl();
    
      for(int id=0;id<3;id++) {
        for(int k=minind_[2];k<maxind[2];k++) {
          for(int j=minind_[1];j<maxind[1];j++) {
            for(int i=minind_[0];i<maxind[0];i++) {
              fd->numfld(level->getNodePosition(IntVector(i,j,k))(id)) ; fd->endl();
            }
          }
        }
      }
    
      fd->unsetstrm();
    }
  
    if(data_->withpart_) {
      // store particles   
      cout << "  " << poutname << endl;
    
      ofstream pstrm(poutname);
      pstrm << "particle description" << endl;
      pstrm << "particle coordinates" << endl;
      streampos nspot = pstrm.tellp();
      pstrm << setw(8) << "XXXXXXXX" << endl;
      int nparts = 0;
  
      for(Level::const_patchIterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
    
        ConsecutiveRangeSet matls = data_->da_->queryMaterials("p.x", patch, index_);
    
        // loop over materials
        for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
            matlIter != matls.end(); matlIter++) {
          const int matl = *matlIter;
          if(!data_->fselect_.wantMaterial(matl)) continue;
          
          ParticleVariable<Matrix3> value;
          ParticleVariable<Point> partposns;
          data_->da_->query(partposns, "p.x", matl, patch, index_);
      
          ParticleSubset* pset = partposns.getParticleSubset();
          for(ParticleSubset::iterator iter = pset->begin();
              iter != pset->end(); iter++) {
        
            Point xpt = partposns[*iter];
            pstrm << setw(8) << ++nparts;
            for(int id=0;id<3;id++) {
              char b[13];
              snprintf(b, 12, "%12.5f", xpt(id));
              pstrm << b << endl;
            }
            pstrm << endl;
          }
      
        }
      }
      cout << "   nparts = " << nparts << endl;
      cout << endl;
  
      pstrm.seekp(nspot);
      pstrm << setw(8) << setfill(' ') << nparts;
    }
  
  }

  void
  EnsightDumper::Step::storeField(string fieldname, const Uintah::TypeDescription * td)
  {
    if(td->getType()==Uintah::TypeDescription::ParticleVariable) {
      if(data_->withpart_) {
        storePartField(fieldname, td);
      }
    } else {
      storeGridField(fieldname, td);
    }
  }
  
  void
  EnsightDumper::Step::storeGridField(string fieldname, const Uintah::TypeDescription * td)
  {
    GridP grid = data_->da_->queryGrid(index_);
    FldDumper * fd = data_->dumper_;
    
    list<ScalarDiag const *> scalardiaggens = createScalarDiags(td, data_->fselect_);
    
    for(list<ScalarDiag const *>::const_iterator diagit(scalardiaggens.begin());
        diagit!=scalardiaggens.end();diagit++) 
      {
        ScalarDiag const * diaggen = *diagit;
        
        char outname[1024];
        
        string diagname = diaggen->name();
        // NOTE: this must match the file name format used in EnsightDumper::addField
        snprintf(outname, 1024, "%s/%s_%s_%04d", 
                 data_->dir_.c_str(), nodots(fieldname).c_str(), 
                 diagname.c_str(), fileindex_);
        
        cout << "  " << outname;
        cout.flush();
      
        ofstream vstrm(outname);
        fd->setstrm(&vstrm);
        int icol(0); // pretty format the columns
        
        ostringstream descb;
        descb << "data field " << nodots(fieldname) << " at " << time_;
        fd->textfld(descb.str(),79,80); fd->endl();
        fd->textfld("part",     79,80); fd->endl();
        fd->numfld(1); fd->endl();
        fd->textfld("block",    79,80); fd->endl();
        
        // only support level 0 for now
        int lnum = 0;
        LevelP level = grid->getLevel(lnum);
        
        // store entire plane in field to allow correct order of writing component
        Uintah::Array3<double> vals(vshape_[0],vshape_[1],vshape_[2]);
        for(Uintah::Array3<double>::iterator vit(vals.begin());vit!=vals.end();vit++)
          *vit = 0.;
        
        cout << ", patch ";
        for(Level::const_patchIterator iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
          const Patch* patch = *iter;
          
          cout << patch->getID() << " ";
          cout.flush();
          
          IntVector ilow, ihigh;
          patch->computeVariableExtents(td->getSubType()->getType(), IntVector(0,0,0), Ghost::None, 0, ilow, ihigh);
          
          // loop over requested materials
          ConsecutiveRangeSet matls = data_->da_->queryMaterials(fieldname, patch, index_);
          for(ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++) {
            int i,j,k;
            const int matl = *matlIter;
            if(!data_->fselect_.wantMaterial(matl)) continue;
            
            // FIXME: all materials get lumped into a single field in ensight 
            NCVariable<double> matvals;
            (*diaggen)(data_->da_, patch, fieldname, matl, index_, matvals);
            for(k=ilow[2];k<ihigh[2];k++) for(j=ilow[1];j<ihigh[1];j++) for(i=ilow[0];i<ihigh[0];i++) {
              IntVector ijk(i,j,k);
              vals[ijk-minind_] += matvals[ijk];
            }
            
          } // materials
        } // patches
	
        // dump this component as text
        for(int k=0;k<vshape_[2];k++) for(int j=0;j<vshape_[1];j++) for(int i=0;i<vshape_[0];i++) {
          fd->numfld(vals[IntVector(i,j,k)]);
          if(++icol==1) 
            {
              fd->endl();
              icol = 0;
            }
          if(icol!=0) fd->endl();
        }
        cout << endl;
        
        fd->unsetstrm();
        
      } // diags  
  }
  
  void
  EnsightDumper::Step::storePartField(string /*fieldname*/, const Uintah::TypeDescription * /*td*/)
  {
    cout << "no particles for you - i spit in your general direction" << endl;
  }

}
