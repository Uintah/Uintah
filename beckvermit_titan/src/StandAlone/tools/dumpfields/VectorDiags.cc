/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include "VectorDiags.h"
#include "TensorDiags.h"

#include <string>
#include <map>
#include <list>

#ifdef _WIN32
#define snprintf _snprintf
#endif

using namespace std;

namespace Uintah {
  using namespace SCIRun;
  
  VectorDiag::~VectorDiag() {}
  
  // --------------------------------------------------------------------------
  
  class VectorValueDiag : public VectorDiag {
  public:
    VectorValueDiag() {}
    string name() const { return "value"; }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, int index, 
                    NCVariable<Vector> & res) const {
      da->query(res, fieldname, imat, patch, index);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, int index, 
                    CCVariable<Vector> & res) const {
      da->query(res, fieldname, imat, patch, index);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, int index, 
                    ParticleSubset * parts,
                    ParticleVariable<Vector> & res) const {
      da->query(res, fieldname, imat, patch, index);
    }
  };
  
  // --------------------------------------------------------------------------
  
  class TensorToVectorDiag : public VectorDiag {
  public:
    virtual ~TensorToVectorDiag() {}
    
    virtual Vector reduce(const Matrix3 & v) const = 0;
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, int index, 
                    NCVariable<Vector> & res) const {
      NCVariable<Matrix3> value;
      da->query(value, fieldname, imat, patch, index);
      res.allocate(value.getLowIndex(), value.getHighIndex());
      for(NCVariable<Matrix3>::iterator it=value.begin();it!=value.end();it++)
        res[it.getIndex()] = reduce(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, int index, 
                    CCVariable<Vector> & res) const {
      CCVariable<Matrix3> value;
      da->query(value, fieldname, imat, patch, index);
      res.allocate(value.getLowIndex(), value.getHighIndex());
      for(CCVariable<Matrix3>::iterator it=value.begin();it!=value.end();it++)
        res[it.getIndex()] = reduce(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, int index, 
                    ParticleSubset * parts,
                    ParticleVariable<Vector> & res) const {
      ParticleVariable<Matrix3> value;
      da->query(value, fieldname, imat, patch, index);
      res.allocate(parts);
      for(ParticleSubset::iterator pit(parts->begin());pit!=parts->end();pit++)
        res[*pit] = reduce(value[*pit]);
    }
  };
  
  
  class TensorCompDiag : public TensorToVectorDiag {
  public:
    TensorCompDiag(char ic) : ic_(ic) {
      snprintf(name_, 16, "component%d", ic_);
    }
    string name() const { return name_;}
    Vector reduce(const Matrix3 & v) const { 
      return Vector(v(ic_,0), v(ic_,1), v(ic_,2)); 
    }
  private:
    char name_[16];
    char ic_;
  };
  
  class TensorMaxEigenvectDiag : public TensorToVectorDiag {
  public:
    TensorMaxEigenvectDiag() {}
    string name() const { return "maxeigenvect"; }
    Vector reduce(const Matrix3 & v) const { 
      double e1, e2, e3;
      v.getEigenValues(e1, e2, e3); 
      return v.getEigenVectors(e1, e1)[0];
    }
  };

  class TensorMidEigenvectDiag : public TensorToVectorDiag {
  public:
    TensorMidEigenvectDiag() {}
    string name() const { return "mideigenvect"; }
    Vector reduce(const Matrix3 & v) const { 
      double e1, e2, e3;
      v.getEigenValues(e1, e2, e3); 
      return v.getEigenVectors(e2, e1)[1];
    }
  };

  class TensorMinEigenvectDiag : public TensorToVectorDiag {
  public:
    TensorMinEigenvectDiag() {}
    string name() const { return "mineigenvect"; }
    Vector reduce(const Matrix3 & v) const {
      double e1, e2, e3;
      v.getEigenValues(e1, e2, e3); 
      return v.getEigenVectors(e3, e1)[2];
    }
  };
  
  // --------------------------------------------------------------------------
  
  class PreTensorToVectorDiag : public VectorDiag {
  public:
    PreTensorToVectorDiag(const TensorDiag * preop_, const TensorToVectorDiag * realdiag_)
      : preop(preop_), realdiag(realdiag_) {}
    
    ~PreTensorToVectorDiag() {}
    
    std::string name() const { return preop->name()+"_"+realdiag->name(); }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, int index, 
                    NCVariable<Vector>  & res) const {
      NCVariable<Matrix3> fullvals;
      (*preop)(da, patch, fieldname, imat, index, fullvals);
      res.allocate(fullvals.getLowIndex(), fullvals.getHighIndex());
      for(NCVariable<Matrix3>::iterator it=fullvals.begin();it!=fullvals.end();it++)
        res[it.getIndex()] = realdiag->reduce(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, int index, 
                    CCVariable<Vector>  & res) const {
      NCVariable<Matrix3> fullvalues;
      (*preop)(da, patch, fieldname, imat, index, fullvalues);
      res.allocate(fullvalues.getLowIndex(), fullvalues.getHighIndex());
      for(CCVariable<Matrix3>::iterator it=fullvalues.begin();it!=fullvalues.end();it++)
        res[it.getIndex()] = realdiag->reduce(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, int index,
                    ParticleSubset * pset,
                    ParticleVariable<Vector> & res) const {
      ParticleVariable<Matrix3> fullvalues;
      fullvalues.allocate(pset);
      (*preop)(da, patch, fieldname, imat, index, pset, fullvalues);
      res.allocate(pset);
      for(ParticleSubset::iterator pit(pset->begin());pit!=pset->end();pit++)
        res[*pit] = realdiag->reduce( fullvalues[*pit] );
    }
    
  private:
    const TensorDiag         * preop;
    const TensorToVectorDiag * realdiag;
  };
  // --------------------------------------------------------------------------
  
  typedef map<Uintah::TypeDescription::Type, vector<VectorDiag *> >     VDiagMap;
  typedef map<string, map<Uintah::TypeDescription::Type, VectorDiag*> > PreVDiagMap;
  
  VDiagMap    _vdiagtable;
  PreVDiagMap _vprediagtable;
  
  void createVectorDiags() {
    if(_vdiagtable.size()) return;
    
    _vdiagtable[Uintah::TypeDescription::Vector].push_back(new VectorValueDiag());
    
    // _vdiagtable[Uintah::TypeDescription::Matrix3].push_back(new TensorCompDiag(0));
    // _vdiagtable[Uintah::TypeDescription::Matrix3].push_back(new TensorCompDiag(1));
    // _vdiagtable[Uintah::TypeDescription::Matrix3].push_back(new TensorCompDiag(2));
    _vdiagtable[Uintah::TypeDescription::Matrix3].push_back(new TensorMinEigenvectDiag());
    _vdiagtable[Uintah::TypeDescription::Matrix3].push_back(new TensorMidEigenvectDiag());
    _vdiagtable[Uintah::TypeDescription::Matrix3].push_back(new TensorMaxEigenvectDiag());
  }
  
  void destroyVectorDiags() {
    for(map<Uintah::TypeDescription::Type, vector<VectorDiag *> >::iterator tit(_vdiagtable.begin());
        tit!=_vdiagtable.end();tit++) {
      for(vector<VectorDiag *>::iterator dit(tit->second.begin());dit!=tit->second.end();dit++) {
        delete *dit;
      }
    }
    _vdiagtable.clear();
  }
  
  void describeVectorDiags(ostream & os)
  {
    createVectorDiags();
    if(_vdiagtable[Uintah::TypeDescription::Vector].size())
      os << "  Vector -> Vector" << endl;
    for(vector<VectorDiag *>::iterator dit(_vdiagtable[Uintah::TypeDescription::Vector].begin());
        dit!=_vdiagtable[Uintah::TypeDescription::Vector].end();dit++) {
      os << "    " << (*dit)->name() << endl;
    }
    if(_vdiagtable[Uintah::TypeDescription::Matrix3].size())
      os << "  Tensor -> Vector" << endl;
    for(vector<VectorDiag *>::iterator dit(_vdiagtable[Uintah::TypeDescription::Matrix3].begin());
        dit!=_vdiagtable[Uintah::TypeDescription::Matrix3].end();dit++) {
      os << "    " << (*dit)->name() << endl;
    }
  }
  
  // --------------------------------------------------------------------------
  
  int numberOfVectorDiags(const Uintah::TypeDescription * fldtype) {
    createVectorDiags();
    if(!_vdiagtable.count(fldtype->getSubType()->getType())) return 0;
    return _vdiagtable[fldtype->getSubType()->getType()].size();
  }
  
  std::string vectorDiagName(const Uintah::TypeDescription * fldtype, int idiag) {
    createVectorDiags();
    return _vdiagtable[fldtype->getSubType()->getType()][idiag]->name();
  }
  
  VectorDiag const * createVectorDiag(const Uintah::TypeDescription * fldtype, int idiag,
                                      const TensorDiag * tensorpreop)
  {
    createVectorDiags();
    Uintah::TypeDescription::Type srctype = fldtype->getSubType()->getType();
    
    VectorDiag const * vdiag =  _vdiagtable[srctype][idiag];
    
    if(tensorpreop != 0 && srctype == Uintah::TypeDescription::Matrix3 ) {
      if(!_vprediagtable[ tensorpreop->name() ].count(srctype) )
        _vprediagtable[tensorpreop->name()][srctype]= 
          new PreTensorToVectorDiag( tensorpreop, dynamic_cast<const TensorToVectorDiag *>(vdiag) );
      
      return _vprediagtable[ tensorpreop->name() ][fldtype->getSubType()->getType()];
    } else {
      return vdiag;
    }
  }
  
  list<Uintah::VectorDiag const *> 
  createVectorDiags(const Uintah::TypeDescription * fldtype, 
                    const SCIRun::FieldSelection & fldselection,
                    const Uintah::TensorDiag * preop)
  {
    list<VectorDiag const *>  res;
    int ndiags = numberOfVectorDiags(fldtype);
    for(int idiag=0;idiag<ndiags;idiag++)
      {
        if(fldselection.wantDiagnostic(vectorDiagName(fldtype, idiag))) {
          res.push_back(createVectorDiag(fldtype, idiag, preop));
        }
      }
    return res;
  }
  
}
