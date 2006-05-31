#include "ScalarDiags.h"
#include "TensorDiags.h"
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>

#include <string>
#include <map>
#include <list>

using namespace std;

namespace Uintah {
  using namespace SCIRun;
  
  ScalarDiag::~ScalarDiag() {}
  
  // --------------------------------------------------------------------------
  
  class ScalarValueDiag : public ScalarDiag {
  public:
    ScalarValueDiag() {}
    virtual string name() const { return "value"; }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    NCVariable<double> & values) const 
    {
      da->query(values, fieldname, imat, patch, time);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    CCVariable<double> & values) const 
    {
      da->query(values, fieldname, imat, patch, time);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    ParticleSubset *,
                    ParticleVariable<double> & values) const 
    {
      da->query(values, fieldname, imat, patch, time);
    }
  };
  class ScalarNormDiag : public ScalarValueDiag {
  public:
    ScalarNormDiag() {}
    virtual string name() const { return "norm"; }
  };
  
  // --------------------------------------------------------------------------
  
  // convert vector to scalar
  class VectorToScalarDiag : public ScalarDiag {
  public:
    virtual ~VectorToScalarDiag() {}
    
    virtual double reduce(const Vector & v) const = 0;
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    NCVariable<double> & values) const \
    {
      NCVariable<Vector>  vectvalues;
      da->query(vectvalues, fieldname, imat, patch, time);
      
      values.allocate(vectvalues.getLowIndex(), vectvalues.getHighIndex());
      for(NCVariable<Vector>::iterator it=vectvalues.begin();it!=vectvalues.end();it++)
        values[it.getIndex()] = reduce(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    CCVariable<double> & values) const  \
    {
      CCVariable<Vector>  vectvalues;
      da->query(vectvalues, fieldname, imat, patch, time);
      
      values.allocate(vectvalues.getLowIndex(), vectvalues.getHighIndex());
      for(CCVariable<Vector>::iterator it=vectvalues.begin();it!=vectvalues.end();it++)
        values[it.getIndex()] = reduce(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    ParticleSubset * parts,
                    ParticleVariable<double> & values) const {
      ParticleVariable<Vector>  vectvalues;
      da->query(vectvalues, fieldname, imat, patch, time);
      values.allocate(parts);
      for(ParticleSubset::iterator pit(parts->begin());pit!=parts->end();pit++)
        values[*pit] = reduce(vectvalues[*pit]);
    }
  };
  
  class VectorCompDiag : public VectorToScalarDiag {
  public:
    VectorCompDiag(char ic) : ic_(ic) {
      snprintf(name_, 12, "component%d", ic_);
    }
    string name() const { return name_; }
    double reduce(const Vector & v) const { return v[ic_]; }
  private:
    char name_[12];
    char ic_;
  };
  
  class VectorMagDiag : public VectorToScalarDiag {
  public:
    string name() const { return "magnitude"; }
    double reduce(const Vector & v) const { return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
  };
  
  class VectorNormDiag : public VectorToScalarDiag {
  public:
    string name() const { return "norm"; }
    double reduce(const Vector & v) const { return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
  };
  
  class VectorMaxDiag : public VectorToScalarDiag {
  public:
    VectorMaxDiag() {}
    string name() const { return "maximum"; }
    double reduce(const Vector & v) const { return v.maxComponent(); }
  };
  
  class VectorMinDiag : public VectorToScalarDiag {
  public:
    VectorMinDiag() {}
    string name() const { return "minimum"; }
    double reduce(const Vector & v) const { return v.minComponent(); }
  };
  
  // --------------------------------------------------------------------------
  
  class TensorToScalarDiag : public ScalarDiag {
  public:
    virtual ~TensorToScalarDiag() {}
    
    virtual double reduce(const Matrix3 & v) const = 0;
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    NCVariable<double> & res) const {
      NCVariable<Matrix3> fullvalues;
      da->query(fullvalues, fieldname, imat, patch, time);
      res.allocate(fullvalues.getLowIndex(), fullvalues.getHighIndex());
      for(NCVariable<Matrix3>::iterator it=fullvalues.begin();it!=fullvalues.end();it++)
        res[it.getIndex()] = reduce(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    CCVariable<double> & res) const {
      CCVariable<Matrix3> fullvalues;
      da->query(fullvalues, fieldname, imat, patch, time);
      res.allocate(fullvalues.getLowIndex(), fullvalues.getHighIndex());
      for(CCVariable<Matrix3>::iterator it=fullvalues.begin();it!=fullvalues.end();it++)
        res[it.getIndex()] = reduce(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    ParticleSubset * parts,
                    ParticleVariable<double> & res) const {
      ParticleVariable<Matrix3>  fullvalues;
      fullvalues.allocate(parts);
      da->query(fullvalues, fieldname, imat, patch, time);
      res.allocate(parts);
      for(ParticleSubset::iterator pit(parts->begin());pit!=parts->end();pit++)
        res[*pit] = reduce(fullvalues[*pit]);
    }
  };
  
  class PreTensorToScalarDiag : public ScalarDiag {
  public:
    PreTensorToScalarDiag(const TensorDiag * preop_, const TensorToScalarDiag * realdiag_)
      : preop(preop_), realdiag(realdiag_) {}
    
    ~PreTensorToScalarDiag() {}
    
    std::string name() const { return preop->name()+"_"+realdiag->name(); }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    NCVariable<double>  & res) const {
      NCVariable<Matrix3> fullvals;
      (*preop)(da, patch, fieldname, imat, time, fullvals);
      res.allocate(fullvals.getLowIndex(), fullvals.getHighIndex());
      for(NCVariable<Matrix3>::iterator it=fullvals.begin();it!=fullvals.end();it++)
        res[it.getIndex()] = realdiag->reduce(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time, 
                    CCVariable<double>  & res) const {
      NCVariable<Matrix3> fullvalues;
      (*preop)(da, patch, fieldname, imat, time, fullvalues);
      res.allocate(fullvalues.getLowIndex(), fullvalues.getHighIndex());
      for(CCVariable<Matrix3>::iterator it=fullvalues.begin();it!=fullvalues.end();it++)
        res[it.getIndex()] = realdiag->reduce(*it);
    }
    
    void operator()(DataArchive * da, const Patch * patch, 
                    const std::string & fieldname,
                    int imat, double time,
                    ParticleSubset * pset,
                    ParticleVariable<double> & res) const {
      ParticleVariable<Matrix3> fullvalues;
      fullvalues.allocate(pset);
      (*preop)(da, patch, fieldname, imat, time, pset, fullvalues);
      res.allocate(pset);
      for(ParticleSubset::iterator pit(pset->begin());pit!=pset->end();pit++)
        res[*pit] = realdiag->reduce( fullvalues[*pit] );
    }
    
  private:
    const TensorDiag         * preop;
    const TensorToScalarDiag * realdiag;
  };
  
  class Matrix3MagDiag : public TensorToScalarDiag {
  public:
    string name() const { return "norm"; }
    double reduce(const Matrix3 & v) const { return v.Norm(); }
  };
  
  class Matrix3Mag2Diag : public TensorToScalarDiag {
  public:
    string name() const { return "normsquared"; }
    double reduce(const Matrix3 & v) const { return v.NormSquared(); }
  };
  
  class Matrix3CompDiag : public TensorToScalarDiag {
  public:
    Matrix3CompDiag(char ic, char jc) : ic_(ic), jc_(jc) {
      snprintf(name_, 12, "component%d%d", ic_, jc_);
    }
    string name() const { return name_; }
    double reduce(const Matrix3 & v) const { return v(ic_,jc_); }
  private:
    char name_[12];
    char ic_, jc_;
  };
  
  class Matrix3MaxAbsDiag : public TensorToScalarDiag {
  public:
    Matrix3MaxAbsDiag() {}
    string name() const { return "maxabselem"; }
    double reduce(const Matrix3 & v) const { return v.MaxAbsElem(); }
  };
  
  class Matrix3MaxEigenDiag : public TensorToScalarDiag {
  public:
    Matrix3MaxEigenDiag() {}
    string name() const { return "maxeigen"; }
    double reduce(const Matrix3 & v) const { 
      double e1, e2, e3;
      v.getEigenValues(e1, e2, e3); 
      return e1; 
    }
  };

  class Matrix3MidEigenDiag : public TensorToScalarDiag {
  public:
    Matrix3MidEigenDiag() {}
    string name() const { return "mideigen"; }
    double reduce(const Matrix3 & v) const { 
      double e1, e2, e3;
      v.getEigenValues(e1, e2, e3); 
      return e2; 
    }
  };

  class Matrix3MinEigenDiag : public TensorToScalarDiag {
  public:
    Matrix3MinEigenDiag() {}
    string name() const { return "mineigen"; }
    double reduce(const Matrix3 & v) const {
      double e1, e2, e3;
      v.getEigenValues(e1, e2, e3); 
      return e3; 
    }
  };
  
  class Matrix3TraceDiag : public TensorToScalarDiag {
  public:
    Matrix3TraceDiag() {}
    string name() const { return "trace"; }
    double reduce(const Matrix3 & v) const { return v(0,0)+v(1,1)+v(2,2); }
  };
  
  class Matrix3PressureDiag : public TensorToScalarDiag {
  public:
    Matrix3PressureDiag() {}
    string name() const { return "pressure"; }
    double reduce(const Matrix3 & v) const { return -(v(0,0)+v(1,1)+v(2,2))/3.; }
  };
  
  class Matrix3EquivDiag : public TensorToScalarDiag {
  public:
    Matrix3EquivDiag() {}
    string name() const { return "equiv"; }
    double reduce(const Matrix3 & v) const { 
      double p = -(v(0,0)+v(1,1)+v(2,2))/3.;
      Matrix3 vdash = v + Matrix3(p,0,0,
                                  0,p,0,
                                  0,0,p);
      return sqrt(1.5)*vdash.Norm();
    }
  };
  
  class Matrix3MinElemDiag : public TensorToScalarDiag {
  public:
    Matrix3MinElemDiag() {}
    string name() const { return "minelem"; }
    double reduce(const Matrix3 & v) const { 
      double res = v(0,0);
      for(int ic=0;ic<3;ic++) for(int jc=0;jc<3;jc++)
        if(v(ic,jc)<res) res = v(ic,jc);
      return res;
    }
  };
  
  class Matrix3MaxElemDiag : public TensorToScalarDiag {
  public:
    Matrix3MaxElemDiag() {}
    string name() const { return "maxelem"; }
    double reduce(const Matrix3 & v) const { 
      double res = v(0,0);
      for(int ic=0;ic<3;ic++) for(int jc=0;jc<3;jc++)
        if(v(ic,jc)>res) res = v(ic,jc);
      return res;
    }
  };
  
  // --------------------------------------------------------------------------
  
  typedef map<TypeDescription::Type, vector<ScalarDiag *> >     SDiagMap;
  typedef map<string, map<TypeDescription::Type, ScalarDiag*> > PreSDiagMap;
  
  SDiagMap    _sdiagtable;
  PreSDiagMap _sprediagtable;
  
  void createScalarDiags() {
    if(_sdiagtable.size()) return;
    
    _sdiagtable[TypeDescription::float_type].push_back(new ScalarValueDiag());
    _sdiagtable[TypeDescription::double_type].push_back(new ScalarValueDiag());
    _sdiagtable[TypeDescription::double_type].push_back(new ScalarNormDiag());
    
    _sdiagtable[TypeDescription::Vector].push_back(new VectorMagDiag());
    _sdiagtable[TypeDescription::Vector].push_back(new VectorNormDiag());
    _sdiagtable[TypeDescription::Vector].push_back(new VectorCompDiag(0));
    _sdiagtable[TypeDescription::Vector].push_back(new VectorCompDiag(1));
    _sdiagtable[TypeDescription::Vector].push_back(new VectorCompDiag(2));
    _sdiagtable[TypeDescription::Vector].push_back(new VectorMinDiag());
    _sdiagtable[TypeDescription::Vector].push_back(new VectorMaxDiag());
    
    _sdiagtable[TypeDescription::Point].push_back(new VectorMagDiag());
    _sdiagtable[TypeDescription::Point].push_back(new VectorCompDiag(0));
    _sdiagtable[TypeDescription::Point].push_back(new VectorCompDiag(1));
    _sdiagtable[TypeDescription::Point].push_back(new VectorCompDiag(2));
    _sdiagtable[TypeDescription::Point].push_back(new VectorMinDiag());
    _sdiagtable[TypeDescription::Point].push_back(new VectorMaxDiag());
    
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3MagDiag());
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3Mag2Diag());
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3CompDiag(0,0));
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3CompDiag(0,1));
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3CompDiag(0,2));
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3CompDiag(1,0));
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3CompDiag(1,1));
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3CompDiag(1,2));
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3CompDiag(2,0));
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3CompDiag(2,1));
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3CompDiag(2,2));
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3MinEigenDiag());
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3MidEigenDiag());
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3MaxEigenDiag());
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3TraceDiag());
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3PressureDiag());
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3EquivDiag());
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3MinElemDiag());
    _sdiagtable[TypeDescription::Matrix3].push_back(new Matrix3MaxElemDiag());
  }
  
  void destroyScalarDiags() {
    for(map<TypeDescription::Type, vector<ScalarDiag *> >::iterator tit(_sdiagtable.begin());
        tit!=_sdiagtable.end();tit++) {
      for(vector<ScalarDiag *>::iterator dit(tit->second.begin());dit!=tit->second.end();dit++) {
        delete *dit;
      }
    }
    _sdiagtable.clear();
  }
  
  void describeScalarDiags(ostream & os)
  {
    createScalarDiags();
    if(_sdiagtable[TypeDescription::double_type].size())
      os << "  Scalar -> Scalar" << endl;
    for(vector<ScalarDiag *>::iterator dit(_sdiagtable[TypeDescription::double_type].begin());
        dit!=_sdiagtable[TypeDescription::double_type].end();dit++) {
      os << "    " << (*dit)->name() << endl;
    }
    
    if(_sdiagtable[TypeDescription::Vector].size())
      os << "  Vector -> Scalar" << endl;
    for(vector<ScalarDiag *>::iterator dit(_sdiagtable[TypeDescription::Vector].begin());
        dit!=_sdiagtable[TypeDescription::Vector].end();dit++) {
      os << "    " << (*dit)->name() << endl;
    }
    if(_sdiagtable[TypeDescription::Matrix3].size())
      os << "  Tensor -> Scalar" << endl;
    for(vector<ScalarDiag *>::iterator dit(_sdiagtable[TypeDescription::Matrix3].begin());
        dit!=_sdiagtable[TypeDescription::Matrix3].end();dit++) {
      os << "    " << (*dit)->name() << endl;
    }
  }
  
  // --------------------------------------------------------------------------
  
  int numberOfScalarDiags(const TypeDescription * fldtype) {
    createScalarDiags();
    if(!_sdiagtable.count(fldtype->getSubType()->getType())) return 0;
    return _sdiagtable[fldtype->getSubType()->getType()].size();
  }
  
  std::string scalarDiagName(const TypeDescription * fldtype, int idiag) {
    createScalarDiags();
    return _sdiagtable[fldtype->getSubType()->getType()][idiag]->name();
  }
  
  ScalarDiag const * createScalarDiag(const TypeDescription * fldtype, int idiag,
                                      const TensorDiag * tensorpreop)
  {
    TypeDescription::Type srctype = fldtype->getSubType()->getType();
    
    createScalarDiags();
    ScalarDiag const * sdiag =  _sdiagtable[srctype][idiag];
    
    if(tensorpreop != 0 && srctype == TypeDescription::Matrix3 ) {
      if(!_sprediagtable[ tensorpreop->name() ].count(srctype) )
        _sprediagtable[tensorpreop->name()][srctype]= 
          new PreTensorToScalarDiag( tensorpreop, dynamic_cast<const TensorToScalarDiag *>(sdiag) );
      
      return _sprediagtable[ tensorpreop->name() ][fldtype->getSubType()->getType()];
    } else {
      return sdiag;
    }
  }
  
  list<ScalarDiag const *> 
  createScalarDiags(const TypeDescription * fldtype, 
                    const FieldSelection & fldselection,
                    const class TensorDiag * tensorpreop)
  {
    list<ScalarDiag const *>  res;
    int ndiags = numberOfScalarDiags(fldtype);
    for(int idiag=0;idiag<ndiags;idiag++)
      {
        if(fldselection.wantDiagnostic(scalarDiagName(fldtype, idiag))) {
          res.push_back(createScalarDiag(fldtype, idiag, tensorpreop));
        }
      }
    return res;
  }
  
}
