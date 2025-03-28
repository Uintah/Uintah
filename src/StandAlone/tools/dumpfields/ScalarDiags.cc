/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#include "ScalarDiags.h"
#include "TensorDiags.h"
#include <Core/Grid/Variables/ParticleVariable.h>

#include <string>
#include <map>
#include <list>

using namespace std;

namespace Uintah {

  ScalarDiag::~ScalarDiag() {}

  // --------------------------------------------------------------------------

  class ScalarValueDiag : public ScalarDiag {
  public:
    ScalarValueDiag() {}
    virtual string name() const { return "value"; }

    //__________________________________
    //
    void operator()(DataArchive * da, const Patch * patch,
                    const std::string & fieldname,
                    int imat, int index,
                    NCVariable<double> & values) const
    {
      da->query(values, fieldname, imat, patch, index);
    }

    //__________________________________
    //
    void operator()(DataArchive * da, const Patch * patch,
                    const std::string & fieldname,
                    int imat, int index,
                    CCVariable<double> & values) const
    {
      da->query(values, fieldname, imat, patch, index);
    }

    //__________________________________
    //
    void operator()(DataArchive * da, const Patch * patch,
                    const std::string & fieldname,
                    int imat, int index,
                    ParticleSubset *,
                    ParticleVariable<double> & values) const
    {
      da->query(values, fieldname, imat, patch, index);
    }
  };

  //______________________________________________________________________
  //
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
    //__________________________________
    //
    void operator()(DataArchive * da, const Patch * patch,
                    const std::string & fieldname,
                    int imat, int index,
                    NCVariable<double> & values) const
    {
      NCVariable<Vector>  vectvalues;
      da->query(vectvalues, fieldname, imat, patch, index);

      values.allocate(vectvalues.getLowIndex(), vectvalues.getHighIndex());

      serial_for( vectvalues.range(), [&](int i, int j, int k) {
        values(i,j,k) = reduce(vectvalues(i,j,k));
      });
    }
    //__________________________________
    //
    void operator()(DataArchive * da, const Patch * patch,
                    const std::string & fieldname,
                    int imat, int index,
                    CCVariable<double> & values) const
    {
      CCVariable<Vector>  vectvalues;
      da->query(vectvalues, fieldname, imat, patch, index);

      values.allocate(vectvalues.getLowIndex(), vectvalues.getHighIndex());

      serial_for( vectvalues.range(), [&](int i, int j, int k) {
        values(i,j,k) = reduce(vectvalues(i,j,k));
      });
    }
    //__________________________________
    //
    void operator()(DataArchive * da, const Patch * patch,
                    const std::string & fieldname,
                    int imat, int index,
                    ParticleSubset * parts,
                    ParticleVariable<double> & values) const
    {
      ParticleVariable<Vector>  vectvalues;
      da->query(vectvalues, fieldname, imat, patch, index);

      values.allocate(parts);

      for(ParticleSubset::iterator pit(parts->begin());pit!=parts->end();pit++)
        values[*pit] = reduce(vectvalues[*pit]);
    }
  };

  //______________________________________________________________________
  //
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

  //______________________________________________________________________
  //
  class VectorMagDiag : public VectorToScalarDiag {
  public:
    string name() const { return "magnitude"; }

    double reduce(const Vector & v) const
    {  return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
  };

  //______________________________________________________________________
  //
  class VectorNormDiag : public VectorToScalarDiag {
  public:
    string name() const { return "norm"; }

    double reduce(const Vector & v) const
    { return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
  };

  //______________________________________________________________________
  //
  class VectorMaxDiag : public VectorToScalarDiag {
  public:
    VectorMaxDiag() {}
    string name() const { return "maximum"; }

    double reduce(const Vector & v) const
    { return v.maxComponent(); }
  };

  //______________________________________________________________________
  //
  class VectorMinDiag : public VectorToScalarDiag {
  public:
    VectorMinDiag() {}
    string name() const { return "minimum"; }

    double reduce(const Vector & v) const
    { return v.minComponent(); }
  };

  // --------------------------------------------------------------------------

  class TensorToScalarDiag : public ScalarDiag {
  public:
    virtual ~TensorToScalarDiag() {}

    virtual double reduce(const Matrix3 & v) const = 0;

    //__________________________________
    //
    void operator()(DataArchive * da, const Patch * patch,
                    const std::string & fieldname,
                    int imat, int index,
                    NCVariable<double> & res) const
    {
      NCVariable<Matrix3> fullvalues;

      da->query(fullvalues, fieldname, imat, patch, index);
      res.allocate(fullvalues.getLowIndex(), fullvalues.getHighIndex());

      serial_for( fullvalues.range(), [&](int i, int j, int k) {
        res(i,j,k) = reduce(fullvalues(i,j,k));
      });
    }
    //__________________________________
    //
    void operator()(DataArchive * da, const Patch * patch,
                    const std::string & fieldname,
                    int imat, int index,
                    CCVariable<double> & res) const
    {
      CCVariable<Matrix3> fullvalues;

      da->query(fullvalues, fieldname, imat, patch, index);
      res.allocate(fullvalues.getLowIndex(), fullvalues.getHighIndex());

      serial_for( fullvalues.range(), [&](int i, int j, int k) {
        res(i,j,k) = reduce(fullvalues(i,j,k));
      });
    }
    //__________________________________
    //
    void operator()(DataArchive * da, const Patch * patch,
                    const std::string & fieldname,
                    int imat, int index,
                    ParticleSubset * parts,
                    ParticleVariable<double> & res) const
    {
      ParticleVariable<Matrix3>  fullvalues;
      fullvalues.allocate(parts);

      da->query(fullvalues, fieldname, imat, patch, index);
      res.allocate(parts);

      for(ParticleSubset::iterator pit(parts->begin());pit!=parts->end();pit++){
        res[*pit] = reduce(fullvalues[*pit]);
      }
    }
  };

  //______________________________________________________________________
  //
  class PreTensorToScalarDiag : public ScalarDiag {
  public:
    PreTensorToScalarDiag(const TensorDiag * preop_, const TensorToScalarDiag * realdiag_)
      : preop(preop_), realdiag(realdiag_) {}

    ~PreTensorToScalarDiag() {}

    std::string name() const {
      return preop->name()+"_"+realdiag->name();
    }

    //__________________________________
    //
    void operator()(DataArchive * da, const Patch * patch,
                    const std::string & fieldname,
                    int imat, int index,
                    NCVariable<double>  & res) const
    {
      NCVariable<Matrix3> fullvals;
      (*preop)(da, patch, fieldname, imat, index, fullvals);

      res.allocate(fullvals.getLowIndex(), fullvals.getHighIndex());

      serial_for( fullvals.range(), [&](int i, int j, int k) {
        res(i,j,k) = realdiag->reduce(fullvals(i,j,k));
      });
    }
    //__________________________________
    //
    void operator()(DataArchive * da, const Patch * patch,
                    const std::string & fieldname,
                    int imat, int index,
                    CCVariable<double>  & res) const
    {
      NCVariable<Matrix3> fullvalues;
      (*preop)(da, patch, fieldname, imat, index, fullvalues);

      res.allocate(fullvalues.getLowIndex(), fullvalues.getHighIndex());

      serial_for( fullvalues.range(), [&](int i, int j, int k) {
        res(i,j,k) = realdiag->reduce(fullvalues(i,j,k));
      });
    }

    //__________________________________
    //
    void operator()(DataArchive * da, const Patch * patch,
                    const std::string & fieldname,
                    int imat, int index,
                    ParticleSubset * pset,
                    ParticleVariable<double> & res) const
    {
      ParticleVariable<Matrix3> fullvalues;
      fullvalues.allocate(pset);

      (*preop)(da, patch, fieldname, imat, index, pset, fullvalues);
      res.allocate(pset);

      for(ParticleSubset::iterator pit(pset->begin());pit!=pset->end();pit++){
        res[*pit] = realdiag->reduce( fullvalues[*pit] );
      }
    }

  private:
    const TensorDiag         * preop;
    const TensorToScalarDiag * realdiag;
  };

  //______________________________________________________________________
  //
  class Matrix3MagDiag : public TensorToScalarDiag {
  public:
    string name() const { return "norm"; }
    double reduce(const Matrix3 & v) const { return v.Norm(); }
  };

  //______________________________________________________________________
  //
  class Matrix3Mag2Diag : public TensorToScalarDiag {
  public:
    string name() const { return "normsquared"; }
    double reduce(const Matrix3 & v) const { return v.NormSquared(); }
  };

  //______________________________________________________________________
  //
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
  //______________________________________________________________________
  //
  class Matrix3MaxAbsDiag : public TensorToScalarDiag {
  public:
    Matrix3MaxAbsDiag() {}
    string name() const { return "maxabselem"; }

    double reduce(const Matrix3 & v) const { return v.MaxAbsElem(); }
  };

  //______________________________________________________________________
  //
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

  //______________________________________________________________________
  //
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
  //______________________________________________________________________
  //
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
  //______________________________________________________________________
  //
  class Matrix3TraceDiag : public TensorToScalarDiag {
  public:
    Matrix3TraceDiag() {}
    string name() const { return "trace"; }
    double reduce(const Matrix3 & v) const
    {
      return v(0,0)+v(1,1)+v(2,2);
    }
  };

  //______________________________________________________________________
  //
  class Matrix3PressureDiag : public TensorToScalarDiag {
  public:
    Matrix3PressureDiag() {}
    string name() const { return "pressure"; }

    double reduce(const Matrix3 & v) const
    {
      return -(v(0,0)+v(1,1)+v(2,2))/3.;
    }
  };
  //______________________________________________________________________
  //
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
  //______________________________________________________________________
  //
  class Matrix3MinElemDiag : public TensorToScalarDiag {
  public:
    Matrix3MinElemDiag() {}
    string name() const { return "minelem"; }

    double reduce(const Matrix3 & v) const {
      double res = v(0,0);

      for(int ic=0;ic<3;ic++) {
        for(int jc=0;jc<3;jc++){
          if(v(ic,jc)<res){
            res = v(ic,jc);
          }
        }
      }
      return res;
    }
  };

  //______________________________________________________________________
  //
  class Matrix3MaxElemDiag : public TensorToScalarDiag {
  public:
    Matrix3MaxElemDiag() {}
    string name() const { return "maxelem"; }

    double reduce(const Matrix3 & v) const {
      double res = v(0,0);

      for(int ic=0;ic<3;ic++) for(int jc=0;jc<3;jc++){
        if(v(ic,jc)>res) {
          res = v(ic,jc);
        }
      }
      return res;
    }
  };

  // --------------------------------------------------------------------------

  typedef Uintah::TypeDescription _td;
  typedef map<Uintah::TypeDescription::Type, vector<ScalarDiag *> >     SDiagMap;
  typedef map<string, map<Uintah::TypeDescription::Type, ScalarDiag*> > PreSDiagMap;

  SDiagMap    _sdiagtable;
  PreSDiagMap _sprediagtable;

  void createScalarDiags() {
    if(_sdiagtable.size()) {
      return;
    }

    _sdiagtable[ _td::float_type ].push_back( new ScalarValueDiag() );
    _sdiagtable[ _td::double_type ].push_back( new ScalarValueDiag() );
    _sdiagtable[ _td::double_type ].push_back( new ScalarNormDiag() );

    _sdiagtable[ _td::Vector ].push_back( new VectorMagDiag() );
    _sdiagtable[ _td::Vector ].push_back( new VectorNormDiag() );
    _sdiagtable[ _td::Vector ].push_back( new VectorCompDiag(0) );
    _sdiagtable[ _td::Vector ].push_back( new VectorCompDiag(1) );
    _sdiagtable[ _td::Vector ].push_back( new VectorCompDiag(2) );
    _sdiagtable[ _td::Vector ].push_back( new VectorMinDiag() );
    _sdiagtable[ _td::Vector ].push_back( new VectorMaxDiag() );

    _sdiagtable[ _td::Point ].push_back( new VectorMagDiag() );
    _sdiagtable[ _td::Point ].push_back( new VectorCompDiag(0) );
    _sdiagtable[ _td::Point ].push_back( new VectorCompDiag(1) );
    _sdiagtable[ _td::Point ].push_back( new VectorCompDiag(2) );
    _sdiagtable[ _td::Point ].push_back( new VectorMinDiag() );
    _sdiagtable[ _td::Point ].push_back( new VectorMaxDiag() );

    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3MagDiag() );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3Mag2Diag() );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3CompDiag(0,0) );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3CompDiag(0,1) );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3CompDiag(0,2) );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3CompDiag(1,0) );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3CompDiag(1,1) );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3CompDiag(1,2) );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3CompDiag(2,0) );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3CompDiag(2,1) );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3CompDiag(2,2) );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3MinEigenDiag() );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3MidEigenDiag() );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3MaxEigenDiag() );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3TraceDiag() );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3PressureDiag() );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3EquivDiag() );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3MinElemDiag() );
    _sdiagtable[ _td::Matrix3 ].push_back( new Matrix3MaxElemDiag() );
  }
  //______________________________________________________________________
  //
  void destroyScalarDiags()
  {
    for(map< _td::Type, vector<ScalarDiag *> >::iterator tit(_sdiagtable.begin());
        tit!=_sdiagtable.end();tit++) {
      for(vector<ScalarDiag *>::iterator dit(tit->second.begin());dit!=tit->second.end();dit++) {
        delete *dit;
      }
    }
    _sdiagtable.clear();
  }
  //______________________________________________________________________
  //
  void describeScalarDiags(ostream & os)
  {
    createScalarDiags();

    if(_sdiagtable[_td::double_type].size()){
      os << "  Scalar -> Scalar" << endl;
    }

    for(vector<ScalarDiag *>::iterator dit(_sdiagtable[_td::double_type].begin());
        dit!=_sdiagtable[_td::double_type].end();dit++) {
      os << "    " << (*dit)->name() << endl;
    }

    if(_sdiagtable[_td::Vector].size()){
      os << "  Vector -> Scalar" << endl;
    }

    for(vector<ScalarDiag *>::iterator dit(_sdiagtable[_td::Vector].begin());
        dit!=_sdiagtable[_td::Vector].end();dit++) {
      os << "    " << (*dit)->name() << endl;
    }

    if(_sdiagtable[_td::Matrix3].size()){
      os << "  Tensor -> Scalar" << endl;
    }

    for(vector<ScalarDiag *>::iterator dit(_sdiagtable[_td::Matrix3].begin());
        dit!=_sdiagtable[_td::Matrix3].end();dit++) {
      os << "    " << (*dit)->name() << endl;
    }
  }

  // --------------------------------------------------------------------------

  int numberOfScalarDiags(const Uintah::TypeDescription * fldtype)
  {
    createScalarDiags();
    if(!_sdiagtable.count(fldtype->getSubType()->getType())){
      return 0;
    }
    return _sdiagtable[fldtype->getSubType()->getType()].size();
  }

  //______________________________________________________________________
  //
  std::string scalarDiagName(const Uintah::TypeDescription * fldtype, int idiag)
  {
    createScalarDiags();
    return _sdiagtable[fldtype->getSubType()->getType()][idiag]->name();
  }

  //______________________________________________________________________
  //
  ScalarDiag const * createScalarDiag(const Uintah::TypeDescription * fldtype, int idiag,
                                      const TensorDiag * tensorpreop)
  {
    _td::Type srctype = fldtype->getSubType()->getType();

    createScalarDiags();
    ScalarDiag const * sdiag =  _sdiagtable[srctype][idiag];

    if(tensorpreop != 0 && srctype == _td::Matrix3 ) {

      if(!_sprediagtable[ tensorpreop->name() ].count(srctype) ){

        _sprediagtable[tensorpreop->name()][srctype]=
          new PreTensorToScalarDiag( tensorpreop, dynamic_cast<const TensorToScalarDiag *>(sdiag) );
      }
      return _sprediagtable[ tensorpreop->name() ][fldtype->getSubType()->getType()];
    } else {
      return sdiag;
    }
  }
  //______________________________________________________________________
  //
  list<ScalarDiag const *>
  createScalarDiags(const Uintah::TypeDescription * fldtype,
                    const FieldSelection & fldselection,
                    const class TensorDiag * tensorpreop)
  {
    list<ScalarDiag const *>  res;
    int ndiags = numberOfScalarDiags(fldtype);
    for(int idiag=0;idiag<ndiags;idiag++){

        if(fldselection.wantDiagnostic(scalarDiagName(fldtype, idiag))) {
          res.push_back(createScalarDiag(fldtype, idiag, tensorpreop));
        }
      }
    return res;
  }

}
