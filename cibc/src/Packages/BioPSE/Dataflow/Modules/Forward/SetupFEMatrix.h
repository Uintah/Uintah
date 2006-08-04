/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
*/


/*
 *  SetupFEMatrix.h:
 *
 *  Written by:
 *   Frank B. Sachse
 *   CVRTI
 *   University of Utah
 *   Nov 2005
 *
 *  Generalized version of code from  
 *   Ruth Nicholson Klepfer, Department of Bioengineering
 *   University of Utah, Oct 1994
 *   Alexei Samsonov, Department of Computer Science
 *   University of Utah, Mar 2001    
 *   Sascha Moehrs, SCI , University of Utah, January 2003 (Hex)
 *   Lorena Kreda, Northeastern University, November 2003 (Tri)
 */

#if !defined(SetupFEMatrix_h)
#define SetupFEMatrix_h

#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Dataflow/Network/Module.h>
#include <Packages/BioPSE/Core/Algorithms/NumApproximation/BuildFEMatrix.h>


using std::endl;

namespace BioPSE {
//#define SETUPFEM_DEBUG

#ifdef SETUPFEM_DEBUG
using std::cerr;
#endif

class SetupFEMatrixParam 
{
public:
  SetupFEMatrixParam() :
    UseCond_(-1),
    UseBasis_(-1),
    gen_(-1),
    nprocessors_(1) 
  {
  }

  FieldHandle fieldH_;
  int UseCond_;
  int UseBasis_;
  int gen_;
  int nprocessors_;
};

        
class SetupFEMatrixAlgo : public DynamicAlgoBase
{
public:
  virtual MatrixHandle execute(const SetupFEMatrixParam& param) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *mtd,
					    const TypeDescription *btd,
					    const TypeDescription *dtd);
};

template<class FIELD>
void
create_conductivities(FIELD, Tensor &, 
                      const SetupFEMatrixParam&, 
                      vector<pair<string, Tensor> >&)
{
}

template<class FIELD>
void
create_conductivities(FIELD, double &, 
                      const SetupFEMatrixParam&, 
                      vector<pair<string, Tensor> >&)
{
}

template<class FIELD> 
void
create_conductivities(FIELD *pField, int, 
                      const SetupFEMatrixParam& SFP, 
                      vector<pair<string, Tensor> > &tens)
{
  if (!SFP.UseCond_ || !SFP.fieldH_->get_property("conductivity_table", tens)) 
  {
    unsigned int maxf = *max_element(pField->fdata().begin(), 
				     pField->fdata().end()); 
      
    vector<double> t(6);
    t[0] = t[3] = t[5] = 1;
    t[1] = t[2] = t[4] = 0;
    Tensor ten(t);
    for (unsigned int i = 0; i <= maxf; i++) 
      tens.push_back(pair<string, Tensor>(to_string((int)i), ten));
  }
}

template<class FIELD>
class SetupFEMatrixAlgoT : public SetupFEMatrixAlgo
{
  //! For the per-conductivity bases
  int gen_;
  int UseCond_;
  int UseBasis_;
  int nprocessors_;

  Array1<Array1<double> > dataBasis_;
  MatrixHandle AmatH_;  // shape information

  MatrixHandle hGblMtrx_;
  

public:
  SetupFEMatrixAlgoT() : 
    gen_(-1), 
    UseCond_(-1), 
    UseBasis_(-1), 
    nprocessors_(1)
  {
  }
      
  //! virtual interface. 
  virtual MatrixHandle execute(const SetupFEMatrixParam& SFP)
  {
#ifdef SETUPFEM_DEBUG
    cerr << "SetupFEMatrixAlgoT::execute" << endl;
#endif
    FIELD *pField= dynamic_cast<FIELD *>(SFP.fieldH_.get_rep());
    ASSERT(pField);
	
    const string hftn = SFP.fieldH_->get_type_description()->get_name();
#ifdef SETUPFEM_DEBUG
    cerr << "SetupFEMatrixAlgoT::execute " << hftn << endl;
#endif
    TypeDescription::td_vec *htdv = 
      SFP.fieldH_->get_type_description(Field::FDATA_TD_E)->get_sub_type();
    const string hfvaltype = (*htdv)[0]->get_name();

    bool index_based;
    if (hfvaltype == "int") 
      index_based = true;
    else if (hfvaltype == "Tensor") 
      index_based = false;
    else 
      ASSERT(0);

    if (SFP.gen_ == gen_ 
	&& hGblMtrx_.get_rep() 
	&& SFP.UseCond_==UseCond_
	&& SFP.UseBasis_==UseBasis_)
      return hGblMtrx_;

    //! Either use supplied tensors, or make an array of identity tensors
    vector<pair<string, Tensor> > tens;
 
    typename FIELD::basis_type::value_type dummy;
    create_conductivities(pField, dummy, SFP, tens);

    //! Cache data values for comparison next time
    gen_ = SFP.gen_;
    UseBasis_ = SFP.UseBasis_;

    nprocessors_ = SFP.nprocessors_;
    if (nprocessors_ > Thread::numProcessors() * 4) 
      nprocessors_ = Thread::numProcessors() * 4;
 
    if (index_based && UseBasis_) { 
      //! If the user wants to use basis matrices, 
      //!    first check to see if we need to recompute them
      if (SFP.fieldH_->mesh()->generation != gen_ || 
	  tens.size() != (unsigned int)(dataBasis_.size())) {
	gen_ = SFP.fieldH_->mesh()->generation;
	//! Need to build basis matrices
	build_basis_matrices(SFP.fieldH_, tens.size());
      }
      //! Have basis matrices, compute combined matrix
      hGblMtrx_ = build_composite_matrix(tens);
    } else           
      BuildFEMatrix<FIELD>::build_FEMatrix(SFP.fieldH_, tens, hGblMtrx_, 
					   1.0, nprocessors_);
  
    return hGblMtrx_;
  }

  void build_basis_matrices(FieldHandle fieldH, unsigned int nconds)
  {
#ifdef SETUPFEM_DEBUG
    cerr << "SetupFEMatrixAlgoT::build_basis_matrices" << endl;
#endif
    Tensor zero(0);
    Tensor identity(1);
    
    MatrixHandle aH;
    vector<pair<string, Tensor> > tens(nconds, pair<string, Tensor>("", zero));
    BuildFEMatrix<FIELD>::build_FEMatrix(fieldH, tens, aH, 
					 1.0, nprocessors_);
    AmatH_ = aH;
    AmatH_.detach(); //! Store our matrix shape
    
    dataBasis_.resize(nconds);
    for (unsigned int i=0; i<nconds; i++) {
      tens[i].first=to_string(i);
      tens[i].second=identity;
      BuildFEMatrix<FIELD>::build_FEMatrix(fieldH, tens, aH,
                                           1.0, nprocessors_);
      SparseRowMatrix *m = dynamic_cast<SparseRowMatrix*>(aH.get_rep());
      dataBasis_[i].resize(m->nnz);
      for (int j=0; j<m->nnz; j++)
	dataBasis_[i][j] = m->a[j];
      tens[i].second=zero;
    }
  }

  MatrixHandle build_composite_matrix(const vector<pair<string,Tensor> >&tens)
  {
#ifdef SETUPFEM_DEBUG
    cerr << "SetupFEMatrixAlgoT::build_composite_matrix" << endl;
#endif
    MatrixHandle fem_mat = AmatH_;
    fem_mat.detach();
    SparseRowMatrix *m = dynamic_cast<SparseRowMatrix*>(fem_mat.get_rep());
    double *sum = m->a;
    for (unsigned int i=0; i<tens.size(); i++)
    {
      double weight = tens[i].second.mat_[0][0];
      for (int j=0; j<dataBasis_[i].size(); j++)
      {
	sum[j] += weight*dataBasis_[i][j];
      }
    }
    return fem_mat;
  }
};  //end class SetupFEMatrixAlgoT

} // end namespace BioPSE

#endif // SetupFEMatrix_h
