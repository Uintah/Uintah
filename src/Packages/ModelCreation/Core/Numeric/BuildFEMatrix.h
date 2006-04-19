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


#ifndef MODELCREATION_CORE_NUMERIC_BUILDFEMATRIX_H
#define MODELCREATION_CORE_NUMERIC_BUILDFEMATRIX_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>

#include <Core/Thread/Barrier.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Basis/Locate.h>

#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/Matrix.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <algorithm>
#include <sgi_stl_warnings_on.h>


namespace ModelCreation {

using namespace SCIRun;


class BuildFEMatrixAlgo : public DynamicAlgoBase
{
public:
  virtual bool BuildFEMatrix(ProgressReporter *pr, FieldHandle input, MatrixHandle& output, MatrixHandle& ctable, int numproc = 1);


};

template <class FIELD>
class BuildFEMatrixAlgoT : public BuildFEMatrixAlgo
{
public:
  virtual bool BuildFEMatrix(ProgressReporter *pr, FieldHandle input, MatrixHandle& output, MatrixHandle& ctable, int numproc = 1);
  
};

template <class FIELD> class FEMBuilder;




// --------------------------------------------------------------------------
// This piece of code was adapted from BuildFEMatrix.h
// Code has been modernized a little to meet demands.

template <class FIELD>
class FEMBuilder : public DynamicAlgoBase
{
public:

  // Constructor needed as Barrier needs to have name
  FEMBuilder() :
    barrier_("FEMBuilder Barrier")
  {
  }

  void build_matrix(FieldHandle input, MatrixHandle& output, MatrixHandle& ctable, int numproc);

private:

  // For parallel implementation
  Barrier barrier_;

  typename FIELD::mesh_type::basis_type mb_;
  typename FIELD::mesh_handle_type hMesh_;

  FieldHandle hField_;
  FIELD *pField_;

  MatrixHandle hA_;
  SparseRowMatrix* pA_;
  int np_;
  int* rows_;
  int* allCols_;
  std::vector<int> colIdx_;

  int domain_dimension;
  int local_dimension_nodes;
  int local_dimension_add_nodes;
  int local_dimension_derivatives;
  int local_dimension;
  int global_dimension_nodes;
  int global_dimension_add_nodes;
  int global_dimension_derivatives;
  int global_dimension; 

  std::vector<std::pair<string, Tensor> > tens_;

  void parallel(int proc);
    
private:
  
  // General case where we can indexed or non indexed data
  template<class T>
  inline Tensor get_tensor(T& val) const
  {
    if (tens_.size() == 0) return(Tensor(static_cast<double>(val)));
    return (tens_[static_cast<size_t>(val)].second);
  }

  // Specific case for when we have a tensor as datatype
  inline Tensor get_tensor(const Tensor& val) const
  {
    return (val);
  }

  inline void add_lcl_gbl(int row, const std::vector<int> &cols, const std::vector<double> &lcl_a)
  {
    for (int i = 0; i < (int)lcl_a.size(); i++) pA_->add(row, cols[i], lcl_a[i]);
  }

private:

  void create_numerical_integration(std::vector<std::vector<double> > &p,std::vector<double> &w,std::vector<std::vector<double> > &d);
  void build_local_matrix(typename FIELD::mesh_type::Elem::index_type c_ind,int row, std::vector<double> &l_stiff,std::vector<std::vector<double> > &p,std::vector<double> &w,std::vector<std::vector<double> >  &d);
  void setup();
  
};

template <class FIELD>
void FEMBuilder<FIELD>::build_matrix(FieldHandle input, MatrixHandle& output, MatrixHandle& ctable, int numproc)
{
  pField_ = dynamic_cast<FIELD *>(input.get_rep());
  hMesh_ = pField_->get_typed_mesh();

  int np = Thread::numProcessors();
  if (np > 5) np = 5;
  if (numproc > 0) { np = numproc; }
  np_ = np;
  
  pField_->get_property("conductivity_table",tens_);
  
  if (ctable.get_rep())
  {
    DenseMatrix* mat = ctable->dense();
    if (mat)
    {
      double* data = ctable->get_data_pointer();
      int m = ctable->nrows();
      Tensor T; 

      if (mat->ncols() == 1)
      {
        for (int p=0; p<m;p++)
        {
          T.mat_[0][0] = data[0*m+p];
          T.mat_[1][0] = 0.0;
          T.mat_[2][0] = 0.0;
          T.mat_[0][1] = 0.0;
          T.mat_[1][1] = data[0*m+p];
          T.mat_[2][1] = 0.0;
          T.mat_[0][2] = 0.0;
          T.mat_[1][2] = 0.0;
          T.mat_[2][2] = data[0*m+p];
          tens_.push_back(std::pair<string, Tensor>("",T));
        }
      }
       
      if (mat->ncols() == 6)
      {
        for (int p=0; p<m;p++)
        {
          T.mat_[0][0] = data[0*m+p];
          T.mat_[1][0] = data[1*m+p];
          T.mat_[2][0] = data[2*m+p];
          T.mat_[0][1] = data[1*m+p];
          T.mat_[1][1] = data[3*m+p];
          T.mat_[2][1] = data[4*m+p];
          T.mat_[0][2] = data[2*m+p];
          T.mat_[1][2] = data[4*m+p];
          T.mat_[2][2] = data[5*m+p];
          tens_.push_back(std::pair<string, Tensor>("",T));
        }
      }

      if (mat->ncols() == 9)
      {
        for (int p=0; p<m;p++)
        {
          T.mat_[0][0] = data[0*m+p];
          T.mat_[1][0] = data[1*m+p];
          T.mat_[2][0] = data[2*m+p];
          T.mat_[0][1] = data[1*m+p];
          T.mat_[1][1] = data[4*m+p];
          T.mat_[2][1] = data[5*m+p];
          T.mat_[0][2] = data[2*m+p];
          T.mat_[1][2] = data[5*m+p];
          T.mat_[2][2] = data[8*m+p];
          tens_.push_back(std::pair<string, Tensor>("",T));
        }
      }
    }
  }
  
  Thread::parallel(this, &FEMBuilder<FIELD>::parallel, np);

  output = hA_;
}



template <class FIELD>
bool BuildFEMatrixAlgoT<FIELD>::BuildFEMatrix(ProgressReporter *pr, FieldHandle input, MatrixHandle& output, MatrixHandle& ctable, int numproc)
{
  // Some sanity checks
  FIELD* field = dynamic_cast<FIELD *>(input.get_rep());
  if (field == 0)
  {
    pr->error("BuildFEMatrix: Could not obtain input field");
    return (false);
  }

  if (ctable.get_rep())
  {
    if ((ctable->ncols() != 2)||(ctable->ncols() != 6)||(ctable->ncols() != 9))
    {
      pr->error("BuildFEMatrix: Conductivity table needs to have 1, 6, or 9 columns");
      return (false);
    } 
    if (ctable->nrows() == 0)
    { 
      pr->error("BuildFEMatrix: ConductivityTable is empty");
      return (false);
    }
  }


  Handle<FEMBuilder<FIELD> > builder = scinew FEMBuilder<FIELD>;
  builder->build_matrix(input,output,ctable,numproc);

  if (output.get_rep() == 0)
  {    
    pr->error("BuildFEMatrix: Could not build output matrix");
    return (false);
  }
  
  return (true);
}




template <class FIELD>
void FEMBuilder<FIELD>::create_numerical_integration(std::vector<std::vector<double> > &p,
                                                   std::vector<double> &w,
                                                   std::vector<std::vector<double> > &d)
{
  p.resize(mb_.GaussianNum);
  w.resize(mb_.GaussianNum);
  d.resize(mb_.GaussianNum);

  for(int i=0; i < mb_.GaussianNum; i++)
  {
    w[i] = mb_.GaussianWeights[i];

    p[i].resize(domain_dimension);
    for(int j=0; j<domain_dimension; j++)
      p[i][j]=mb_.GaussianPoints[i][j];

    d[i].resize(local_dimension*domain_dimension);
    mb_.get_derivate_weights(p[i], (double *)&d[i][0]);
  }
}


//! build line of the local stiffness matrix
template <class FIELD>
void FEMBuilder<FIELD>::build_local_matrix(typename FIELD::mesh_type::Elem::index_type c_ind,
                                            int row, std::vector<double> &l_stiff,
                                            std::vector<std::vector<double> > &p,
                                            std::vector<double> &w,
                                            std::vector<std::vector<double> >  &d)
{
  Tensor T = get_tensor(pField_->value(c_ind));

  double Ca = T.mat_[0][0];
  double Cb = T.mat_[0][1];
  double Cc = T.mat_[0][2];
  double Cd = T.mat_[1][1];
  double Ce = T.mat_[1][2];
  double Cf = T.mat_[2][2];


  for(int i=0; i<local_dimension; i++)
    l_stiff[i] = 0.0;

  int local_dimension2=2*local_dimension;

  for (unsigned int i = 0; i < d.size(); i++)
  {
    std::vector<Point> Jv;
    hMesh_->derivate(p[i], c_ind, Jv);
    double J[9], Ji[9];
    J[0] = Jv[0].x();
    J[1] = Jv[0].y();
    J[2] = Jv[0].z();
    J[3] = Jv[1].x();
    J[4] = Jv[1].y();
    J[5] = Jv[1].z();
    J[6] = Jv[2].x();
    J[7] = Jv[2].y();
    J[8] = Jv[2].z();
	
    double detJ = InverseMatrix3x3(J, Ji);
    ASSERT(detJ>0);
    detJ*=w[i]*hMesh_->get_basis().volume();

    const double& Ji0 = Ji[0];
    const double& Ji1 = Ji[1];
    const double& Ji2 = Ji[2];
    const double& Ji3 = Ji[3];
    const double& Ji4 = Ji[4];
    const double& Ji5 = Ji[5];
    const double& Ji6 = Ji[6];
    const double& Ji7 = Ji[7];
    const double& Ji8 = Ji[8];
	
    const double *Nxi = &d[i][0];
    const double *Nyi = &d[i][local_dimension];
    const double *Nzi = &d[i][local_dimension2];
    const double &Nxip = Nxi[row];
    const double &Nyip = Nyi[row];
    const double &Nzip = Nzi[row];
    const double uxp = detJ*(Nxip*Ji0+Nyip*Ji1+Nzip*Ji2);
    const double uyp = detJ*(Nxip*Ji3+Nyip*Ji4+Nzip*Ji5);
    const double uzp = detJ*(Nxip*Ji6+Nyip*Ji7+Nzip*Ji8);
    const double uxyzpabc = uxp*Ca+uyp*Cb+uzp*Cc;
    const double uxyzpbde = uxp*Cb+uyp*Cd+uzp*Ce;
    const double uxyzpcef = uxp*Cc+uyp*Ce+uzp*Cf;
	
    for (unsigned int j = 0; j<local_dimension; j++)
    {
      const double &Nxj = Nxi[j];
      const double &Nyj = Nyi[j];
      const double &Nzj = Nzi[j];
	
      const double ux = Nxj*Ji0+Nyj*Ji1+Nzj*Ji2;
      const double uy = Nxj*Ji3+Nyj*Ji4+Nzj*Ji5;
      const double uz = Nxj*Ji6+Nyj*Ji7+Nzj*Ji8;
      l_stiff[j] += ux*uxyzpabc+uy*uxyzpbde+uz*uxyzpcef;
    }
  }

}

template <class FIELD>
void FEMBuilder<FIELD>::setup()
{
  domain_dimension = mb_.domain_dimension();
  ASSERT(domain_dimension>0);

  local_dimension_nodes = mb_.number_of_mesh_vertices();
  local_dimension_add_nodes = mb_.number_of_mesh_vertices()-mb_.number_of_vertices();
  local_dimension_derivatives = 0;
  local_dimension = local_dimension_nodes + local_dimension_add_nodes + local_dimension_derivatives; //!< degrees of freedom (dofs) of system
  ASSERT(mb_.dofs()==local_dimension);

  typename FIELD::mesh_type::Node::size_type mns;
  hMesh_->size(mns);
  global_dimension_nodes = mns;
  global_dimension_add_nodes = pField_->get_basis().size_node_values();
  global_dimension_derivatives = pField_->get_basis().size_derivatives();
  global_dimension = global_dimension_nodes+global_dimension_add_nodes+global_dimension_derivatives;

  hMesh_->synchronize(Mesh::EDGES_E | Mesh::NODE_NEIGHBORS_E);
  rows_ = scinew int[global_dimension+1];
  
  colIdx_.resize(np_+1);
}


// -- callback routine to execute in parallel
template <class FIELD>
void FEMBuilder<FIELD>::parallel(int proc_num)
{
  if (proc_num == 0)
  {
    setup();
  }
  
  barrier_.wait(np_);

  //! distributing dofs among processors
  const int start_gd = global_dimension * proc_num/np_;
  const int end_gd  = global_dimension * (proc_num+1)/np_;

  //! creating sparse matrix structure
  std::vector<unsigned int> mycols;
  mycols.reserve((end_gd - start_gd)*local_dimension*8);  //<! rough estimate

  typename FIELD::mesh_type::Elem::array_type ca;
  typename FIELD::mesh_type::Node::array_type na;
  typename FIELD::mesh_type::Edge::array_type ea;
  std::vector<int> neib_dofs;

  //! loop over system dofs for this thread
  for (unsigned int i = start_gd; i<end_gd; i++)
  {
    rows_[i] = mycols.size();

    neib_dofs.clear();
    //! check for nodes
    if (i<global_dimension_nodes)
    {
      //! get neighboring cells for node
      typename FIELD::mesh_type::Node::index_type idx;
      hMesh_->to_index(idx,i);
      hMesh_->get_elems(ca, idx);
    }
    else if (i<global_dimension_nodes+global_dimension_add_nodes)
    {
      //! check for additional nodes at edges
      //! get neighboring cells for node
      const int ii = i-global_dimension_nodes;
      typename FIELD::mesh_type::Edge::index_type idx;
      typename FIELD::mesh_type::Node::array_type nodes;
      typename FIELD::mesh_type::Elem::array_type elems;
      typename FIELD::mesh_type::Elem::array_type elems2;

      hMesh_->to_index(idx,ii);
      hMesh_->get_nodes(nodes,idx);
      hMesh_->get_elems(elems,nodes[0]);
      hMesh_->get_elems(elems2,nodes[1]);
      ca.clear();
      for (int v=0; v < elems.size(); v++)
         for (int w=0; w <elems2.size(); w++)
            if (elems[v] == elems2[w]) ca.push_back(elems[v]);
    }
    else
    {
      //! check for derivatives - to do
    }
	
    for(unsigned int j = 0; j < ca.size(); j++)
    {
      //! get neighboring nodes
      hMesh_->get_nodes(na, ca[j]);

      for(unsigned int k = 0; k < na.size(); k++) 
      {
        neib_dofs.push_back((int)(na[k])); // Must cast to (int) for SGI compiler. :-(
      }

      //! check for additional nodes at edges
      if (global_dimension_add_nodes)
      {
        //! get neighboring edges
        hMesh_->get_edges(ea, ca[j]);

        for(unsigned int k = 0; k < ea.size(); k++)
          neib_dofs.push_back(global_dimension + ea[k]);
      }
    }
	
    std::sort(neib_dofs.begin(), neib_dofs.end());

    for (unsigned int j=0; j<neib_dofs.size(); j++)
    {
      if (j == 0 || neib_dofs[j] != (int)mycols.back())
      {
        mycols.push_back(neib_dofs[j]);
      }
    }
  }

  colIdx_[proc_num] = mycols.size();

  //! check point
  barrier_.wait(np_);

  int st = 0;
  if (proc_num == 0)
  {
    for(int i=0; i<np_; i++)
    {
      const int ns = colIdx_[i];
      colIdx_[i] = st;
      st += ns;
    }

    colIdx_[np_] = st;
    allCols_ = scinew int[st];
  }

  //! check point
  barrier_.wait(np_);

  //! updating global column by each of the processors
  const int s = colIdx_[proc_num];
  const int n = mycols.size();

  for(int i=0; i<n; i++)
    allCols_[i+s] = mycols[i];

  for(int i = start_gd; i<end_gd; i++)
    rows_[i] += s;

  //! check point
  barrier_.wait(np_);

  //! the main thread makes the matrix
  if (proc_num == 0)
  {
    rows_[global_dimension] = st;
    pA_ = scinew SparseRowMatrix(global_dimension, global_dimension, rows_, allCols_, st);
    hA_ = pA_;
  }

  //! check point
  barrier_.wait(np_);

  //! zeroing in parallel
  const int ns = colIdx_[proc_num];
  const int ne = colIdx_[proc_num+1];
  double* a = &pA_->a[ns], *ae=&pA_->a[ne];

  while (a<ae) *a++=0.0;

  std::vector<std::vector<double> > ni_points;
  std::vector<double> ni_weights;
  std::vector<std::vector<double> > ni_derivatives;
  create_numerical_integration(ni_points, ni_weights, ni_derivatives);

  std::vector<double> lsml; //!< line of local stiffnes matrix
  lsml.resize(local_dimension);
      	
  //! loop over system dofs for this thread
  for (int i = start_gd; i<end_gd; i++)
  {
    if (i < global_dimension_nodes)
    {
      //! check for nodes
      //! get neighboring cells for node
      typename FIELD::mesh_type::Node::index_type idx;
      hMesh_->to_index(idx,i);
      hMesh_->get_elems(ca,idx);
    }
    else if (i < global_dimension_nodes + global_dimension_add_nodes)
    {
      //! check for additional nodes at edges
      //! get neighboring cells for additional nodes
      const int ii=i-global_dimension_nodes;
      typename FIELD::mesh_type::Edge::index_type idx;
      typename FIELD::mesh_type::Node::array_type nodes;
      typename FIELD::mesh_type::Elem::array_type elems;
      typename FIELD::mesh_type::Elem::array_type elems2;

      hMesh_->to_index(idx,ii);
      hMesh_->get_nodes(nodes,idx);
      hMesh_->get_elems(elems,nodes[0]);
      hMesh_->get_elems(elems2,nodes[1]);
      ca.clear();
      for (int v=0; v < elems.size(); v++)
         for (int w=0; w <elems2.size(); w++)
            if (elems[v] == elems2[w]) ca.push_back(elems[v]);

    }
    else
    {
      //! check for derivatives - to do
    }
	
    //! loop over elements attributed elements
    for (int j = 0; j < (int)ca.size(); j++)
    {
      neib_dofs.clear();
      int dofi = -1; //!< index of global dof in local dofs
      hMesh_->get_nodes(na, ca[j]); //!< get neighboring nodes
      for(int k = 0; k < (int)na.size(); k++)
      {
        neib_dofs.push_back((int)(na[k])); // Must cast to (int) for SGI compiler :-(
        if ((int)na[k] == i) dofi = neib_dofs.size()-1;
      }
      //! check for additional nodes at edges
      if (global_dimension_add_nodes)
      {
        hMesh_->get_edges(ea, ca[j]); //!< get neighboring edges
        for(int k = 0; k < (int)ea.size(); k++)
        {
          neib_dofs.push_back(global_dimension + ea[k]);
          if ((int)na[k] == i)
          dofi = neib_dofs.size() - 1;
        }
      }
      ASSERT(dofi!=-1);
      ASSERT((int)neib_dofs.size() == local_dimension);
      build_local_matrix(ca[j], dofi, lsml, ni_points, ni_weights, ni_derivatives);
      add_lcl_gbl(i, neib_dofs, lsml);
    }
  }

  for (int i=start_gd; i<end_gd; i++)
  {
    double sum=0.0, sumabs=0.0;
    for (int j=0; j<global_dimension; j++)
    {
      sum += pA_->get(i,j);
      sumabs += fabs(pA_->get(i,j));
    }
  }
  
  barrier_.wait(np_);
}

} // end namespace ModelCreation

#endif 
