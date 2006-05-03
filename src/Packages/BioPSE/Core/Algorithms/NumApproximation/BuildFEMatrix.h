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
 *  BuildFEMatrix.h:  class to build FE matrix
 *
 *  Written by:
 *   F. B. Sachse
 *   CVRTI
 *   University of Utah
 *   Nov 2005
 *
 *  Generalized version of code from
 *   Alexei Samsonov, Department of Computer Science, University of Utah
 *   March 2001   (Tet)
 *   Lorena Kreda, Northeastern University, October 2003 (Tri)
 *   Sascha Moehrs, SCI , University of Utah, January 2003 (Hex)
 *
 */

#include <iostream>

#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/Matrix.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Basis/Bases.h>

#include <Core/Datatypes/GenericField.h>
#include <algorithm>

//#define BUILDFEM_DEBUG

namespace BioPSE {

using namespace SCIRun;

template<class Field>
class BuildFEMatrix {
public:

  static bool build_FEMatrix(FieldHandle hField,
			     vector<pair<string, Tensor> >& tens,
			     MatrixHandle& hA, double unitsScale,
			     int num_procs=1);

private:
  typedef typename Field::basis_type FieldType;
  typedef typename Field::mesh_type Mesh;
  typedef typename Field::mesh_type::basis_type MeshType;
  typedef typename Field::mesh_handle_type MeshHandle;

  //! Private data members
  MeshType mb_;
  MeshHandle hMesh_;

  FieldType fb;
  FieldHandle hField_;
  Field *pField_;

  MatrixHandle& hA_;
  SparseRowMatrix* pA_;
  int np_;
  int* rows_;
  int* allCols_;
  Barrier barrier_;
  vector<int> colIdx_;
  vector<pair<string, Tensor> >& tens_;
  double unitsScale_;
  int domain_dimension;
  int local_dimension_nodes;
  int local_dimension_add_nodes;
  int local_dimension_derivatives;
  int local_dimension;
  int global_dimension_nodes;
  int global_dimension_add_nodes;
  int global_dimension_derivatives;
  int global_dimension;

  BuildFEMatrix(FieldHandle hField,
                vector<pair<string, Tensor> >& tens,
                MatrixHandle& hA,
                double unitsScale,
                int np=1):
    hField_(hField),
    hA_(hA),
    np_(np),
    rows_(NULL),
    allCols_(NULL),
    barrier_("BuildFEMatrix barrier"),
    colIdx_(np+1),
    tens_(tens),
    unitsScale_(unitsScale)
  {
    pField_= dynamic_cast<Field *>(hField.get_rep());
    hMesh_ = pField_->get_typed_mesh();
  }

  virtual ~BuildFEMatrix() {}


  //!< p is the gaussian points
  //!< w is the gaussian weights
  //!< d si the derivate weights at the gaussian points
  void create_numerical_integration(vector<vector<double> > &p,
				    vector<double> &w,
				    vector<vector<double> > &d);

  inline const Tensor& get_tensor(int val) const
  {
    return tens_[val].second;
  }

  inline const Tensor& get_tensor(const Tensor& val) const
  {
    return val;
  }

  //! Build line of the local stiffness matrix.
  void build_local_matrix(typename Mesh::Elem::index_type c_ind,
                          int row, vector<double> &l_stiff,
			  vector<vector<double> > &p,
			  vector<double> &w,
			  vector<vector<double> >  &d);

  inline void add_lcl_gbl(int row, const vector<int> &cols,
			  const vector<double> &lcl_a)
  {
    for (int i = 0; i < (int)lcl_a.size(); i++)
      pA_->add(row, cols[i], lcl_a[i]);
  }

  void setup();

  // -- Callback routine to execute in parallel.
  void parallel(int proc);
};


template <class Field>
bool
BuildFEMatrix<Field>::build_FEMatrix(FieldHandle hField,
                                     vector<pair<string, Tensor> >& tens,
                                     MatrixHandle& hA, double unitsScale,
                                     int num_procs)
{
  int np = Thread::numProcessors();

  if ( np > 2 )
  {
    np /= 2;
    if (np > 10)
    {
      np = 5;
    }
  }

  if (num_procs > 0) { np = num_procs; }

  hA = 0;

  BuildFEMatrix *hMaker =
    scinew BuildFEMatrix(hField, tens, hA, unitsScale, np);

  Thread::parallel(hMaker, &BuildFEMatrix::parallel, np);

  // -- refer to the object one more time not to make it die before
  delete hMaker;

  return hA.get_rep() != 0;
}


template <class Field>
void
BuildFEMatrix<Field>::create_numerical_integration(vector<vector<double> > &p,
                                                   vector<double> &w,
                                                   vector<vector<double> > &d)
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
template <class Field>
void
BuildFEMatrix<Field>::build_local_matrix(typename BuildFEMatrix::Mesh::Elem::index_type c_ind,
                                         int row, vector<double> &l_stiff,
                                         vector<vector<double> > &p,
                                         vector<double> &w,
                                         vector<vector<double> >  &d)
{
  typedef double onerow[3]; // This 'hack' is necessary to compile under IRIX CC
  const onerow *C = get_tensor(dynamic_cast<Field *>(hField_.get_rep())->value(c_ind)).mat_;

  double Ca = C[0][0]*unitsScale_;
  double Cb = C[0][1]*unitsScale_;
  double Cc = C[0][2]*unitsScale_;
  double Cd = C[1][1]*unitsScale_;
  double Ce = C[1][2]*unitsScale_;
  double Cf = C[2][2]*unitsScale_;
	
  for(int i=0; i<local_dimension; i++)
    l_stiff[i] = 0.0;

  int local_dimension2=2*local_dimension;

  for (unsigned int i = 0; i < d.size(); i++)
  {
    vector<Point> Jv;
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

    //cerr << J[0] << '\t' << J[1] << '\t' << J[2] << '\n';
    //cerr << J[3] << '\t' << J[4] << '\t' << J[5] << '\n';
    //cerr << J[6] << '\t' << J[7] << '\t' << J[8] << '\n';
	
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
	
    for (int j = 0; j<local_dimension; j++)
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


template <class Field>
void
BuildFEMatrix<Field>::setup()
{
  domain_dimension = mb_.domain_dimension();
  ASSERT(domain_dimension>0);

  local_dimension_nodes = mb_.number_of_mesh_vertices();
  local_dimension_add_nodes = mb_.number_of_mesh_vertices()-mb_.number_of_vertices();
  local_dimension_derivatives = 0;
  local_dimension = local_dimension_nodes + local_dimension_add_nodes + local_dimension_derivatives; //!< degrees of freedom (dofs) of system
  ASSERT(mb_.dofs()==local_dimension);

  typename Mesh::Node::size_type mns;
  hMesh_->size(mns);
  global_dimension_nodes = mns;
  global_dimension_add_nodes = pField_->get_basis().size_node_values();
  global_dimension_derivatives = pField_->get_basis().size_derivatives();
  global_dimension = global_dimension_nodes+global_dimension_add_nodes+global_dimension_derivatives;

#ifdef BUILDFEM_DEBUG
  cerr << "Gdn " <<  global_dimension_nodes << endl;
  cerr << "Gdan " <<  global_dimension_add_nodes << endl;
  cerr << "Gdd " <<  global_dimension_derivatives << endl;
  cerr << "Gdd " <<  global_dimension << endl;
#endif

  hMesh_->synchronize(Mesh::EDGES_E | Mesh::NODE_NEIGHBORS_E);
  rows_ = scinew int[global_dimension+1];
}


// -- Callback routine to execute in parallel.
template <class Field>
void
BuildFEMatrix<Field>::parallel(int proc)
{
#ifdef BUILDFEM_DEBUG
  cerr << "BuildFEMatrix::parallel " << proc << endl;
#endif

  if (proc == 0)
  {
    setup();
  }

  barrier_.wait(np_);

  //! Distributing dofs among processors.
  const int start_gd = global_dimension * proc/np_;
  const int end_gd  = global_dimension * (proc+1)/np_;

  //! Creating sparse matrix structure.
  vector<unsigned int> mycols;
  //! Estimate for elements in submatrix of sparse matrix.
  mycols.reserve((end_gd - start_gd)*local_dimension*8);

  typename Mesh::Elem::array_type ca;
  typename Mesh::Node::array_type na;
  typename Mesh::Edge::array_type ea;
  vector<int> neib_dofs;

  //! Loop over system dofs for this thread.
  for (int i = start_gd; i<end_gd; i++)
  {
    rows_[i] = mycols.size();

    neib_dofs.clear();
    //! Check for nodes.
    if (i<global_dimension_nodes)
    {
      //! Convert int index to node index.
      typename Mesh::Node::index_type ii;
      hMesh_->to_index(ii, i);
      //! Get neighboring cells for node.
      hMesh_->get_elems(ca, ii);
    }
    else if (i<global_dimension_nodes+global_dimension_add_nodes)
    {
      //! Check for additional nodes at edges.
      //! Get neighboring cells for node.
      //      const int ii = i-global_dimension_nodes;
      //      hMesh_->get_cells(ca, typename Mesh::Edge::index_type(ii));
    }
    else
    {
      //! Check for derivatives - to do.
    }
	
    for(unsigned int j = 0; j < ca.size(); j++)
    {
      //! Get neighboring nodes.
      hMesh_->get_nodes(na, ca[j]);

      for(unsigned int k = 0; k < na.size(); k++)
      {
        // Must cast to (int) for SGI compiler. :-(
        neib_dofs.push_back((int)(na[k]));
      }

      //! Check for additional nodes at edges.
      if (global_dimension_add_nodes)
      {
        //! Get neighboring edges.
        hMesh_->get_edges(ea, ca[j]);

        for(unsigned int k = 0; k < ea.size(); k++)
          neib_dofs.push_back(global_dimension + ea[k]);
      }
    }
	
    sort(neib_dofs.begin(), neib_dofs.end());

#ifdef BUILDFEM_DEBUG
    cerr << i << ' ';
#endif

    for (unsigned int j=0; j<neib_dofs.size(); j++)
    {
      if (j == 0 || neib_dofs[j] != (int)mycols.back())
      {
        mycols.push_back(neib_dofs[j]);
#ifdef BUILDFEM_DEBUG
        cerr << neib_dofs[j] << ' ';
#endif
      }
    }
#ifdef BUILDFEM_DEBUG
    cerr << endl;
#endif
  }

  colIdx_[proc] = mycols.size();

  //! Check point.
  barrier_.wait(np_);

  int st = 0;
  if (proc == 0)
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

  //! Check point.
  barrier_.wait(np_);

  //! Updating global column by each of the processors.
  const int s = colIdx_[proc];
  const int n = mycols.size();

  for(int i=0; i<n; i++)
    allCols_[i+s] = mycols[i];

  for(int i = start_gd; i<end_gd; i++)
    rows_[i] += s;

  //! Check point.
  barrier_.wait(np_);

  //! The main thread makes the matrix.
  if (proc == 0)
  {
    rows_[global_dimension] = st;
    pA_ = scinew SparseRowMatrix(global_dimension, global_dimension, rows_, allCols_, st);
    hA_ = pA_;
  }

  //! Check point.
  barrier_.wait(np_);

  //! Zeroing in parallel.
  const int ns = colIdx_[proc];
  const int ne = colIdx_[proc+1];
  double* a = &pA_->a[ns], *ae = &pA_->a[ne];

  while (a<ae) *a ++= 0.0;

  vector<vector<double> > ni_points;
  vector<double> ni_weights;
  vector<vector<double> > ni_derivatives;
  create_numerical_integration(ni_points, ni_weights, ni_derivatives);

  vector<double> lsml; //!< line of local stiffnes matrix
  lsml.resize(local_dimension);

#ifdef BUILDFEM_DEBUG
{
  typename Mesh::Elem::iterator ea,eb;
  hMesh_->begin(ea);
  hMesh_->end(eb);
	
  for (;ea!=eb;++ea) {
    for (int i=0; i<4; i++) {
      build_local_matrix(*ea, i, lsml, ni_points, ni_weights, ni_derivatives);
      for(unsigned int j=0 ; j<lsml.size(); j++)
        cerr << lsml[j] << ' ';
      cerr << endl;
    }
    cerr << endl;
  }
}
#endif

  //! Loop over system dofs for this thread.
  for (int i = start_gd; i<end_gd; i++)
  {
    if (i < global_dimension_nodes)
    {
      //! Convert int index to node index.
      typename Mesh::Node::index_type ii;
      hMesh_->to_index(ii, i);
      //! Get neighboring cells for node.
      hMesh_->get_elems(ca, ii);
    }
    else if (i < global_dimension_nodes + global_dimension_add_nodes)
    {
      //! Check for additional nodes at edges.
      //! Get neighboring cells for additional nodes.
      //      const int ii=i-global_dimension_nodes;
      //      hMesh_->get_cells(ca, typename Mesh::Edge::index_type(ii));
    }
    else
    {
      //! Check for derivatives - to do.
    }
	
    //! Loop over elements attributed elements.
    for (int j = 0; j < (int)ca.size(); j++)
    {
      neib_dofs.clear();
      int dofi = -1; //!< Index of global dof in local dofs.
      hMesh_->get_nodes(na, ca[j]); //!< Get neighboring nodes.
      for(int k = 0; k < (int)na.size(); k++)
      {
        // Must cast to (int) for SGI compiler :-(
	neib_dofs.push_back((int)(na[k]));
	if ((int)na[k] == i)
	  dofi = neib_dofs.size()-1;
      }
      //! Check for additional nodes at edges.
      if (global_dimension_add_nodes)
      {
	hMesh_->get_edges(ea, ca[j]); //!< Get neighboring edges.
	for(int k = 0; k < (int)ea.size(); k++)
        {
	  neib_dofs.push_back(global_dimension + ea[k]);
	  if ((int)na[k] == i)
	    dofi = neib_dofs.size() - 1;
	}
      }

      ASSERT(dofi!=-1);
      ASSERT((int)neib_dofs.size() == local_dimension);

      build_local_matrix(ca[j], dofi, lsml, ni_points,
                         ni_weights, ni_derivatives);
#ifdef BUILDFEM_DEBUG
      cerr << i << ", " << j << " (" << (int) ca[j] << "," << dofi << ") ";

      for(unsigned int j=0 ; j<lsml.size(); j++)
	cerr << lsml[j] << ' ';
      cerr << endl;
#endif
      add_lcl_gbl(i, neib_dofs, lsml);
    }
  }

#ifdef BUILDFEM_DEBUG
  for (int i=start_gd; i<end_gd; i++)
  {
    double sum=0.0, sumabs=0.0;
    for (int j=0; j<global_dimension; j++)
    {
      sum += pA_->get(i,j);
      sumabs += fabs(pA_->get(i,j));
    }
    cerr << i << '\t' << "Sum: " << sum << "  SumAbs:" << sumabs << endl;
  }
#endif

  barrier_.wait(np_);
}


} // end namespace BioPSE
