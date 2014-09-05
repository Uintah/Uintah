/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  
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
 *  ApplyFEMCurrentSource.cc: Builds the RHS of the FE matrix for
 *  current sources
 *
 *  Written by:
 *   David Weinstein, University of Utah, May 1999
 *   Alexei Samsonov, March 2001
 *   Lorena Kreda, Northeastern University, November 2003
 *   Frank B. Sachse, February 2006
 */


#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Basis/Bases.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>

namespace BioPSE {

using namespace SCIRun;
  
class ApplyFEMCurrentSourceAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(ProgressReporter *PR, FieldHandle &hField, FieldHandle &hSource, MatrixHandle &hMapping, bool dipole, unsigned int sourceNode, unsigned int sinkNode, ColumnMatrix** rhs, SparseRowMatrix **w) = 0;
    
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
                                            const TypeDescription *mtd,
                                            const TypeDescription *btd,
                                            const TypeDescription *dtd);
};


template<class FIELD>
class ApplyFEMCurrentSourceAlgoT : public ApplyFEMCurrentSourceAlgo 
{
  typedef PointCloudMesh<ConstantBasis<Point> > PCMesh;
  typedef ConstantBasis<Vector> ConVecBasis;
  typedef GenericField<PCMesh, ConVecBasis, vector<Vector> > PCVecField; 
  typedef ConstantBasis<double> ConScaBasis;
  typedef GenericField<PCMesh, ConScaBasis, vector<double> > PCScaField;
 
  typedef typename FIELD::basis_type FBT;
  typedef typename FIELD::mesh_type FMT;
  typedef typename FMT::basis_type MBT;

  static bool ud_pair_less(const std::pair<unsigned int, double> &a,
                           const std::pair<unsigned int, double> &b)
  {
      return a.first < b.first;
  };

  void execute_dipole(ProgressReporter *pr, FieldHandle &hField, FieldHandle &hSource, ColumnMatrix** rhs, SparseRowMatrix **w)
  {
    FMT *mesh=dynamic_cast<FMT *>(hField->mesh().get_rep());
    ASSERT(mesh);

    mesh->synchronize(Mesh::LOCATE_E);
    typename FMT::Node::size_type nsize; 
    mesh->size(nsize);

    typename FMT::Node::array_type nodes;
    
    typename FMT::Elem::size_type sz;
    mesh->size(sz);

    if (*rhs == 0)
    {
      *rhs = scinew ColumnMatrix(nsize);
      (*rhs)->set_property("units", string("volts"), false);
      (*rhs)->zero();
    }

    PCVecField *hDipField = dynamic_cast<PCVecField*> (hSource.get_rep());
    if (!hDipField)
    {
      pr->error("Sources field is not of type PointCloudField<Vector>.");
      return;
    }

    //! Computing contributions of dipoles to RHS
    PCMesh::Node::iterator ii;
    PCMesh::Node::iterator ii_end;
    hDipField->get_typed_mesh()->begin(ii);
    hDipField->get_typed_mesh()->end(ii_end);
    vector<pair<unsigned int, double> > weights;
    vector<double> coords;
    unsigned int dofs = MBT::dofs();
    int dim = mesh->get_basis().domain_dimension();
    double J[9], Ji[9];
    double grad[3];
    vector<double> wd(dofs*dim);

    for (; ii != ii_end; ++ii)
    {
      // Position of the dipole.
      Point pos;
      hDipField->get_typed_mesh()->get_point(pos, *ii);
      // Correct unit of dipole moment -> should be checked.
      const Vector &dir = hDipField->value(*ii);
    
      //	cerr << "Pos " << pos << endl;
      //	cerr << "Dir " << dir << endl;


      
      typename FMT::Cell::index_type loc;
      if (mesh->locate(loc, pos))
      {
        pr->msg_stream() << "Source pos="<<pos<<" dir="<<dir<<
          " found in elem "<<loc<<endl;

        if (fabs(dir.x()) > 0.000001)
        {
          weights.push_back(pair<unsigned int, double>(loc*3+0, dir.x()));
        }
        if (fabs(dir.y()) > 0.000001)
        {
          weights.push_back(pair<unsigned int, double>(loc*3+1, dir.y()));
        }
        if (fabs(dir.z()) > 0.000001)
        {
          weights.push_back(pair<unsigned int, double>(loc*3+2, dir.z()));
        }
	

        if (!mesh->get_coords(coords, pos, loc))
        {
          // internal error
          return;
        }
        //	cerr << "coords: " << coords[0] << " " << coords[1] << " " << coords[2] << endl;
	  
        // get the mesh Jacobian for the element.
        vector<Point> Jv;
        mesh->derivate(coords, loc, Jv);



        // TO DO:
        // Squeeze out more STL vector operations as they require memory
        // being reserved, we should have simple C style arrays which are build
        // directly on the stack. As this is mostly used for volume data, it has 
        // only been optimized for this kind of data

        ASSERT(dim >=1 && dim <=3);
        if (dim == 3)
        {
          J[0] = Jv[0].x();
          J[1] = Jv[0].y();
          J[2] = Jv[0].z();
          J[3] = Jv[1].x();
          J[4] = Jv[1].y();
          J[5] = Jv[1].z();
          J[6] = Jv[2].x();
          J[7] = Jv[2].y();
          J[8] = Jv[2].z();        
        }
        else if (dim == 2)
        {
          Vector J2 = Cross(Jv[0].asVector(),Jv[1].asVector());
          J2.normalize();
          J[0] = Jv[0].x();
          J[1] = Jv[0].y();
          J[2] = Jv[0].z();
          J[3] = Jv[1].x();
          J[4] = Jv[1].y();
          J[5] = Jv[1].z();
          J[6] = J2.x();
          J[7] = J2.y();
          J[8] = J2.z();    
        }
        else
        {
          // The same thing as for the surface but then for a curve.
          // Again this matrix should have a positive determinant as well. It actually
          // has an internal degree of freedom, which is not being used.
          Vector J1, J2;
          Jv[0].asVector().find_orthogonal(J1,J2);
          J[0] = Jv[0].x();
          J[1] = Jv[0].y();
          J[2] = Jv[0].z();
          J[3] = J1.x();
          J[4] = J1.y();
          J[5] = J1.z();
          J[6] = J2.x();
          J[7] = J2.y();
          J[8] = J2.z();          
        }

        InverseMatrix3x3(J,Ji);    
    
        MBT::get_derivate_weights(coords, &wd[0]);
	  
        mesh->get_nodes(nodes, loc);
        
        for(int i=0; i<MBT::number_of_vertices(); i++) 
        {
          grad[0] = wd[i]*Ji[0]+wd[i+dofs]*Ji[1]+wd[i+2*dofs]*Ji[2];
          grad[1] = wd[i]*Ji[3]+wd[i+dofs]*Ji[4]+wd[i+2*dofs]*Ji[5];
          grad[2] = wd[i]*Ji[6]+wd[i+dofs]*Ji[7]+wd[i+2*dofs]*Ji[8];
          
          (**rhs)[nodes[i]] += grad[0]*dir.x() + grad[1]*dir.y() + grad[2]*dir.z();
        }

      }
      else
      {
        pr->error("Dipole not located within mesh");
      }
    }

    int *rr = scinew int[2]; rr[0] = 0; rr[1] = (int)weights.size();
    int *cc = scinew int[weights.size()];
    double *dd = scinew double[weights.size()];
    std::sort(weights.begin(), weights.end(), ud_pair_less);
    for (unsigned int i=0; i < weights.size(); i++)
    {
      cc[i] = weights[i].first;
      dd[i] = weights[i].second;
    }
    
    *w = scinew SparseRowMatrix(1, 3*sz, rr, cc, (int)weights.size(), dd);
  }


  void execute_sources_and_sinks(ProgressReporter *PR, FieldHandle &hField, FieldHandle &hSource, MatrixHandle &hMapping, unsigned int sourceNode, unsigned int sinkNode, ColumnMatrix** rhs)
  {   
    FMT *mesh=dynamic_cast<FMT *>(hField->mesh().get_rep());
    ASSERT(mesh);

    mesh->synchronize(Mesh::LOCATE_E);
    typename FMT::Node::size_type nsize; 
    mesh->size(nsize);

    if (*rhs == 0)
    {
      (*rhs) = scinew ColumnMatrix(nsize);
      (*rhs)->set_property("units", string("volts"), false);
      (*rhs)->zero();
    }

    // if we have an Mapping matrix and a Source field and all types are good,
    // hCurField will be valid after this block
    PCScaField *hCurField = 0;
    if (hMapping.get_rep() && hSource.get_rep())
    {
      hCurField = dynamic_cast<PCScaField*> (hSource.get_rep());
      if (!hCurField)
      {
        PR->error("Can only use a PointCloudField<double> as source when using an Mapping matrix and a Source field -- this mode is for specifying current densities");
        return;
      }
      if (hCurField->data_size() != (unsigned int)hMapping->nrows())
      {
        PR->error("Source field and Mapping matrix size mismatch.");
        return;
      }
      if (static_cast<unsigned int>(nsize) != static_cast<unsigned int>(hMapping->ncols()))
      {
        PR->error("Mesh field and Mapping matrix size mismatch.");
        return;
      }
    }
	

    // if we have don't have a Mapping matrix, use the source/sink indices
    // directly as volume nodes
      
    // if we do have a Mapping matrix, but we don't have a Source field,
    // then the source/sink indices refer to the PointCloud, so use the
    // Mapping matrix to get their corresponding volume node indices.
      
    // if we have a Mapping matrix AND a Source field, then ignore the
    // source/sink indices.  The Mapping matrix defines how the PointCloud
    //  nodes map to volume mesh nodes, and the Source field gives a
    // scalar quantity (current density) for each source.

    if (!hMapping.get_rep())
    {
      if (sourceNode >= nsize || sinkNode >= nsize)
      {
        PR->error("SourceNode or SinkNode was out of mesh range.");
        return;
      }
      (**rhs)[sourceNode] += -1;
      (**rhs)[sinkNode] += 1;
      return;
    }
      
    if (!hCurField)
    {
      if (sourceNode < (unsigned int)hMapping->nrows() &&
          sinkNode < (unsigned int)hMapping->nrows())
      {
        int *cc;
        double *vv;
        int ccsize;
        int ccstride;
        hMapping->getRowNonzerosNoCopy(sourceNode, ccsize, ccstride, cc, vv);
        ASSERT(ccsize);
        sourceNode = cc?cc[0]:0;
        hMapping->getRowNonzerosNoCopy(sinkNode, ccsize, ccstride, cc, vv);
        ASSERT(ccsize);
        sinkNode = cc?cc[0]:0;
      }
      else
      {
        PR->error("SourceNode or SinkNode was out of mapping range.");
        return;
      }
      (**rhs)[sourceNode] += -1;
      (**rhs)[sinkNode] += 1;
      return;
    }
      
    PCMesh::Node::iterator ii;
    PCMesh::Node::iterator ii_end;
    hCurField->get_typed_mesh()->begin(ii);
    hCurField->get_typed_mesh()->end(ii_end);
    for (; ii != ii_end; ++ii)
    {
      double currentDensity;
      hCurField->value(currentDensity, *ii);
	
      int *cc;
      double *vv;
      int ccsize;
      int ccstride;
	
      hMapping->getRowNonzerosNoCopy((int)(*ii), ccsize, ccstride, cc, vv);
	
      for (int j=0; j < ccsize; j++)
      {
        (**rhs)[cc?cc[j*ccstride]:j] += vv[j*ccstride] * currentDensity;
      }
    }  
  } 
    
public:
  ApplyFEMCurrentSourceAlgoT() {};
  virtual ~ApplyFEMCurrentSourceAlgoT() {};

  virtual void execute(ProgressReporter *PR, FieldHandle &hField, FieldHandle &hSource, MatrixHandle &hMapping, bool dipole, unsigned int sourceNode, unsigned int sinkNode, ColumnMatrix** rhs, SparseRowMatrix **w) 
  {
    if (dipole)
      execute_dipole(PR, hField, hSource, rhs, w);
    else 
      execute_sources_and_sinks(PR, hField, hSource, hMapping, sourceNode, sinkNode, rhs);
  }
};


} // End namespace BioPSE
