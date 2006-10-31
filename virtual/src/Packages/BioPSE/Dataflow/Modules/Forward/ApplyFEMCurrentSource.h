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
  virtual void execute(ProgressReporter *PR, FieldHandle &hField, FieldHandle &hSource, MatrixHandle &hMapping, bool dipole, unsigned int sourceNode, unsigned int sinkNode, ColumnMatrix** rhs, ColumnMatrix **w) = 0;
    
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

  void execute_dipole(ProgressReporter *PR, FieldHandle &hField, FieldHandle &hSource, ColumnMatrix** rhs, ColumnMatrix **w)
  {
    FMT *mesh=dynamic_cast<FMT *>(hField->mesh().get_rep());
    ASSERT(mesh);

    mesh->synchronize(Mesh::LOCATE_E);
    typename FMT::Node::size_type nsize; 
    mesh->size(nsize);

    if (*rhs == 0)
    {
      *rhs = scinew ColumnMatrix(nsize);
      (*rhs)->set_property("units", string("volts"), false);
      (*rhs)->zero();
    }

    PCVecField *hDipField = dynamic_cast<PCVecField*> (hSource.get_rep());
    if (!hDipField)
    {
      PR->error("Sources field is not of type PointCloudField<Vector>.");
      return;
    }

    //! Computing contributions of dipoles to RHS
    PCMesh::Node::iterator ii;
    PCMesh::Node::iterator ii_end;
    hDipField->get_typed_mesh()->begin(ii);
    hDipField->get_typed_mesh()->end(ii_end);
    vector<double> weights;
    vector<double> coords;
    vector<double> wd(MBT::dofs()*MBT::domain_dimension());

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
        PR->msg_stream() << "Source pos="<<pos<<" dir="<<dir<<
          " found in elem "<<loc<<endl;

        if (fabs(dir.x()) > 0.000001)
        {
          weights.push_back(loc*3);
          weights.push_back(dir.x());
        }
        if (fabs(dir.y()) > 0.000001)
        {
          weights.push_back(loc*3+1);
          weights.push_back(dir.y());
        }
        if (fabs(dir.z()) > 0.000001)
        {
          weights.push_back(loc*3+2);
          weights.push_back(dir.z());
        }
	

        if (!mesh->get_coords(coords, pos, loc))
        {
          // internal error
          return;
        }
        //	cerr << "coords: " << coords[0] << " " << coords[1] << " " << coords[2] << endl;
	  

        MBT::get_derivate_weights(coords, &wd[0]);
	  
        vector<Point> Jv;
        mesh->derivate(coords, loc, Jv);

        DenseMatrix J(3, Jv.size());
        int i=0;
        vector<Point>::iterator iter = Jv.begin();
        while(iter != Jv.end()) {
          Point &p = *iter++;
          J.put(i, 0, p.x());
          J.put(i, 1, p.y());
          J.put(i, 2, p.z());
          ++i;
        }

        Vector dirl;
        dirl.x(J.get(0,0)*dir.x()+J.get(0,1)*dir.y()+J.get(0,2)*dir.z());
        dirl.y(J.get(1,0)*dir.x()+J.get(1,1)*dir.y()+J.get(1,2)*dir.z());
        dirl.z(J.get(2,0)*dir.x()+J.get(2,1)*dir.y()+J.get(2,2)*dir.z());

        typename FMT::Node::array_type cell_nodes;
        mesh->get_nodes(cell_nodes, loc);
        for(int i=0; i<MBT::number_of_vertices(); i++) {
          Vector g;
          g.x(wd[i]);
          g.y(wd[i+MBT::dofs()]);
          g.z(wd[i+2*MBT::dofs()]);
          const double dp=Dot(g,dirl);

          // cerr << g << '\t' << dp << endl;

          if (i<MBT::number_of_mesh_vertices()) {
            (**rhs)[cell_nodes[i]] += dp;
          }
          else {
            // to do
          }
        }	  
      }
      else
      {
        PR->msg_stream() << "Dipole: "<< pos <<" not located within mesh!"<<endl;
      }
    }

    *w = scinew ColumnMatrix(weights.size());
    for (int i=0; i< (int)weights.size(); i++) 
      (*w)->put(i, weights[i]); 
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
      if (nsize != hMapping->ncols())
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
      if ((int)sourceNode >= nsize || (int)sinkNode >= nsize)
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

  virtual void execute(ProgressReporter *PR, FieldHandle &hField, FieldHandle &hSource, MatrixHandle &hMapping, bool dipole, unsigned int sourceNode, unsigned int sinkNode, ColumnMatrix** rhs, ColumnMatrix **w) 
  {
    if (dipole)
      execute_dipole(PR, hField, hSource, rhs, w);
    else 
      execute_sources_and_sinks(PR, hField, hSource, hMapping, sourceNode, sinkNode, rhs);
  }
};


} // End namespace BioPSE
