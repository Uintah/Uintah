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
 *  ApplyFEMCurrentSource.cc: Builds the RHS of the FE matrix for
 *  current sources
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   May 1999
 *  Modified by:
 *   Alexei Samsonov
 *   March 2001
 *  Copyright (C) 1999, 2001 SCI Group
 *
 *   Lorena Kreda, Northeastern University, November 2003
 */

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/CurveField.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Trig.h>
#include <Core/Math/MiscMath.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <Packages/BioPSE/Core/Algorithms/NumApproximation/ReferenceElement.h>

namespace BioPSE {

using namespace SCIRun;

class ApplyFEMCurrentSource : public Module {
private:

  void execute_dipole();

  void execute_sources_and_sinks();

  GuiInt sourceNodeTCL_;
  GuiInt sinkNodeTCL_;
  GuiString modeTCL_;

public:
  //! Constructor/Destructor
  ApplyFEMCurrentSource(GuiContext *context);
  virtual ~ApplyFEMCurrentSource();

  //! Public methods
  virtual void execute();
};

DECLARE_MAKER(ApplyFEMCurrentSource)


ApplyFEMCurrentSource::ApplyFEMCurrentSource(GuiContext *context)
  : Module("ApplyFEMCurrentSource", context, Filter, "Forward", "BioPSE"),
    sourceNodeTCL_(context->subVar("sourceNodeTCL")),
    sinkNodeTCL_(context->subVar("sinkNodeTCL")),
    modeTCL_(context->subVar("modeTCL"))
{
}


ApplyFEMCurrentSource::~ApplyFEMCurrentSource()
{
}


void
ApplyFEMCurrentSource::execute()
{
  if (modeTCL_.get() == "dipole")
  {
    execute_dipole();
  }
  else if (modeTCL_.get() == "sources and sinks")
  {
    execute_sources_and_sinks();
  }
  else
  {
    error("Unreachable code, bad mode.");
  }
}


void
ApplyFEMCurrentSource::execute_dipole()
{
  FieldIPort *iportField = (FieldIPort *)get_iport("Mesh");
  FieldIPort *iportSource = (FieldIPort *)get_iport("Sources");
  MatrixIPort *iportRhs = (MatrixIPort *)get_iport("Input RHS");

  MatrixOPort *oportRhs = (MatrixOPort *)get_oport("Output RHS");
  MatrixOPort *oportWeights = (MatrixOPort *)get_oport("Output Weights");

  // Get the input mesh.
  FieldHandle hField;
  if (!iportField->get(hField) || !hField.get_rep()) {
    error("Can't get handle to input mesh.");
    return;
  }

  // Get the input dipoles.
  FieldHandle hSource;
  if (!iportSource->get(hSource) || !hSource.get_rep()) {
    error("Can't get handle to Source field.");
    return;
  }
  PointCloudField<Vector> *hDipField = 
    dynamic_cast<PointCloudField<Vector>*> (hSource.get_rep());
  if (!hDipField)
  {
    error("Sources field is not of type PointCloudField<Vector>.");
    return;
  }
	
  TetVolMesh *hTetMesh = 0;
  HexVolMesh *hHexMesh = 0;
  TriSurfMesh *hTriMesh = 0;
  if ((hTetMesh = dynamic_cast<TetVolMesh*>(hField->mesh().get_rep())))
  {
    remark("Input is a 'TetVolField'");
  }
  else if ((hHexMesh = dynamic_cast<HexVolMesh*>(hField->mesh().get_rep())))
  {    
    remark("Input is a 'HexVolField'");
  }
  else if ((hTriMesh = dynamic_cast<TriSurfMesh*> (hField->mesh().get_rep())))
  {
    remark("Input is a 'TriSurfField'");
  }
  else
  {
    error("Supplied field is not 'TetVolField' or 'HexVolField' or 'TriSurfField'");
    return;
  }

  int nsize = 0;
  if (hTetMesh)
  {
    TetVolMesh::Node::size_type nsizeTet; hTetMesh->size(nsizeTet);
    nsize = nsizeTet;
  }
  else if (hHexMesh)
  {
    HexVolMesh::Node::size_type nsizeHex; hHexMesh->size(nsizeHex);
    nsize = nsizeHex;
  }
  else if (hTriMesh)
  {
    TriSurfMesh::Node::size_type nsizeTri; hTriMesh->size(nsizeTri);
    nsize = nsizeTri;
  }

  if (nsize <= 0)
  {
    error("Input mesh has zero size");
    return;
  }

  // If the user passed in a vector the right size, copy it into ours.
  ColumnMatrix* rhs = 0;
  MatrixHandle  hRhsIn;
  if (iportRhs->get(hRhsIn) && hRhsIn.get_rep())
  {
    if (hRhsIn->nrows() == nsize && hRhsIn->ncols() == 1)
    {
      rhs = scinew ColumnMatrix(nsize);
      string units;
      if (hRhsIn->get_property("units", units))
        rhs->set_property("units", units, false);

      for (int i=0; i < nsize; i++)
      {
        rhs->put(i, hRhsIn->get(i, 0));
      }
    }
    else
    {
      warning("The supplied RHS doesn't correspond to the input mesh in size.  Creating empty one.");
    }
  }
  if (rhs == 0)
  {
    rhs = scinew ColumnMatrix(nsize);
    rhs->set_property("units", string("volts"), false);
    rhs->zero();
  }

  // Process mesh.
  if (hTetMesh)
  {
    hTetMesh->synchronize(Mesh::LOCATE_E);
	
    //! Computing contributions of dipoles to RHS
    PointCloudMesh::Node::iterator ii;
    PointCloudMesh::Node::iterator ii_end;
    hDipField->get_typed_mesh()->begin(ii);
    hDipField->get_typed_mesh()->end(ii_end);
    vector<double> weights;
    for (; ii != ii_end; ++ii)
    {
      // Position of the dipole.
      Point pos;
      hDipField->get_typed_mesh()->get_point(pos, *ii);
      // Correct unit of dipole moment -> should be checked.
      const Vector &dir = hDipField->value(*ii);

      TetVolMesh::Cell::index_type loc;
      if (hTetMesh->locate(loc, pos))
      {
        msgStream_ << "Source pos="<<pos<<" dir="<<dir<<
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
		
        Vector g1, g2, g3, g4;
        hTetMesh->get_gradient_basis(loc, g1, g2, g3, g4);
		
        const double s1 = Dot(g1, dir);
        const double s2 = Dot(g2, dir);
        const double s3 = Dot(g3, dir);
        const double s4 = Dot(g4, dir);
		
        TetVolMesh::Node::array_type cell_nodes;
        hTetMesh->get_nodes(cell_nodes, loc);
        (*rhs)[cell_nodes[0]] += s1;
        (*rhs)[cell_nodes[1]] += s2;
        (*rhs)[cell_nodes[2]] += s3;
        (*rhs)[cell_nodes[3]] += s4;
      }
      else
      {
        msgStream_ << "Dipole: "<< pos <<" not located within mesh!"<<endl;
      }
    }

    ColumnMatrix* w = scinew ColumnMatrix(weights.size());
    for (int i=0; i< (int)weights.size(); i++) { w->put(i, weights[i]); }

    oportWeights->send(MatrixHandle(w));
    oportRhs->send(MatrixHandle(rhs));
  }
  else if (hHexMesh)
  {
    hHexMesh->synchronize(Mesh::LOCATE_E);
	
    //! Computing contributions of dipoles to RHS
    PointCloudMesh::Node::iterator ii;
    PointCloudMesh::Node::iterator ii_end;
    hDipField->get_typed_mesh()->begin(ii);
    hDipField->get_typed_mesh()->end(ii_end);
    //vector<double> weights;
    ReferenceElement rE;
    for (; ii != ii_end; ++ii)
    {
      // Position of the dipole.
      Point pos;
      hDipField->get_typed_mesh()->get_point(pos, *ii);
      // Correct unit of dipole moment -> should be checked.
      const Vector &dir = hDipField->value(*ii);

      HexVolMesh::Cell::index_type loc;
      if (hHexMesh->locate(loc, pos))
      {
        msgStream_ << "Source p="<<pos<<" dir="<< dir <<
          " found in elem "<< loc <<endl;
      }
      else
      {
        msgStream_ << "Dipole: "<< pos <<
          " not located within mesh!"<<endl;
      }

      // Get dipole in reference element.
      HexVolMesh::Node::array_type n_array;
      hHexMesh->get_nodes(n_array, loc);
      Point a, b;
      hHexMesh->get_point(a, n_array[0]);
      hHexMesh->get_point(b, n_array[6]);
      const double drx = rE.isp1(pos.x(), a.x(), b.x());
      const double dry = rE.isp2(pos.y(), a.y(), b.y());
      const double drz = rE.isp3(pos.z(), a.z(), b.z());

      // Update rhs
      for (int i=0; i <8; i++)
      {
        const double val =
          dir[0] * rE.dphidx(i, drx, dry, drz) / rE.dpsi1dx(a.x(), b.x()) +
          dir[1] * rE.dphidy(i, drx, dry, drz) / rE.dpsi2dy(a.y(), b.y()) +
          dir[2] * rE.dphidz(i, drx, dry, drz) / rE.dpsi3dz(a.z(), b.z());
        rhs->put((int)n_array[i], val);
      }
    }
    oportRhs->send(MatrixHandle(rhs));
  }
  else if (hTriMesh)
  {
    hTriMesh->synchronize(Mesh::LOCATE_E);

    //! Computing contributions of dipoles to RHS.
    PointCloudMesh::Node::iterator ii;
    PointCloudMesh::Node::iterator ii_end;
    hDipField->get_typed_mesh()->begin(ii);
    hDipField->get_typed_mesh()->end(ii_end);
    vector<double> weights;
    for (; ii != ii_end; ++ii)
    {
      // Position of the dipole.
      Point pos;
      hDipField->get_typed_mesh()->get_point(pos, *ii);
      // Correct unit of dipole moment -> should be checked.
      const Vector &dir = hDipField->value(*ii);

      TriSurfMesh::Face::index_type loc;
      if (hTriMesh->locate(loc, pos))
      {
        msgStream_ << "Source pos="<<pos<<" dir="<<dir<<
          " found in elem "<< loc <<endl;

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
	
        Vector g1, g2, g3;
        hTriMesh->get_gradient_basis(loc, g1, g2, g3);

        const double s1 = Dot(g1, dir);
        const double s2 = Dot(g2, dir);
        const double s3 = Dot(g3, dir);
		
        TriSurfMesh::Node::array_type face_nodes;
        hTriMesh->get_nodes(face_nodes, loc);
        (*rhs)[face_nodes[0]] += s1;
        (*rhs)[face_nodes[1]] += s2;
        (*rhs)[face_nodes[2]] += s3;
      }
      else
      {
        msgStream_ << "Dipole: "<< pos <<" not located within mesh!"<<endl;
      }
    } // end for loop

    ColumnMatrix* w = scinew ColumnMatrix(weights.size());
    for (int i=0; i < (int)weights.size(); i++) { w->put(i, weights[i]); }

    oportWeights->send(MatrixHandle(w));
    oportRhs->send(MatrixHandle(rhs));
  }
}


void
ApplyFEMCurrentSource::execute_sources_and_sinks()
{
  FieldIPort *iportField = (FieldIPort *)get_iport("Mesh");
  FieldIPort *iportSource = (FieldIPort *)get_iport("Sources");
  MatrixIPort *iportMapping = (MatrixIPort *)get_iport("Mapping");
  MatrixIPort *iportRhs = (MatrixIPort *)get_iport("Input RHS");

  MatrixOPort *oportRhs = (MatrixOPort *)get_oport("Output RHS");

  //! Obtaining handles to computation objects
  FieldHandle hField;
  if (!iportField->get(hField) || !hField.get_rep()) {
    error("Can't get handle to input mesh.");
    return;
  }

  TetVolMesh *hTetMesh = 0;
  HexVolMesh *hHexMesh = 0;
  TriSurfMesh *hTriMesh = 0;
  if ((hTetMesh = dynamic_cast<TetVolMesh*>(hField->mesh().get_rep())))
  {
    remark("Input is a 'TetVolField'");
  }
  else if ((hHexMesh = dynamic_cast<HexVolMesh*> (hField->mesh().get_rep())))
  {
    remark("Input is a 'HexVolField'");
  }
  else if ((hTriMesh = dynamic_cast<TriSurfMesh*> (hField->mesh().get_rep())))
  {
    remark("Input is a 'TriSurfField'");
  }
  else
  {
    error("Supplied field is not 'TetVolField', 'TriSurfField', or 'HexVolField'");
    return;
  }

  int nsize = 0;
  if (hTetMesh)
  {
    TetVolMesh::Node::size_type nsizeTet; hTetMesh->size(nsizeTet);
    nsize = nsizeTet;
  }
  else if (hHexMesh)
  {
    HexVolMesh::Node::size_type nsizeHex; hHexMesh->size(nsizeHex);
    nsize = nsizeHex;
  }
  else if (hTriMesh)
  {
    TriSurfMesh::Node::size_type nsizeTri; hTriMesh->size(nsizeTri);
    nsize = nsizeTri;
  }

  if (nsize <= 0)
  {
    error("Input mesh has zero size");
    return;
  }

  // If the user passed in a vector the right size, copy it into ours.
  ColumnMatrix* rhs = 0;
  MatrixHandle  hRhsIn;
  if (iportRhs->get(hRhsIn) && hRhsIn.get_rep())
  {
    if (hRhsIn->ncols() == 1 && hRhsIn->nrows() == nsize)
    {
      rhs = scinew ColumnMatrix(nsize);
      string units;
      if (hRhsIn->get_property("units", units))
        rhs->set_property("units", units, false);

      for (int i=0; i < nsize; i++)
      {
        rhs->put(i, hRhsIn->get(i, 0));
      }
    }
    else
    {
      warning("The supplied RHS doesn't correspond to the input mesh in size.  Creating empty one.");
    }
  }
  if (rhs == 0)
  {
    rhs = scinew ColumnMatrix(nsize);
    rhs->set_property("units", string("volts"), false);
    rhs->zero();
  }

  // process mesh
  if (hTetMesh || hHexMesh)
  {
    MatrixHandle hMapping;
    iportMapping->get(hMapping);
    FieldHandle hSource;
    iportSource->get(hSource);
	
    unsigned int sourceNode = Max(sourceNodeTCL_.get(), 0);
    unsigned int sinkNode = Max(sinkNodeTCL_.get(), 0);
	
    // if we have an Mapping matrix and a Source field and all types are good,
    //  hCurField will be valid after this block
    PointCloudField<double> *hCurField = 0;
    if (hMapping.get_rep() && hSource.get_rep())
    {
      hCurField = dynamic_cast<PointCloudField<double>*> (hSource.get_rep());
      if (!hCurField)
      {
        error("Can only use a PointCloudField<double> as source when using an Mapping matrix and a Source field -- this mode is for specifying current densities");
        return;
      }
      if (hCurField->data_size() != (unsigned int)hMapping->nrows())
      {
        error("Source field and Mapping matrix size mismatch.");
        return;
      }
      if (nsize != hMapping->ncols())
      {
        error("Mesh field and Mapping matrix size mismatch.");
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
        error("SourceNode or SinkNode was out of mesh range.");
        return;
      }
      (*rhs)[sourceNode] += -1;
      (*rhs)[sinkNode] += 1;
      oportRhs->send(MatrixHandle(rhs));
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
        error("SourceNode or SinkNode was out of mapping range.");
        return;
      }
      (*rhs)[sourceNode] += -1;
      (*rhs)[sinkNode] += 1;
      oportRhs->send(MatrixHandle(rhs));
      return;
    }

    PointCloudMesh::Node::iterator ii;
    PointCloudMesh::Node::iterator ii_end;
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
        (*rhs)[cc?cc[j*ccstride]:j] += vv[j*ccstride] * currentDensity;
      }
    }

    oportRhs->send(MatrixHandle(rhs));
  } 
  else if (hTriMesh)
  {
    MatrixHandle hMapping;
    iportMapping->get(hMapping);
    unsigned int sourceNode = Max(sourceNodeTCL_.get(), 0);
    unsigned int sinkNode = Max(sinkNodeTCL_.get(), 0);

    if (hMapping.get_rep())
    {
      if (nsize != hMapping->ncols())
      {
        error("Mesh field and Mapping matrix size mismatch.");
        return;
      }

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
        error("SourceNode or SinkNode was out of mapping range.");
        return;
      }
    }

    if (sourceNode >= (unsigned int)nsize || sinkNode >= (unsigned int)nsize)
    {
      error("SourceNode or SinkNode was out of mesh range.");
      return;
    }

    msgStream_ << "sourceNode="<<sourceNode<<" sinkNode="<<sinkNode<<"\n";
    (*rhs)[sourceNode] += -1;
    (*rhs)[sinkNode] += 1;

    //! Sending result
    oportRhs->send(MatrixHandle(rhs));
  }
}


} // End namespace BioPSE
