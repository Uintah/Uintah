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
 *  ApplyFEMElectodeSource.cc: Builds the RHS of the FE matrix for
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

class ApplyFEMElectrodeSource : public Module {
private:
  enum ElectrodeModelType
    {
      CONTINUUM_MODEL = 0,
      GAP_MODEL
    };

  void ProcessTriElectrodeSet( ColumnMatrix* rhs, TriSurfMeshHandle hTriMesh );
  double CalcContinuumTrigCurrent(Point p, int index, int numBoundaryNodes);
  double ComputeTheta(Point);

public:
  //! Constructor/Destructor
  ApplyFEMElectrodeSource(GuiContext *context);
  virtual ~ApplyFEMElectrodeSource();

  //! Public methods
  virtual void execute();
};

DECLARE_MAKER(ApplyFEMElectrodeSource)


ApplyFEMElectrodeSource::ApplyFEMElectrodeSource(GuiContext *context)
  : Module("ApplyFEMElectrodeSource", context, Filter, "Forward", "BioPSE")
{
}


ApplyFEMElectrodeSource::~ApplyFEMElectrodeSource()
{
}


void
ApplyFEMElectrodeSource::execute()
{
  FieldIPort *iportField = (FieldIPort *)get_iport("Mesh");
  MatrixIPort *iportRhs = (MatrixIPort *)get_iport("Input RHS");

  MatrixOPort *oportRhs = (MatrixOPort *)get_oport("Output RHS");

  //! Obtaining handles to computation objects
  FieldHandle hField;
  if (!iportField->get(hField) || !hField.get_rep()) {
    error("Can't get handle to input mesh.");
    return;
  }

  TriSurfMesh *hTriMesh = 0;
  if ((hTriMesh = dynamic_cast<TriSurfMesh*> (hField->mesh().get_rep())))
  {
    remark("Input is a 'TriSurfField'");
  }
  else
  {
    error("Only TriSurfField type is supported in electrode set mode");
    return;
  }

  TriSurfMesh::Node::size_type nsizeTri; hTriMesh->size(nsizeTri);
  const int nsize = nsizeTri;
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

  ProcessTriElectrodeSet( rhs, hTriMesh );
  
  //! Sending result
  oportRhs->send(MatrixHandle(rhs));
}


// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// ApplyFEMElectrodeSource::ProcessTriElectrodeSet
//
// Description: This method isolates a specialized block of code that
// handles the TriSurfMesh and 'Electrode Set' mode.
//
// Inputs:
//
// Returns:
//
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void
ApplyFEMElectrodeSource::ProcessTriElectrodeSet( ColumnMatrix* rhs,
                                               TriSurfMeshHandle hTriMesh )
{
  MatrixIPort *iportMapping = (MatrixIPort *)get_iport("Mapping");
  MatrixIPort *iportCurrentPattern =
    (MatrixIPort *)get_iport("Current Pattern");
  MatrixIPort *iportCurrentPatternIndex =
    (MatrixIPort *)get_iport("CurrentPatternIndex");
  MatrixIPort *iportElectrodeParams =
    (MatrixIPort *)get_iport("Electrode Parameters");
  FieldIPort *iportFieldBoundary = (FieldIPort *)get_iport("Boundary");
  MatrixIPort *iportBoundaryToMesh =
    (MatrixIPort *)get_iport("Boundary Transfer Matrix");

  int numParams=4;

  // Get the electrode parameters input vector
  // -----------------------------------------
  MatrixHandle  hElectrodeParams;

  if (!iportElectrodeParams->get(hElectrodeParams) ||
      !hElectrodeParams.get_rep())
  {
    error("Can't get handle to electrode parameters matrix.");
    return;
  }

  ColumnMatrix* electrodeParams = scinew ColumnMatrix(numParams);
  electrodeParams=dynamic_cast<ColumnMatrix*>(hElectrodeParams.get_rep());

  unsigned int electrodeModel = (unsigned int)((*electrodeParams)[0]);
  int numElectrodes           = (int) ( (*electrodeParams)[1]);
  double electrodeLen         = (*electrodeParams)[2];
  int startNodeIndex          = (int) ( (*electrodeParams)[3]);

  // Get the current pattern input
  // -----------------------------
  // These are the input currents at each electrode - later we will
  // combine with electrode information to produce an electrode
  // density. In 2D, we assume the electrode height is 1 because we
  // assume no variation in the z direction, hence we set h=1 so that
  // it doesn't influence the computation.  This input is used only
  // for models other than the continuum model

  MatrixHandle  hCurrentPattern;
  if ((!iportCurrentPattern->get(hCurrentPattern) ||
       !hCurrentPattern.get_rep()) && (electrodeModel != CONTINUUM_MODEL))
  {
    error("Can't get handle to current pattern matrix.");
    return;
  }

  // Get the current pattern index
  // -----------------------------
  // This is used for calculating the current value if the continuum
  // model is used
  MatrixHandle  hCurrentPatternIndex;
  ColumnMatrix* currPatIdx;
  int           k = 0;

  // Copy the input current index into local variable, k
  // ---------------------------------------------------
  if (iportCurrentPatternIndex->get(hCurrentPatternIndex) &&
      (currPatIdx=dynamic_cast<ColumnMatrix*>(hCurrentPatternIndex.get_rep())) &&
      (currPatIdx->nrows() == 1))
  {
    k=static_cast<int>((*currPatIdx)[0]);
  }
  else
  {
    msgStream_ << "The supplied current pattern index is not a 1x1 matrix" << endl;
  }

  // Get the FieldBoundary input
  // ---------------------------
  FieldHandle      hFieldBoundary;
  CurveMeshHandle  hBoundaryMesh;
  LockingHandle<CurveField<double> > hCurveBoundary;
  bool boundary = false;
  if ( iportFieldBoundary->get(hFieldBoundary) )
  {
    if (hFieldBoundary.get_rep())
    {
      // Check field type - this only works for CurveFields<double>
      // extracted from a TriSurf
      if ( (hFieldBoundary->get_type_name(0) == "CurveField") &&
           (hFieldBoundary->get_type_name(1) == "double") )
      {
        remark("Field boundary input is a CurveField<double>");
        hCurveBoundary = dynamic_cast<CurveField<double>*> ( hFieldBoundary.get_rep() );
        hBoundaryMesh = hCurveBoundary->get_typed_mesh();
        boundary = true;
      }
      else
      {
        remark("Supplied boundary field is not of type CurveField<double>");
      }
    }
  }
  else
  {
    msgStream_ << "There is an error in the supplied boundary field" << endl;
  }

  // If a boundary field was supplied, check for the matrix that maps
  // boundary nodes to mesh nodes.
  MatrixHandle      hBoundaryToMesh;

  if (boundary)
  {
    if ( !(iportBoundaryToMesh->get(hBoundaryToMesh) &&
           hBoundaryToMesh.get_rep()) )
    {
      // disable susequent boundary-related code if we had a problem here
      boundary = false;
    }
  }

  // Get the mapping matrix
  // --------------------
  // This is the location of the electrodes mapped onto the body
  // mesh. The presence of this input means the user is electing to
  // specify electrode locations manually rather than use an automatic
  // placement scheme selected through the electrode manager.
  MatrixHandle hMapping;
  if ( iportMapping->get(hMapping) && hMapping.get_rep())
  {
    // Unimplemented
  }
  // If electrode mapping field is not supplied, determine electrode
  // centers using number of electrodes, spacing from the electrode
  // manager and extracted field boundary
  else
  {
    // Unimplemented
  }

  // Make a local copy of the input current pattern Hold off on
  // copying the current pattern until after we check if there's an
  // mappingolated electrode field as this could influence the value of
  // numElectrodes Also, this input is not needed for the continuum
  // case and may not be present in this case.
  ColumnMatrix* currentPattern = scinew ColumnMatrix(numElectrodes);
  currentPattern=dynamic_cast<ColumnMatrix*>(hCurrentPattern.get_rep());

  // Allocate vector for the mesh-to-electrode-map
  ColumnMatrix* meshToElectrodeMap;
  TriSurfMesh::Node::size_type msize;
  hTriMesh->size(msize);
  int numMeshNodes = (int) msize;

  meshToElectrodeMap = scinew ColumnMatrix(msize);

  // Initialize meshToElectrodeMap to all -1s. -1 indicates a
  // non-electrode node; later we will identify the electrode nodes.
  for (int i = 0; i < numMeshNodes; i++)
  {
    (*meshToElectrodeMap)[i] = -1;
  }

  // TRI + ELECTRODE SET + CONTINUUM MODEL
  // -------------------------------------
  if (electrodeModel == CONTINUUM_MODEL)
  {
    if (boundary)
    {
      // Visit each node on the boundary mesh.
      CurveMesh::Node::iterator nodeItr;
      CurveMesh::Node::iterator nodeItrEnd;
  	
      hBoundaryMesh->begin(nodeItr);
      hBoundaryMesh->end(nodeItrEnd);

      int *       meshNodeIndex;
      double *    weight;
      int meshNodeIndexsize;
      int meshNodeIndexstride;

      int numBoundaryNodes = hBoundaryToMesh->nrows();

      for (; nodeItr != nodeItrEnd; ++nodeItr)
      {		
        Point p;
        hBoundaryMesh->get_point(p, *nodeItr);

        // Find the corresponding node index in the body (TriSurf) mesh.
        hBoundaryToMesh->getRowNonzerosNoCopy(*nodeItr, meshNodeIndexsize,
                                              meshNodeIndexstride,
                                              meshNodeIndex, weight);
		
        int rhsIndex = meshNodeIndex?meshNodeIndex[0]:0;

        // Get the value for the current at this node and store this
        // value in the RHS output vector
        (*rhs)[rhsIndex] = CalcContinuumTrigCurrent(p, k, numBoundaryNodes);

        // Tag this node as an "electrode" node
        (*meshToElectrodeMap)[rhsIndex] = (*nodeItr);
      }
    } // end if (boundary)
  } // end else (if model == CONTINUUM_MODEL)
  // TRI + ELECTRODE SET + GAP MODEL
  // -------------------------------
  else if (electrodeModel == GAP_MODEL )
  {
    // Originally, we didn't execute if an electrode mapping matrix was
    // not supplied because this is the only way we know where the
    // electrodes are on the input mesh.  Supplying a point cloud
    // field of electrode positions could still be an option, but it
    // is not supported now.  The equivalent effect can be obtained
    // using the ElectrodeManager module.  The hMapping input is
    // ignored by this part of ApplyFEMElectrodeSource.

    // The code below places electrodes on the boundary of the input field.

    // Traverse the boundary (curve) field and determine its length
    if (!boundary)
    {
      error("Cannot proceed without a field boundary");
      return;
    }

    // Iterate over edges in the boundary and build a look-up-table
    // that maps each node index to its neighbor node indices.
    CurveMesh::Node::size_type nsize;
    hBoundaryMesh->size(nsize);
    int numBoundaryNodes = (int) nsize;

    Array1<Array1<CurveMesh::Node::index_type> > neighborNodes;
    neighborNodes.resize(numBoundaryNodes);

    Array1<Array1<CurveMesh::Edge::index_type> > neighborEdges;
    neighborEdges.resize(numBoundaryNodes);

    Array1<double> edgeLength;
    edgeLength.resize(numBoundaryNodes);

    CurveMesh::Node::array_type childNodes;

    CurveMesh::Edge::iterator edgeItr;
    CurveMesh::Edge::iterator edgeItrEnd;

    hBoundaryMesh->begin(edgeItr);
    hBoundaryMesh->end(edgeItrEnd);

    double boundaryLength = 0.0;

    for (; edgeItr != edgeItrEnd; ++edgeItr)
    {		
      hBoundaryMesh->get_nodes(childNodes, *edgeItr);
      unsigned int nodeIndex0 = (unsigned int) childNodes[0];
      unsigned int nodeIndex1 = (unsigned int) childNodes[1];

      neighborNodes[nodeIndex0].add(nodeIndex1);
      neighborNodes[nodeIndex1].add(nodeIndex0);
      neighborEdges[nodeIndex0].add(*edgeItr);
      neighborEdges[nodeIndex1].add(*edgeItr);

      // Store the edge length for future reference
      edgeLength[(unsigned int) *edgeItr] = hBoundaryMesh->get_size(*edgeItr);

      // Accumulate the total boundary length
      boundaryLength += edgeLength[(unsigned int) *edgeItr];

    }

    double electrodeSeparation = boundaryLength / numElectrodes;

    // Using the map we just created (neighborNodes), traverse the
    // boundary and assign electrode nodes Create an array that maps
    // boundary node index to electrode index. Initialize this array
    // to -1's meaning each boundary node is not assigned to an
    // electrode. A boundary node may only belong to one electrode.

    Array1<int> nodeElectrodeMap;
    nodeElectrodeMap.resize(numBoundaryNodes);
    for (int i = 0; i < numBoundaryNodes; i++)
    {
      nodeElectrodeMap[i] = -1;
    }

    Array1<Array1<bool> > nodeFlags;
    nodeFlags.resize(numBoundaryNodes);
    for (int i = 0; i < numBoundaryNodes; i++)
    {
      nodeFlags[i].resize(2);
      for (int j = 0; j < 2; j++)
      {
        nodeFlags[i][j] = false;
      }
    }

    Array1<Array1<double> > adjacentEdgeLengths;
    adjacentEdgeLengths.resize(numBoundaryNodes);
    for (int i = 0; i < numBoundaryNodes; i++)
    {
      adjacentEdgeLengths[i].resize(2);
      for (int j = 0; j < 2; j++)
      {
        adjacentEdgeLengths[i][j] = 0.0;
      }
    }

    // Let the node in the boundary mesh given by startNodeIndex (in
    // the electrodeParams input) be the first node in the first
    // electrode.
    int prevNode = -1;
    int currNode = startNodeIndex;
    int nextNode = neighborNodes[currNode][1];
    // Selecting element [0] or [1] influences the direction in which
    // we traverse the boundary (this should be investigated; [1]
    // seems to work well relative to the analytic solution.

    double cumulativeElectrodeLength = 0.0;
    double cumulativeElectrodeSeparation = 0.0;

    bool done = false;

    // Maximum error we can accept = 1/2 avg. edge length.
    double maxError = boundaryLength/numBoundaryNodes/2;
    // Abs difference between a desired length and a current cumulative length.
    double currError = 0.0;

    // Index of the boundary edge currently being considered.
    int currEdgeIndex = 0;

    // Flag to indicate this is the first node in an electrode
    bool firstNode = true;

    for (int i = 0; i < numElectrodes; i++)
    {
      while (!done)
      {
        // Label the current node with the current electrode ID
        if (nodeElectrodeMap[currNode] == -1)
        {
          nodeElectrodeMap[currNode] = i;
        }

        if (firstNode)
        {
          nodeFlags[currNode][0] = true;
          firstNode = false;
        }

        // Traverse the boundary until distance closest to the desired
        // electrode length is achieved.

        // First, determine if this is the degenerate 1-node electrode case
        // ----------------------------------------------------------------
        if (electrodeLen <= maxError)
        {
          nodeFlags[currNode][1] = true;  // the current node is the last node
          done = true;
          cumulativeElectrodeLength = 0.0;
        }

        // Find the index of the edge between currNode and nextNode
        // --------------------------------------------------------
        int candidateEdgeIndex0 = neighborEdges[currNode][0];
        int candidateEdgeIndex1 = neighborEdges[currNode][1];

        if ((int) neighborEdges[nextNode][0] == candidateEdgeIndex0 )
        {
          currEdgeIndex = candidateEdgeIndex0;
        }
        else if ((int) neighborEdges[nextNode][1] == candidateEdgeIndex0 )
        {
          currEdgeIndex = candidateEdgeIndex0;
        }
        else if ((int) neighborEdges[nextNode][0] == candidateEdgeIndex1 )
        {
          currEdgeIndex = candidateEdgeIndex1;
        }
        else if ((int) neighborEdges[nextNode][1] == candidateEdgeIndex1 )
        {
          currEdgeIndex = candidateEdgeIndex1;
        }

        // For first nodes that are not also last nodes, store the
        // forward direction adjacent edge length
        if (nodeFlags[currNode][1] != true)
        {
          adjacentEdgeLengths[currNode][1] = edgeLength[currEdgeIndex];
        }

        // Handle case where electrode covers more than one node
        if (!done)
        {
          // Determine if it is better to include the next node or the
          // next two nodes (If the effective electrode length will be
          // closer to the desired electrode length.)
          double testLength1 = cumulativeElectrodeLength
            + edgeLength[currEdgeIndex];
          double testError1 = Abs(electrodeLen - testLength1);

          // Advance along boundary to test addition of the next node.
          int tempPrevNode = currNode;
          int tempCurrNode = nextNode;
          int tempNextNode = -1;
          if ((int) neighborNodes[tempCurrNode][1] != tempPrevNode)
          {
            tempNextNode = (int) neighborNodes[tempCurrNode][1];
          }
          else
          {
            tempNextNode = (int) neighborNodes[tempCurrNode][0];
          }

          // Find the index of the edge between tempCurrNode and tempNextNode
          // ----------------------------------------------------------------
          int candidateEdgeIndex0 = neighborEdges[tempCurrNode][0];
          int candidateEdgeIndex1 = neighborEdges[tempCurrNode][1];

          int tempEdgeIndex = -1;

          if ((int) neighborEdges[tempNextNode][0] == candidateEdgeIndex0 )
          {
            tempEdgeIndex = candidateEdgeIndex0;
          }
          else if ((int) neighborEdges[tempNextNode][1] == candidateEdgeIndex0 )
          {
            tempEdgeIndex = candidateEdgeIndex0;
          }
          else if ((int) neighborEdges[tempNextNode][0] == candidateEdgeIndex1 )
          {
            tempEdgeIndex = candidateEdgeIndex1;
          }
          else if ((int) neighborEdges[tempNextNode][1] == candidateEdgeIndex1 )
          {
            tempEdgeIndex = candidateEdgeIndex1;
          }

          double testLength2 = testLength1 + edgeLength[tempEdgeIndex];
          double testError2 = Abs(electrodeLen - testLength2);

          if (testError1 < testError2)
          {
            // This means the nearer node achieves an electrode length
            // closer to that desired and that this node is the last
            // node in the electrode.
            nodeElectrodeMap[nextNode] = i;
            nodeFlags[nextNode][1] = true;
            cumulativeElectrodeLength = testLength1;
  	
            // We also need to store the backward direction adjacent
            // edge length for nextNode
            adjacentEdgeLengths[nextNode][0] = edgeLength[currEdgeIndex];

            done = true;
          }
          else
          {
            // This means the further node achieves an electrode
            // length closer to that desired.
            nodeElectrodeMap[nextNode] = i;
            cumulativeElectrodeLength = testLength1;

            // For middle nodes, we need to store both the backward
            // and forward adjacent edge lengths for nextNode.
            adjacentEdgeLengths[nextNode][0] = edgeLength[currEdgeIndex];
            adjacentEdgeLengths[nextNode][1] = edgeLength[tempEdgeIndex];
          }

          // Advance node pointers whether the electrode stops or continues.
          prevNode = tempPrevNode;
          currNode = tempCurrNode;
          nextNode = tempNextNode;
        } // end if (!done)
      }  // end while (!done)

      // At this point, we've finished with the current electrode.
      // Now we need to find the first node in the next electrode -
      // this will be based on the value of
      // cumulativeElectrodeSeparation which we can initialize here to
      // the value of cumulativeElectrodeLength.
      cumulativeElectrodeSeparation = cumulativeElectrodeLength;

      bool startNewElectrode = false;

      while (!startNewElectrode)
      {
        cumulativeElectrodeSeparation += edgeLength[currEdgeIndex];

        currError = Abs(electrodeSeparation - cumulativeElectrodeSeparation);

        if (currError <= maxError)
        {
          // We're within 1/2 an edge segment of the ideal electrode
          // separation.
          prevNode = currNode;
          currNode = nextNode;

          // Initialize nextNode.
          if ((int) neighborNodes[currNode][1] != prevNode)
          {
            nextNode = neighborNodes[currNode][1];
          }
          else
          {
            nextNode = neighborNodes[currNode][0];
          }

          startNewElectrode = true;
        }
        else if (cumulativeElectrodeSeparation > electrodeSeparation)
        {
          // The current error is greater than we allow, and we've
          // exceeded the separation we want.  We're trying to make
          // the first node in the next electrode equal to the last
          // node in the last electrode - this is not allowed
          error("Electrodes cannot overlap.");
          return;
        }
        // Otherwise, The current error is greater than 1/2 an edge
        // segment, and the cumulativeElectrodeSeparation is still
        // less than what we want. This happens when we have more than
        // one non-electrode node between electrodes.  do nothing in
        // this case.

        if (!startNewElectrode)
        {
          prevNode = currNode;
          currNode = nextNode;
          if ((int)neighborNodes[currNode][1] != prevNode)
          {
            nextNode = neighborNodes[currNode][1];
          }
          else
          {
            nextNode = neighborNodes[currNode][0];
          }
        }
      }  // end while (!startNewElectrode)

      done = false;
      firstNode = true;
      cumulativeElectrodeLength = 0.0;
      cumulativeElectrodeSeparation = 0.0;
    }

    // Determine the currents for the RHS vector
    // -----------------------------------------
    for (int i = 0; i < numBoundaryNodes; i++)
    {
      // Note: size of the currentPattern vector must be equal to the
      // number of electrodes!!  test this above
      if (nodeElectrodeMap[i] != -1 )  // this is an electrode node
      {
        double basisInt = 0.0;
        double current = 0.0;
        // Special case: single node electrode.
        if ( (nodeFlags[i][0] == 1) && (nodeFlags[i][1] == 1) )
        {
          current = (*currentPattern)[ nodeElectrodeMap[i] ];
        }
        // This is the first node in an electrode.
        else if (nodeFlags[i][0] == 1)
        {
          basisInt = 0.5 * adjacentEdgeLengths[i][1];
          current = basisInt * (*currentPattern)[ nodeElectrodeMap[i] ];
        }
        // This is the last node in an electrode.
        else if (nodeFlags[i][1] == 1)
        {
          basisInt = 0.5 * adjacentEdgeLengths[i][0];
          current = basisInt * (*currentPattern)[ nodeElectrodeMap[i] ];
        }
        else  // this is a middle node in an electrode
        {
          basisInt = 0.5 * adjacentEdgeLengths[i][0] +
            0.5 * adjacentEdgeLengths[i][1];
          current = basisInt * (*currentPattern)[ nodeElectrodeMap[i] ];
        }

        int *meshNodeIndex;
        double *weight;
        int meshNodeIndexsize;
        int meshNodeIndexstride;

        // Find the corresponding TriSurfMesh node index
        hBoundaryToMesh->getRowNonzerosNoCopy(i, meshNodeIndexsize,
                                              meshNodeIndexstride,
                                              meshNodeIndex, weight);
	
        int rhsIndex = meshNodeIndex?meshNodeIndex[0]:0;

        (*rhs)[rhsIndex] = current;

        // Tag this node as an "electrode" node using the electrode index
        (*meshToElectrodeMap)[rhsIndex] = nodeElectrodeMap[i];
      }
    }
  } // end if GAP model

  //! Send the meshToElectrodeMap
  MatrixOPort *oportMeshToElectrodeMap =
    (MatrixOPort *)get_oport("Mesh to Electrode Map");
  oportMeshToElectrodeMap->send(MatrixHandle(meshToElectrodeMap));
}


// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// ApplyFEMElectrodeSource::CalcContinuumTrigCurrent
//
// Description:
//
// Inputs:
//
// Returns:
//
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

double
ApplyFEMElectrodeSource::CalcContinuumTrigCurrent(Point p, int k,
                                                int numBoundaryNodes)
{
  double current;

  const double theta = ComputeTheta(p);

  if ( k < (numBoundaryNodes/2) + 1 )
  {
    current = cos(k*theta);
  }
  else
  {
    current = sin((k-numBoundaryNodes/2)*theta);
  }

  return current;
}


// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// ApplyFEMElectrodeSource::ComputeTheta
//
// Description: Find the angle, theta, the input point makes with the
// positive x axis.  This is a helper method for CalcContinuumTrigCurrent.
//
// Inputs:  Point p
//
// Returns: double theta, ( 0 <= theta < 2*PI )
//
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

double
ApplyFEMElectrodeSource::ComputeTheta(Point p)
{
  double theta = 0.0;

  if ((p.x() <= 0) && (p.y() >= 0))
  {
    theta = Atan(p.y()/(p.x() + 0.0000000001)) + PI;
  }

  if ((p.x() <= 0) && (p.y() <= 0))
  {
    theta = Atan(p.y()/(p.x() + 0.0000000001)) + PI;
  }

  if ((p.x() >= 0) && (p.y() <= 0))
  {
    theta = Atan(p.y()/(p.x() + 0.0000000001)) + 2*PI;
  }

  if ((p.x() >= 0) && (p.y() >= 0))
  {
    theta = Atan(p.y()/(p.x() + 0.0000000001));
  }

  return theta;
}


} // End namespace BioPSE
