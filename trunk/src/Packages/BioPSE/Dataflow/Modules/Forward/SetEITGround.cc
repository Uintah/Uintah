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
 *  SetEITGround.cc:
 *
 *  Written by:
 *   Lorena Kreda
 *   Updated January 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/PointCloudMesh.h>

namespace BioPSE {

using namespace SCIRun;

class SetEITGround : public Module {
    //! Private data

    //! Input ports
    MatrixIPort* iportInPot_;
    MatrixIPort* iportMeshToElectrodeMap_;

    //! Output ports
    MatrixOPort* oportGndPot_;
    MatrixOPort* oportElectrodePot_;

public:
    GuiString methodTCL_; //method of applying a ground, ie. "zero mean at electrodes"

    //! Constructor/Destructor
    SetEITGround(GuiContext *context);
    virtual ~SetEITGround();

    //! Public methods
    virtual void execute();
};

DECLARE_MAKER(SetEITGround)

SetEITGround::SetEITGround(GuiContext *context)
             : Module("SetEITGround", context, Source, "Forward", "BioPSE")
             , methodTCL_(context->subVar("methodTCL"))
{}

SetEITGround::~SetEITGround()
{}

void SetEITGround::execute()
{
    iportInPot_ = (MatrixIPort *)get_iport("InputPotentialVector");
    iportMeshToElectrodeMap_ = (MatrixIPort *)get_iport("Mesh to Electrode Map");

    oportGndPot_ = (MatrixOPort *)get_oport("OutputPotentialVector");
    oportElectrodePot_ = (MatrixOPort *)get_oport("Electrode Potentials");
  
    // Validate inputs.
    MatrixHandle hInPot;
    if (!iportInPot_->get(hInPot) || !hInPot.get_rep())
    {
        error("Can't get handle to input potential vector.");
        return;
    }

    MatrixHandle hMeshToElectrodeMap;
    if (!iportMeshToElectrodeMap_->get(hMeshToElectrodeMap) || !hMeshToElectrodeMap.get_rep())
    {
        error("Can't get handle to input mesh to electrode map vector.");
        return;
    }

    // Assign local pointer to input potential vector
    ColumnMatrix *InPot = dynamic_cast<ColumnMatrix*>(hInPot.get_rep());
  
    if (!InPot)
    {
        error("can't cast input and/or output as a column!");
        return;
    }

    // determine the dimension (number of nodes) of the input potential vector
    int numNodes = InPot->nrows();

    //cout << "Number of mesh nodes: " << numNodes << endl;

    // Assign local pointer to input mesh-to-electrde-map vector
    ColumnMatrix *meshToElectrodeMap = dynamic_cast<ColumnMatrix*>(hMeshToElectrodeMap.get_rep());
  

    // allocate memory for the potentials we are going to compute and send to the output
    ColumnMatrix* gndPot = scinew ColumnMatrix(numNodes);
    if (!gndPot)
    {
        error("can't allocate output column matrix!");
        return;
    }
  
    double sum_potential  = 0.0;
    double mean_potential = 0.0;
    int numElectrodeNodes = 0;

    for (int i=0; i<numNodes; i++)
    {
        if ( (*meshToElectrodeMap)[i] != -1)
        {
            sum_potential += (*InPot)[i];
            numElectrodeNodes++;
	}
    }

    mean_potential = sum_potential/numElectrodeNodes;

    cout << "numElectrodeNodes: " << numElectrodeNodes << " Mean boundary node potential: " << mean_potential << endl;

    // subtract the boundary node mean from the input potential at each node and 
    // store this in the output potential vector.

    for (int i=0; i<numNodes; i++)
    {
      (*gndPot)[i] = (*InPot)[i] - mean_potential;  
    }

    // Determine the number of electrodes. The meshToElectrodeMap is a column matrix where each element contains the
    // electrode index for the particular mesh node index. If a mesh node j is not "covered by" an electrode, 
    // meshToElectrodeMap[j] = -1. If mesh node j does belong to an electrode, then meshToElectrodeMap[j] contains the
    // index of that electrode. The maximum value in meshToElectrodeMap is the number of electrodes minus one, since 
    // zero is a valid electrode index.

    // Find the maximum value in meshToElectrodeMap.
    int numElectrodes = -1;
    for (int j = 0; j < numNodes; j++)
    {
        if ( (*meshToElectrodeMap)[j] > numElectrodes)
        {
  	    numElectrodes = (int) (*meshToElectrodeMap)[j];
	}
    }
    numElectrodes++;  // add one because electrode index is zero based

    cout << "SetEITGround: numElectrodes found: " << numElectrodes << endl;
    
    // allocate memory for the electrode potentials we are going to extract and send to the output
    ColumnMatrix* electrodePot = scinew ColumnMatrix(numElectrodes);
    if (!electrodePot)
    {
        error("can't allocate output for electrode potentials column matrix!");
        return;
    }

    // Initialize the electrodePot vector to all zeros
    // -----------------------------------------------
    for (int i = 0; i < numElectrodes; i++)
    {
        (*electrodePot)[i] = 0.0;
    }
    
    // Search through the meshToElectrodeMap and pull out electrode voltages from
    // the grounded potential vector. If an electrode has more than one node, take the 
    // average value.
    // -------------------------------------------------------------------------------
    // declare array to store the count of the number of nodes on each electrode
    Array1<int>  nodeCount;
    nodeCount.resize(numElectrodes);
    for (int i = 0; i < numElectrodes; i++)
    {
        nodeCount[i] = 0;
    }

    for (int i = 0; i < numNodes; i++)
    {
        if ( (*meshToElectrodeMap)[i] != -1)
        {
	    // count the number of nodes on each electrode
 	    // -------------------------------------------
	    nodeCount[(int) (*meshToElectrodeMap)[i] ]++;
            // accumulate the sum of electrode potentials
            // ------------------------------------------	    
            (*electrodePot)[ (int) (*meshToElectrodeMap)[i] ] +=  (*gndPot)[i];
	}
    }

    for (int i = 0; i < numElectrodes; i++)
    {
        (*electrodePot)[i] /= nodeCount[i];
    }

    // Shift the values circularly by one - this is just a test
    double saveFirstValue = (*electrodePot)[0];
    for (int i = 0; i < numElectrodes-1; i++)
    {
      (*electrodePot)[i] = (*electrodePot)[i+1];
    }
    (*electrodePot)[numElectrodes-1] = saveFirstValue;
    
    // print out values - for development debugging; leave commented out for now
    //cout << "input and output vectors" << endl;
    //for (int i=0; i<numNodes; i++)
    //{
    //    cout << (*InPot)[i] << " " << (*gndPot)[i] << endl;
    //}

    for (int i=0; i<numElectrodes; i++)
    {
      cout << i << " " << (*electrodePot)[i] << endl;
    }

    //! Sending results
    oportGndPot_->send(MatrixHandle(gndPot)); 
    oportElectrodePot_->send(MatrixHandle(electrodePot)); 

}

} // End namespace BioPSE


