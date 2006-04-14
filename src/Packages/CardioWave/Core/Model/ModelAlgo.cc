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

#include <Packages/CardioWave/Core/Model/ModelAlgo.h>
#include <Packages/ModelCreation/Core/Fields/FieldsAlgo.h>
#include <Packages/ModelCreation/Core/Numeric/NumericAlgo.h>
#include <Packages/ModelCreation/Core/DataIO/DataIOAlgo.h>
#include <Packages/ModelCreation/Core/Converter/ConverterAlgo.h>
#include <Packages/CardioWave/Core/Model/BuildMembraneTable.h>
#include <Packages/CardioWave/Core/Model/BuildStimulusTable.h>
#include <Core/Datatypes/MatrixOperations.h>

#include <fstream>

namespace CardioWave {

using namespace SCIRun;
using namespace ModelCreation;


ModelAlgo::ModelAlgo(ProgressReporter* pr) :
  AlgoLibrary(pr)
{
}

bool ModelAlgo::DMDBuildMembraneTable(FieldHandle ElementType, FieldHandle MembraneModel, MatrixHandle CompToGeom, MatrixHandle NodeLink, MatrixHandle ElemLink, MembraneTable& MembraneTable)
{
  BuildMembraneTableAlgo algo;
  MatrixHandle Dummy; // need to remove this one
  return(algo.BuildMembraneTable(pr_,ElementType,MembraneModel,CompToGeom,NodeLink, ElemLink,MembraneTable,Dummy));
}

bool ModelAlgo::DMDBuildStimulusTable(FieldHandle ElementType, FieldHandle StimulusModel, MatrixHandle CompToGeom,  double stimulusdomain, StimulusTable& StimulusTable)
{
  BuildStimulusTableAlgo algo;
  return(algo.BuildStimulusTable(pr_,ElementType,StimulusModel,CompToGeom,stimulusdomain,true,StimulusTable));
}

bool ModelAlgo::DMDBuildStimulusTableByElement(FieldHandle ElementType, FieldHandle StimulusModel, MatrixHandle CompToGeom,  double stimulusdomain, StimulusTable& StimulusTable)
{
  BuildStimulusTableAlgo algo;
  return(algo.BuildStimulusTable(pr_,ElementType,StimulusModel,CompToGeom,stimulusdomain,false,StimulusTable));
}

bool ModelAlgo::DMDBuildReferenceTableByElement(FieldHandle ElementType, FieldHandle ReferenceModel, MatrixHandle CompToGeom,  double referencedomain, ReferenceTable& ReferenceTable)
{
  BuildStimulusTableAlgo algo;
  return(algo.BuildStimulusTable(pr_,ElementType,ReferenceModel,CompToGeom,referencedomain,false,ReferenceTable));
}

bool ModelAlgo::DMDBuildReferenceTable(FieldHandle ElementType, FieldHandle ReferenceModel, MatrixHandle CompToGeom,  double referencedomain, ReferenceTable& ReferenceTable)
{
  BuildStimulusTableAlgo algo;
  return(algo.BuildStimulusTable(pr_,ElementType,ReferenceModel,CompToGeom,referencedomain,true,ReferenceTable));
}

bool ModelAlgo::DMDReferenceTableToMatrix(ReferenceTable ReferenceTable,MatrixHandle& M)
{
  M = dynamic_cast<Matrix *>(scinew DenseMatrix(ReferenceTable.size(),2));
  if (M.get_rep() == 0) return(false);
  double *dataptr = M->get_data_pointer();
  
  int p =0;
  for (size_t k=0; k <ReferenceTable.size(); k++)
  {
    dataptr[p++] = static_cast<double>(ReferenceTable[k].node);
    dataptr[p++] = static_cast<double>(ReferenceTable[k].weight);
  }
  
  return (true);
}


bool ModelAlgo::DMDStimulusTableToMatrix(StimulusTable StimulusTable,MatrixHandle& M)
{
  M = dynamic_cast<Matrix *>(scinew DenseMatrix(StimulusTable.size(),2));
  if (M.get_rep() == 0) return(false);
  double *dataptr = M->get_data_pointer();
  
  int p =0;
  for (size_t k=0; k <StimulusTable.size(); k++)
  {
    dataptr[p++] = static_cast<double>(StimulusTable[k].node);
    dataptr[p++] = static_cast<double>(StimulusTable[k].weight);
  }
  
  return (true);
}


bool ModelAlgo::DMDMembraneTableToMatrix(MembraneTable MembraneTable,MatrixHandle& M)
{
  M = dynamic_cast<Matrix *>(scinew DenseMatrix(MembraneTable.size(),4));
  if (M.get_rep() == 0) return(false);
  double *dataptr = M->get_data_pointer();
  
  int p =0;
  for (int k=0; k <MembraneTable.size(); k++)
  {
    dataptr[p++] = static_cast<double>(MembraneTable[k].node0);
    dataptr[p++] = static_cast<double>(MembraneTable[k].node1);
    dataptr[p++] = static_cast<double>(MembraneTable[k].node2);
    dataptr[p++] = static_cast<double>(MembraneTable[k].surface);
  }
  
  return (true);
}

bool ModelAlgo::DMDBuildSimulation(BundleHandle SimulationBundle, StringHandle FileName, BundleHandle& VisualizationBundle, StringHandle& Script)
{
  // Define all the dynamic algorithms.
  // Forward the ProgressReporter so everything can forward an error

  ModelCreation::FieldsAlgo  fieldsalgo(pr_);
  ModelCreation::NumericAlgo numericalgo(pr_);
  ModelCreation::ConverterAlgo converteralgo(pr_);
  ModelCreation::DataIOAlgo  dataioalgo(pr_);

  // Step 0: Some sanity checks
  
  if (SimulationBundle.get_rep() == 0)
  {
    error("DMDBuildSimulation: SimulationBundle is empty");
    return (false);
  }

  if (FileName.get_rep() == 0)
  {
    error("DMDBuildSimulation: FileName is empty");
    return (false);  
  }
  
  // Step 1: Define the filenames
  
  std::string filename_simbundle;   // Store the full configuration to desk so we can retrieve it for debugging or database purposes
  std::string filename_visbundle;   // Visualization Bundle, as we need to renumber the system we need a back projection all information is contained in this bundle
  std::string filename_sysmatrix;   // The system matrix
  std::string filename_nodetype;    // A vector with the nodetype for each node in the system
  std::string filename_surface;     // Surface factors for the cell membrane
  std::string filename_in;          // Parameter file
  std::string filename_script;      // The script to build simulation
  std::string filename_membrane;    // Membrane description (geometry)
  std::string filename_stimulus;    // Stimulus nodes in a file
  std::string filename_reference;   // Reference nodes in a file   
  
  // Try to derive the filename
  // Remove the .sim.bdl extension of the simulation setup bundle file
  // First remove .bdl and then .sim 
  filename_simbundle  = FileName->get();
  std::string filenamebase = filename_simbundle;
  if ((filename_simbundle.size() > 4)&&(filename_simbundle.substr(filename_simbundle.size()-4)==std::string(".bdl")))
  {
    filenamebase = filename_simbundle.substr(0,filename_simbundle.size()-4);
  }
  if ((filenamebase.size() > 4)&&(filenamebase.substr(filenamebase.size()-4)==std::string(".sim")))
  {
    filenamebase = filenamebase.substr(0,filenamebase.size()-4);
  }
  
  // We now have the filenamebase and now define each of the new filenames
 
  filename_visbundle = filenamebase + ".vis.bdl";
  filename_sysmatrix = filenamebase + ".fem.spr";
  filename_nodetype  = filenamebase + ".nt.bvec";
  filename_surface   = filenamebase + ".area.vec";
  filename_in        = filenamebase + ".in";
  filename_script    = filenamebase + ".script.pl";
  filename_membrane  = filenamebase + ".mem.txt";
  filename_stimulus  = filenamebase + ".stim.txt";
  filename_reference = filenamebase + ".ref.txt";
   
  // Step 2: Save the model to file, we need to have an archived version
  // This will allow us to go back to the original data
  
  if (!(dataioalgo.WriteBundle(std::string(filename_simbundle),SimulationBundle)))
  {
    error("DMDBuildSimulator: Could not save simulation bundle to disk");
    return (false);    
  }
   
  // Step 3: Build the Finite Element Model 

  BundleHandle Domain = SimulationBundle->getBundle("Domain");
  if (Domain.get_rep() == 0)
  {
    error("DMDBuildSimulator: Domain is not defined");
    return (false);        
  }
  
  FieldHandle ElementType = Domain->getField("ElementType");
  FieldHandle Conductivity = Domain->getField("Conductivity");
  MatrixHandle ConductivityTable = Domain->getMatrix("ConductivityTable");
  MatrixHandle ElemLink = Domain->getMatrix("ElemLink");
  MatrixHandle NodeLink = Domain->getMatrix("NodeLink");

  if (ElemLink.get_rep() == 0)
  {
    if (ElementType->is_property("ElemLink")) ElementType->get_property("ElemLink",ElemLink);
    if (ElementType->is_property("NodeLink")) ElementType->get_property("NodeLink",NodeLink);
  }
  
  if (ElementType.get_rep() == 0)
  {
    error("DMDBuildDomain: No field called 'ElementType' was defined");
    return (false);
  }

  if (Conductivity.get_rep() == 0)
  {
    error("DMDBuildDomain: No field called 'Conductivity' was defined");
    return (false);
  }

  // Get all the information from the sub bundles
  // in the model

  std::vector<FieldHandle> Membranes; // Collect all the Membrane Geometries in one vector
  std::vector<double>      nodetypes; // Get the corresponding nodetype as well

  std::vector<FieldHandle> References;      // Collect all the Geometries of the References in here
  std::vector<double>      referencevalues;
  std::vector<bool>        usereferencevalues;

  std::vector<FieldHandle> Stimulus;
  std::vector<double>      stimulusdomain;
  std::vector<double>      stimuluscurrent;
  std::vector<double>      stimulusstart;
  std::vector<double>      stimulusend;
  std::vector<bool>        stimulusiscurrentdensity;

  size_t num_membranes = 0;
  size_t num_stimulus = 0;
  size_t num_references = 0;


  // Loop through all sub bundles in the mother bundle to get all the individual
  // components of the model. Next to the volume each piece of membrane has its
  // own geometry. Similarly every electrode (stimulus and reference has its own geometry)
  for (size_t p=0; p < SimulationBundle->numBundles(); p++)
  {
    // Do we have a membrane definition
    std::string bundlename = SimulationBundle->getFieldName(p);
    if (bundlename.substr(0,9) == "Membrane_")
    {
      BundleHandle MembraneBundle = SimulationBundle->getBundle(bundlename);
      FieldHandle Geometry = MembraneBundle->getField("Geometry");
      MatrixHandle NodeType = MembraneBundle->getMatrix("NodeType");
      if (Geometry.get_rep()||NodeType.get_rep())
      {
        double nodetype;
        converteralgo.MatrixToDouble(NodeType,nodetype);
        // Move the information in the vectors
        // We use vectors here as we are only shifting pointers
        // It is just book keeping
        Membranes.push_back(Geometry);
        nodetypes.push_back(nodetype);
      }
      else
      {
        warning("DMDBuildDomain: One of the membrane fields misses a Geometry or a NodeType");
      }
    }

    if (bundlename.substr(0,10) == "Reference_")
    {
      BundleHandle ReferenceBundle =  SimulationBundle->getBundle(bundlename);
      FieldHandle Geometry = ReferenceBundle->getField("Geometry");
      MatrixHandle RefValue = ReferenceBundle->getMatrix("RefValue");

      if (Geometry.get_rep())
      {
        References.push_back(Geometry);
        // If no reference value is given we assume one
        // If one is given we just need to convert it.
        if (RefValue.get_rep())
        {
          double refvalue = 0.0;
          converteralgo.MatrixToDouble(RefValue,refvalue);
          referencevalues.push_back(refvalue);
          usereferencevalues.push_back(true);
        }
        else
        {
          referencevalues.push_back(0.0);
          usereferencevalues.push_back(false);
        }
      }
      else
      {
        warning("DMDBuildDomain: One of the reference fields misses a Geometry");
      }

    }

    if (bundlename.substr(0,9) == "Stimulus_")
    {
      BundleHandle StimulusBundle = SimulationBundle->getBundle(bundlename);
      FieldHandle  Geometry = StimulusBundle->getField("Geometry");
      MatrixHandle Domain = StimulusBundle->getMatrix("Domain");
      MatrixHandle Current = StimulusBundle->getMatrix("Current");
      MatrixHandle Start = StimulusBundle->getMatrix("Start");
      MatrixHandle End = StimulusBundle->getMatrix("End");
      MatrixHandle CurrentDensity = StimulusBundle->getMatrix("CurrentDensity");

      // Check whether we have all the information needed
      if ((Geometry.get_rep())&&(Domain.get_rep())&&(Start.get_rep())&&(End.get_rep()))
      {
        double domain;
        double start;
        double end;
        double current;

        Stimulus.push_back(Geometry);
        converteralgo.MatrixToDouble(Domain,domain);
        converteralgo.MatrixToDouble(Start,start);
        converteralgo.MatrixToDouble(End,end);

        stimulusdomain.push_back(domain);
        stimulusstart.push_back(end);     
        stimulusend.push_back(start);

        if (Current.get_rep())
        {
          converteralgo.MatrixToDouble(Current,current);
          stimuluscurrent.push_back(current);
          stimulusiscurrentdensity.push_back(false);
        }
        else if (CurrentDensity.get_rep())
        {
          converteralgo.MatrixToDouble(CurrentDensity,current);
          stimuluscurrent.push_back(current);
          stimulusiscurrentdensity.push_back(true);
        }
        else
        {
          stimuluscurrent.push_back(0.0);
          stimulusiscurrentdensity.push_back(false);
        }
      }
      else
      {
        warning("DMDBuildDomain: One of the stimulus fields misses a Geometry or a stimulation Domain, or a Starting time or a Ending time");
      }
    }
  }

  MatrixHandle GeomToComp;
  MatrixHandle CompToGeom;
  
  // Optional linkage of boundaries
  if (NodeLink.get_rep())
  {
    if(!(fieldsalgo.LinkToCompGridByDomain(ElementType,NodeLink,GeomToComp,CompToGeom)))
    {
      error("DMDBuildDomain: Could not build computational grid to geometrical mesh linkage matrices");
      return (false);  
    }
  }

  // We should have ordered all the geometric information now
  
  num_membranes = Membranes.size();
  num_stimulus  = Stimulus.size();
  num_references = References.size();
  
  // To be able to map data back onto the geometry we will be needed mapping matrices
  // Hence define a lot of properties. Most of it we need for the model or the
  // visualization bundle
  std::vector<MembraneTable>     membranetable(num_membranes);
  std::vector<FieldHandle>       membranefield(num_membranes);
  std::vector<MatrixHandle>      membranemapping(num_membranes);
  std::vector<int>               membranenumnodes(num_membranes);
  std::vector<int>               membranenumelems(num_membranes);
  
  // A synapse node contains the parameters of the membrane in CW they are separate nodes
  // They are not contained in the domain, they are just computational.
  int num_synnodes = 0;

  // Get all the information
  for (size_t p=0; p<membranetable.size();p++)
  {
    // Get the size
    // This function will dynamically compile if needed
    if (!(fieldsalgo.GetFieldInfo(Membranes[p],membranenumnodes[p],membranenumelems[p])))
    {
      error("DMDBuildDomain: Could get information from membrane field");
      return(false);    
    }
    
    // In the visualization we want to have a linear version, hence we need to
    // convert it to a linear one. This function will build a new one if needed
    if (!(fieldsalgo.ClearAndChangeFieldBasis(Membranes[p],Membranes[p],"Linear")))
    {
      error("DMDBuildDomain: Could not build a linear field for the membrane");
      return(false);        
    }
    
    // This is the expensive function, it figures out how the membrane pannels
    // are linked into the volumetric model. As we need not save this information
    // we are reconstructing it here. The reason for not keeping it is that this
    // way the user can clip and delete panels without having to remember each
    // change, as they all change the node numbering.
    if (!(DMDBuildMembraneTable(ElementType,Membranes[p],CompToGeom,NodeLink,ElemLink,membranetable[p])))
    {
      error("DMDBuildDomain: Could build membrane model");
      return(false);
    }
    
    // Figure out how many synapse nodes are needed.
    num_synnodes += membranetable[p].size();
  }
  
  // Build the Finite Element Model here
  // We enter optional fields here, if they are empty handles the routine will skip
  // them.
  MatrixHandle fematrix;
  if(!(numericalgo.BuildFEMatrix(Conductivity,fematrix,-1,ConductivityTable,GeomToComp,CompToGeom)))
  {
    error("DMDBuildDomain: Could not build FE Matrix");
    return(false);
  }
  
  // Figure out how many nodes we actually have
  int num_volumenodes = fematrix->nrows();
  int num_totalnodes = num_volumenodes + num_synnodes;
  
  // Resize the matrix so we can use it to store the synapse nodes as well
  if(!(numericalgo.ResizeMatrix(fematrix,fematrix,num_totalnodes,num_totalnodes)))
  {
    error("DMDBuildDomain: Could not resize FE matrix");
    return (false);
  }
  
  // Build a vector for the surface areas
  MatrixHandle volvec = dynamic_cast<Matrix*>(scinew DenseMatrix(num_totalnodes,1));
  if (volvec.get_rep() == 0)
  {
    error("DMDBuildDomain: Could not allocate Volume Vector Matrix");
    return (false);
  }
  double* volvecptr = volvec->get_data_pointer();
  
  // Build a vector for the node types
  MatrixHandle nodetypevec = dynamic_cast<Matrix*>(scinew DenseMatrix(num_totalnodes,1));
  if (nodetypevec.get_rep() == 0)
  {
    error("DMDBuildDomain: Could not allocate Volume Vector Matrix");
    return (false);
  }
  double* nodetypevecptr = volvec->get_data_pointer();
  
  // Build the Membrane connections

  int synnum = num_volumenodes;
  SparseElementVector sev(num_synnodes*3);
  int k = 0;
  for (int p=0; p<membranetable.size();p++)
  {
    SparseElementVector sevmapping(2*membranetable[p].size());
    int j = 0;
    for (int q=0; q< membranetable[p].size(); q++)
    { 
      sev[k].row = synnum;
      sev[k].col = synnum;
      sev[k].val = 1.0;
      volvecptr[synnum] = membranetable[p][q].surface;
      nodetypevecptr[synnum] = nodetypes[p];
      k++; synnum++; 
      sev[k].row = membranetable[p][q].node1;
      sev[k].col = membranetable[p][q].node2;
      sev[k].val = 1.0;
      k++;    
      sev[k].row = membranetable[p][q].node2;
      sev[k].col = membranetable[p][q].node1;
      sev[k].val = 1.0;
      k++; 
      sevmapping[j].row = membranetable[p][q].node0;
      sevmapping[j].col = membranetable[p][q].node1;
      sevmapping[j].val = -1.0;      
      j++;
      sevmapping[j].row = membranetable[p][q].node0;
      sevmapping[j].col = membranetable[p][q].node2;
      sevmapping[j].val = 1.0;
      j++;
    }
    
    if (!(numericalgo.CreateSparseMatrix(sevmapping, membranemapping[p],membranenumnodes[p],num_synnodes)))
    {
      error("DMDBuildDomain: Could not build sparse mapping matrix");
      return (false);    
    }
  
    sevmapping.clear();
  }
  

  
  MatrixHandle synmatrix;
  if(!(numericalgo.CreateSparseMatrix(sev, synmatrix, num_totalnodes, num_totalnodes)))
  {
    error("DMDBuildDomain: Could not build synapse sparse matrix");
    return (false);
  }  

  // Free some memory
  sev.clear();
  
  // Somehow CardioWave uses the negative matrix for its
  MatrixHandle sysmatrix = synmatrix - fematrix;
  
  // free some memory
  synmatrix = 0;
  fematrix = 0;
  
  if (sysmatrix.get_rep() == 0)
  {
    error("DMDBuildDomain: Could not build system sparse matrix");
    return (false);  
  }

  MatrixHandle mapping;
  if(!(numericalgo.ReverseCuthillmcKee(sysmatrix,sysmatrix,mapping)))
  {
    error("DMDBuildDomain: Matrix reordering failed");
    return (false);    
  }

  // Reorder domain properties
  nodetypevec = mapping*nodetypevec;
  volvec = mapping*volvec;

  if (!(dataioalgo.WriteMatrix(filename_sysmatrix,sysmatrix,"CardioWave Sparse Matrix")))
  {
    error("DMDBuildDomain: Could not write system matrix");  
    return (false);
  }
  if (!(dataioalgo.WriteMatrix(filename_surface,volvec,"CardioWave FP Vector")))
  {
    error("DMDBuildDomain: Could not write volume vector");  
    return (false);
  }

  if (!(dataioalgo.WriteMatrix(filename_nodetype,nodetypevec,"CardioWave Byte Vector")))
  {
    error("DMDBuildDomain: Could not write nodetype vector");  
    return (false);
  }

  // clean memory
  sysmatrix = 0;
  nodetypevec = 0;
  volvec = 0;
  
  
  MatrixHandle imapping = mapping->transpose();
  if (imapping.get_rep() == 0)
  {
    error("DMDBuildDomain: Could not build mapping matrix");
    return (false);      
  }

  SparseRowMatrix* spr = dynamic_cast<SparseRowMatrix *>(imapping.get_rep());
  if (spr == 0)
  {
    error("DMDBuildDomain: Could not build mapping matrix");
    return (false);     
  }

  int* renumber = spr->columns;
  
  // Build membrane table
  for (int p=0; p<membranetable.size();p++)
  {
    for (int q=0; q< membranetable[p].size(); q++)
    { 
      membranetable[p][q].node0 = renumber[membranetable[p][q].node0];
      membranetable[p][q].node1 = renumber[membranetable[p][q].node1];
      membranetable[p][q].node2 = renumber[membranetable[p][q].node2];
    }
  }
  
  // Build membrane table file
  
  
  try
  {
    std::ofstream memfile;
    memfile.open(filename_membrane.c_str());
    for (int p=0; p<membranetable.size();p++)
    {
      for (int q=0; q< membranetable[p].size(); q++)
      { 
      memfile << membranetable[p][q].node0 << " " << membranetable[p][q].node1 << " " << membranetable[p][q].node2 << "\n";
      }
    }
  }
  catch (...)
  {
    error("DMDBuildDomain: Could write membrane connection file");
    return (false);  
  }

  if (!(numericalgo.ResizeMatrix(imapping,imapping,num_totalnodes,num_volumenodes)))
  {
    error("DMDBuildDomain: Could not resize mapping matrix");
    return (false);        
  }  

  if (!(fieldsalgo.ClearAndChangeFieldBasis(ElementType,ElementType,"Linear")))
  {
    error("DMDBuildDomain: Could not clear field");
    return (false);            
  }

  BundleHandle VisBundle = scinew Bundle();
  BundleHandle VolumeField = scinew Bundle();
  
  VolumeField->setField("Field",ElementType);
  VolumeField->setMatrix("Mapping",imapping);
  VisBundle->setBundle("Tissue",VolumeField);
  
  for (size_t p=0; p <num_membranes; p++)
  {
    std::ostringstream oss;
    oss << "Membrane_" << p+1;
    
    BundleHandle MembraneBundle = scinew Bundle;
    MembraneBundle->setField("Field",membranefield[p]);
    MembraneBundle->setMatrix("Mapping",membranemapping[p]);
    VisBundle->setBundle(oss.str(),MembraneBundle);
  }

  if (!(dataioalgo.WriteBundle(filename_visbundle,VisBundle)))
  {
    error("DMDBuildDomain: Could not write visualization information");
    return (false);     
  }
   
  return (true);
}
  

} // end namespace SCIRun
