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
#include <Packages/ModelCreation/Core/Field/FieldsAlgo.h>
#include <Packages/ModelCreation/Core/Numeric/NumericAlgo.h>
#include <Packages/ModelCreation/Core/DataIO/DataIOAlgo.h>
#include <Packages/ModelCreation/Core/Converter/ConverterAlgo.h>
#include <Packages/CardioWave/Core/Model/BuildMembraneTable.h>

namespace CardioWave {

using namespace SCIRun;
using namespace ModelCreation;


ModelAlgo::ModelAlgo(ProgressReporter* pr) :
  AlgoLibrary(pr)
{
}

bool ModelAlgo::DMDBuildMembraneTable(FieldHandle elementtype, FieldHandle membranemodel, MembraneTableList& membranetable)
{
  BuildMembraneTableAlgo algo;
  return(algo.BuildMembraneTable(pr_,elementtype,membranemodel,membranetable));
}

bool ModelAlgo::DMDBuildSimulator(BundleHandle Model, std::string filename)
{
  ModelCreation::FieldsAlgo  fieldsalgo(pr_);
  ModelCreation::NumericAlgo numericalgo(pr_);
  ModelCreation::ConverterAlgo converteralgo(pr_);
  ModelCreation::DataIOAlgo  dataio(pr_);
  // Build Filenames output
  
  if (filename.size() == 0) 
  {
    error("DMDBuildDomain: Could not resolve filename");
    return (false);
  }

  std::string filenamebase = filename;
  if ((filename.size() > 4)&&(filename.substr(filename.size()-4)==std::string(".bdl")))
  {
    filenamebase = filename.substr(0,filename.size()-4);
  }

  std::string filename_bundle = filenamebase + ".bdl";
  std::string filename_parameters = filenamebase + ".param.in";
  std::string filename_buildsimulator = filenamebase + ".sim.pl";

  // Step one save the model to file
  
  if (!(dataio.WriteBundle(filename_bundle,Model)))
  {
    error("DMDBuildSimulator: Could not save model bundle to disk");
    return (false);    
  }
  
  // Setup Parameter output file
  
  std::string param;
  std::string buildsimulator;
  std::vector<std::string> ofiles;
  
  // Disassemble Model Bundle
  
  FieldHandle elementtype;
  FieldHandle conductivity;
  
  // Deal with the domain first
  // The domain is specified by means of an element type and a conductivity field
  
  elementtype = Model->getField("elementtype");
  if (!(elementtype.get_rep()))
  {
    error("DMDBuildSimulator: Model does not have a field called 'elementtype' to specify cell type and ECS region");
    return (false);
  }

  conductivity = Model->getField("conductivity");
  if (!(conductivity.get_rep()))
  {
    error("DMDBuildSimulator: Model does not have a field called 'conductivity' to specify electrical properties");
    return (false);
  }
  
  // deal with the membrane definitions

  std::vector<FieldHandle> membranes;
  std::vector<double>      nodetypes;
  
  int m = 0;
  while(1)
  {  
    std::ostringstream membranename;
    membranename << "membrane_" << m+1 ;
    BundleHandle Membrane = Model->getBundle(membranename.c_str());
    if (Membrane.get_rep() == 0) break;
    
    FieldHandle geometry = Membrane->getField("geometry");
    MatrixHandle nodetypemat = Membrane->getMatrix("nodetype");
    StingHandle memparam = Membrane->getString("param");
    
    if ((geometry.get_rep()==0)||(nodetypemat.get_rep()==0)) break;
    m++;
    
    double nodetype = 0;
    converteralgo.MatrixToDouble(nodetypemat,nodetype);
    membranes.push_back(geometry);
    nodetypes.push_back(nodetype);
    
    param += "# " + membranename.c_str() + "\n\n" + memparam.get() +"\n\n";    
  }

  if (m==0)
  {
    error("DMDBuildSimulator: Model does not have any valid membranes");
    return (false);  
  }
  
  // Handle geometric part of the model
  
  MatrixHandle sysmatrix, volvec, nodetype, mapping;
  
  if(!(DMDBuildDomain(elementtype,conductivity,membranes,nodetypes,filename))) return (false);
  
  // Handle solver/timestepper


  // Handle simulation output
  
  std::string outputfile;
  std::string outputfile_header;


  // Write parameters file
  
  


}

bool ModelAlgo::DMDBuildDomain(FieldHandle elementtype, FieldHandle conductivity, std::vector<FieldHandle> membrane, std::vector<double> nodetypes, std::string filename);   
{
  ModelCreation::FieldsAlgo fieldalgo(pr_);
  ModelCreation::NumericAlgo numericalgo(pr_);
  ModelCreation::DataIOAlgo dataiocalgo(pr_);

  std::string filename_systemmatrix = filenamebase + ".fem.spr";
  std::string filename_volumevector = filenamebase + ".vol.vec";
  std::string filename_nodetypevector = filenamebase + ".nt.vec";
  std::string filename_visbundle = filenamebase + ".vis.bdl";
  
  size_t nummembranes = membrane.size();
  std::vector<MembraneTableList> membranetable(nummembranes);
  std::vector<FieldHandle>       membranefield(nummembranes);
  std::vector<MatrixHandle>      membranemapping(nummembranes);
  std::vector<int>               membranenumnodes(nummembranes);
  std::vector<int>               membranenumelems(nummembranes);
  
  int num_synnodes = 0;

  for (size_t p=0; p<membranetable.size();p++)
  {
    if (!(fieldalgo.GetFieldInfo(membrane[p],membranenumnodes[p],membranenumelems[p])))
    {
      error("DMDBuildDomain: Could get information from membrane field");
      return(false);    
    }
    
    if (!(fieldalgo.ClearAndChangeFieldBasis(membrane[p],membrane[p],"Linear")))
    {
      error("DMDBuildDomain: Could not build a linear field for the membrane");
      return(false);        
    }
    
    if (!(DMDBuildMembraneTable(elementtype,membrane[p],mebranetable[p])))
    {
      error("DMDBuildDomain: Could build membrane model");
      return(false);
    }
    
    num_synnodes += membranetable[p].size());
  }
  
  MatrixHandle fematrix;
  if(!(numericalgo.BuildFEMatrix(conductivity,fematrix)))
  {
    error("DMDBuildDomain: Could not build FE Matrix");
    return(false);
  }
  
  int num_volumenodes = fematrix->nrows();
  int num_totalnodes = num_volumenodes + num_synnodes;
  
  if(!(numericalgo.ResizeMatrix(fematrix,fematrix,num_totalnodes,num_totalnodes)))
  {
    error("DMDBuildDomain: Could not resize FE matrix");
    return (false);
  }
  
  volvec = dynamic_cast<Matrix*>(scinew DenseMatrix(num_totalnodes,1));
  if (volvec.get_rep() == 0)
  {
    error("DMDBuildDomain: Could not allocate Volume Vector Matrix");
    return (false);
  }
  
  double* volvecptr = volvec.get_data_pointer();
  
  nodetypevec = dynamic_cast<Matrix*>(scinew DenseMatrix(num_totalnodes,1));
  if (nodetypevec.get_rep() == 0)
  {
    error("DMDBuildDomain: Could not allocate Volume Vector Matrix");
    return (false);
  }
  double* nodetypevecptr = volvec.get_data_pointer();
  
  int synnum = num_volumenodes;
  SparseElementVector sev(num_synnodes*5);
  int k = 0;
  for (size_t p=0; p<membranetable.size();p++)
  {
    SparseElementVector sevmapping(2*membranetable[p].size());
    int j = 0;
    for (size_t q=0; q< membranetable[p].size(); q++)
    { 
      sev[k].row = synnum;
      sev[k].col = synnum;
      sev[k].val = 1.0;
      volvecptr[synnum] = membranetable[p][q].surface;
      nodetypevecptr[synnum] = nodetypes[p];
      k++; synnum++; 
      sev[k].row = membranetable[p][q].node1;
      sev[k].col = membranetable[p][q].node1;
      sev[k].val = 0.0;
      k++;
      sev[k].row = membranetable[p][q].node1;
      sev[k].col = membranetable[p][q].node2;
      sev[k].val = 0.0;
      k++;    
      sev[k].row = membranetable[p][q].node2;
      sev[k].col = membranetable[p][q].node1;
      sev[k].val = 0.0;
      k++; 
      sev[k].row = membranetable[p][q].node2;
      sev[k].col = membranetable[p][q].node2;
      sev[k].val = 0.0;
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
  }

  MatrixHandle synmatrix;
  if(!(numericalgo.CreateSparseMatrix(sev, synmatrix, num_totalnodes, num_totalnodes)))
  {
    error("DMDBuildDomain: Could not build synapse sparse matrix");
    return (false);
  }  
  
  // Free some memory
  sev.clear();
  
  sysmatrix = synmatrix - fematrix;
  
  // free some memory
  synmatrix = 0;
  femmatrix = 0;
  
  if (sysmatrix.get_rep() == 0)
  {
    error("DMDBuildDomain: Could not build system sparse matrix");
    return (false);  
  }

  if(!(numericalgo.ReverseCuthillmcKee(sysmatrix,sysmatrix,mapping)))
  {
    error("DMDBuildDomain: Matrix reordering failed");
    return (false);    
  }
  
  // Reorder domain properties
  nodetype = mapping*nodetype;
  volvec = mapping*volvec;

  if (!(dataioalgo.WriteMatrix(filename_systemmatrix,sysmatrix,"CardioWave Sparse Matrix")))
  {
    error("DMDBuildDomain: Could not write system matrix");  
    return (false);
  }
  if (!(dataioalgo.WriteMatrix(filename_volumevector,volvec,"CardioWave FP Vector")))
  {
    error("DMDBuildDomain: Could not write volume vector");  
    return (false);
  }

  if (!(dataioalgo.WriteMatrix(filename_nodetypevector,nodetype,"CardioWave Byte Vector")))
  {
    error("DMDBuildDomain: Could not write nodetype vector");  
    return (false);
  }

  sysmatrix = 0;
  nodetype = 0;
  volvec = 0;
  
  MatrixHandle imapping = mapping->transpose();
  if (imapping.get_rep() == 0)
  {
    error("DMDBuildDomain: Could not build mapping matrix");
    return (false);      
  }
  
  if (!(numericalgo.ResizeMatrix(imapping,imapping,num_totalnodes,num_volumenodes)))
  {
    error("DMDBuildDomain: Could not resize mapping matrix");
    return (false);        
  }  

  if (!(fieldsalgo.ClearAndChangeFieldBasis(elementtype,elementtype,"Linear")))
  {
    error("DMDBuildDomain: Could not clear field");
    return (false);            
  }

  BundleHandle VisBundle = scinew Bundle();
  BundleHandle VolumeConductor = scinew Bundle();
  
  VolumeConductor->setField("field",elementtype);
  VolumeConductor->setMatrix("mapping",imapping);
  VisBundle->setBundle("volumeconductor",VolumeConductor);
  
  for (size_t p=0; p <nummembranes; p++)
  {
    std::ostringstream oss;
    oss << "membrane_" << p+1;
    
    BundleHandle MembraneBundle = scinew Bundle;
    MembraneBundle->setField("field",membranefield[p]);
    MembraneBundle->setMatrix("mapping",membranemapping[p]);
    VisBundle->setBundle(oss.str_c(),MembraneBundle);
  }

  if (!(dataioalgo.WriteBundle(filename_visbundle,VisBundle)))
  {
    error("DMDBuildDomain: Could not write visualization information");
    return (false);     
  }

  return (true);

} // end namespace SCIRun
