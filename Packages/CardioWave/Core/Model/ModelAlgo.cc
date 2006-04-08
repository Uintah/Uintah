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

namespace CardioWave {

using namespace SCIRun;
using namespace ModelCreation;


ModelAlgo::ModelAlgo(ProgressReporter* pr) :
  AlgoLibrary(pr)
{
}

bool ModelAlgo::DMDBuildMembraneTable(FieldHandle ElementType, FieldHandle MembraneModel, MatrixHandle CompToGeom, MatrixHandle NodeLink, MatrixHandle ElemLink, MembraneTable& MembraneTable, MatrixHandle& MappingMatrix)
{
  BuildMembraneTableAlgo algo;
  return(algo.BuildMembraneTable(pr_,ElementType,MembraneModel,CompToGeom,NodeLink, ElemLink,MembraneTable,MappingMatrix));
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
  for (size_t k=0; k <MembraneTable.size(); k++)
  {
    dataptr[p++] = static_cast<double>(MembraneTable[k].node0);
    dataptr[p++] = static_cast<double>(MembraneTable[k].node1);
    dataptr[p++] = static_cast<double>(MembraneTable[k].node2);
    dataptr[p++] = static_cast<double>(MembraneTable[k].surface);
  }
  
  return (true);
}

bool ModelAlgo::DMDBuildSimulator(BundleHandle Model, std::string filename)
{

  return (true);
}
  

} // end namespace SCIRun
