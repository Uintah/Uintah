//----- ColdflowMixingModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/ColdflowMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Core/Geometry/IntVector.h>
using namespace Uintah;
using namespace SCIRun;

//****************************************************************************
// Default constructor for ColdflowMixingModel
//****************************************************************************
ColdflowMixingModel::ColdflowMixingModel():MixingModel()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
ColdflowMixingModel::~ColdflowMixingModel()
{
}

//****************************************************************************
// Problem Setup for ColdflowMixingModel
//****************************************************************************
void 
ColdflowMixingModel::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("ColdflowMixingModel");

  // Read the mixing variable streams, total is noofStreams 0 
  d_numMixingVars = 0;
  for (ProblemSpecP stream_db = db->findBlock("Stream");
       stream_db != 0; stream_db = stream_db->findNextBlock("Stream")) {

    // Create the stream and add it to the vector
    d_streams.push_back(Stream());
    stream_db->require("density", d_streams[d_numMixingVars].d_density);
    stream_db->require("temperature", d_streams[d_numMixingVars].d_temperature);
    ++d_numMixingVars;
  }
  d_numMixingVars--;
}

//****************************************************************************
// Actually compute the properties here
//****************************************************************************
void 
ColdflowMixingModel::computeProps(const InletStream& inStream,
				  Stream& outStream)
{
  // for combustion calculations mixingmodel will be called
  // this is similar to prcf.f
  double local_den = 0.0;
  double mixFracSum = 0.0;
  double localTemp = 0.0;
  double drhodf = 0.0;
  for (int ii = 0; ii < d_numMixingVars; ii++ ) {
    local_den += inStream.d_mixVars[ii]/d_streams[ii].d_density; 
    localTemp += inStream.d_mixVars[ii]*d_streams[ii].d_temperature; 
    mixFracSum += inStream.d_mixVars[ii];
  }
  local_den += (1.0 - mixFracSum)/d_streams[d_numMixingVars].d_density;
  localTemp += (1.0 - mixFracSum)*d_streams[d_numMixingVars].d_temperature;
  drhodf = (1.0-mixFracSum)*d_streams[0].d_density +
            inStream.d_mixVars[0]*d_streams[1].d_density;
  outStream.d_temperature = localTemp;
  outStream.d_drhodf = d_streams[0].d_density*d_streams[1].d_density*
    (d_streams[0].d_density - d_streams[1].d_density)/(drhodf*drhodf);
  if (local_den <= 0.0)
    throw InvalidValue("Computed zero density in ColdflowMixingModel" );
  else
    outStream.d_density = (1.0/local_den);
}

//
// $Log$
// Revision 1.5  2002/05/20 18:14:13  sparker
// Fix gcc 3.1 problems and warnings
//
// Revision 1.4  2002/02/28 03:05:46  rawat
// Added divergence constraint and modified pressure/outlet boundary condition.
//
// Revision 1.3  2001/07/27 20:51:40  sparker
// Include file cleanup
// Fix uninitialized array element
//
// Revision 1.2  2001/07/16 21:15:38  rawat
// added enthalpy solver and Jennifer's changes in Mixing and Reaction model required for ILDM and non-adiabatic cases
//
// Revision 1.1  2001/01/31 16:35:30  rawat
// Implemented mixing and reaction models for fire.
//
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
//

