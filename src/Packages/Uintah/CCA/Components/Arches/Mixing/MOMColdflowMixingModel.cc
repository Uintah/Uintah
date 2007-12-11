//----- MOMColdflowMixingModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Mixing/MOMColdflowMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Core/Geometry/IntVector.h>
using namespace Uintah;
using namespace SCIRun;

//****************************************************************************
// Default constructor for MOMColdflowMixingModel
//****************************************************************************
MOMColdflowMixingModel::MOMColdflowMixingModel(bool calcReactingScalar,
					       bool calcEnthalpy,
					       bool calcVariance)
  :MixingModel(),
  d_calcReactingScalar(calcReactingScalar),
  d_calcEnthalpy(calcEnthalpy),
  d_calcVariance(calcVariance)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
MOMColdflowMixingModel::~MOMColdflowMixingModel()
{
}

//****************************************************************************
// Problem Setup for MOMColdflowMixingModel
//****************************************************************************
void 
MOMColdflowMixingModel::problemSetup(const ProblemSpecP& params)
{

  if (d_calcReactingScalar||d_calcEnthalpy||d_calcVariance)
    throw InvalidValue("MOMColdflowMixingModel can only be a function of mixture fraction",
                       __FILE__, __LINE__ );
  ProblemSpecP db = params->findBlock("MOMColdFlowMixingModel");

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
MOMColdflowMixingModel::computeProps(const InletStream& inStream,
				     Stream& outStream)
{
  double local_den = 0.0;
  double mixFracSum = 0.0;
  double localTemp = 0.0;
  double drhodf = 0.0;
  double volcoal = inStream.cellvolume * inStream.d_mixVars[0];
  double volair  = inStream.cellvolume * (1.0 - inStream.d_mixVars[0]);

  local_den = (d_streams[0].d_density*volcoal + d_streams[1].d_density*volair)/inStream.cellvolume;

  outStream.d_density = local_den;
  outStream.d_temperature = 0.0;
  outStream.d_drhodf = 0.0;
}

