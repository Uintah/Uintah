/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- ColdflowMixingModel.cc --------------------------------------------------

#include <CCA/Components/Arches/Mixing/ColdflowMixingModel.h>
#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Geometry/IntVector.h>
using namespace Uintah;

//****************************************************************************
// Default constructor for ColdflowMixingModel
//****************************************************************************
ColdflowMixingModel::ColdflowMixingModel(bool calcReactingScalar,
                                         bool calcEnthalpy,
                                         bool calcVariance):
                                  MixingModel(),
                                  d_calcReactingScalar(calcReactingScalar),
                                  d_calcEnthalpy(calcEnthalpy),
                                  d_calcVariance(calcVariance)
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
  if (d_calcReactingScalar||d_calcEnthalpy||d_calcVariance)
    throw InvalidValue("ColdflowMixingModel can only be a function of mixture fraction",
                       __FILE__, __LINE__ );
  ProblemSpecP db = params->findBlock("ColdFlowMixingModel");

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
    throw InvalidValue("Computed zero density in ColdflowMixingModel", __FILE__, __LINE__ );
  else
    outStream.d_density = (1.0/local_den);
}

