/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- MOMColdflowMixingModel.cc --------------------------------------------------

#include <CCA/Components/Arches/Mixing/MOMColdflowMixingModel.h>
#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Geometry/IntVector.h>
using namespace Uintah;

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
  double volcoal = inStream.cellvolume * inStream.d_mixVars[0];
  double volair  = inStream.cellvolume * (1.0 - inStream.d_mixVars[0]);

  local_den = (d_streams[0].d_density*volcoal + d_streams[1].d_density*volair)/inStream.cellvolume;

  outStream.d_density = local_den;
  outStream.d_temperature = 0.0;
  outStream.d_drhodf = 0.0;
}

