/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  SamplerInterface.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <Packages/MIT/Dataflow/Modules/Metropolis/Sampler.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/SamplerInterface.h>

namespace MIT {
  

SamplerInterface::SamplerInterface( Sampler *part, PartInterface *parent )
  : PartInterface( part, parent, "SamplerInterface" ), sampler_(part)
{
  num_iterations_ = 1000;
  current_iter_ = 1;
  subsample_ = 1;
  kappa_ = 0.2;
}
 
SamplerInterface::~SamplerInterface()
{
}

void
SamplerInterface::go( int i)
{
  sampler_->go( i );
}

void
SamplerInterface::theta( vector<double> *t )
{
  Sampler* part = (Sampler*) part_;
  part->theta_(t);
}
void
SamplerInterface::sigma( vector<vector<double> > *s)
{
  for (int i = 0; i < s->size(); i++)
    for (int j = 0; j < (*s)[i].size(); j++)
    { //to be completed... <something>[i][j] = s[i][j];
    }
}


} // namespace MIT

