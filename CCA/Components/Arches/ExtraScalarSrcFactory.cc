/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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



#include <CCA/Components/Arches/ExtraScalarSrcFactory.h>
#include <CCA/Components/Arches/ExtraScalarSrc.h>
#include <CCA/Components/Arches/ZeroExtraScalarSrc.h>
#include <CCA/Components/Arches/CO2RateSrc.h>
#include <CCA/Components/Arches/SO2RateSrc.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>

using namespace std;
using namespace Uintah;

ExtraScalarSrcFactory::ExtraScalarSrcFactory()
{
}

ExtraScalarSrcFactory::~ExtraScalarSrcFactory()
{
}

ExtraScalarSrc* ExtraScalarSrcFactory::create(const ArchesLabel* label, 
                                              const MPMArchesLabel* MAlb,
                                              const VarLabel* d_src_label,
                                              const std::string d_src_name)
{
  if (d_src_name == "zero_src"){ 
    return(scinew ZeroExtraScalarSrc(label, MAlb, d_src_label));
  }
  else if (d_src_name == "co2_rate_src"){
    return(scinew CO2RateSrc(label, MAlb, d_src_label));
  }
  else if (d_src_name == "so2_rate_src"){
    return(scinew SO2RateSrc(label, MAlb, d_src_label));
  }
  else {
    throw ProblemSetupException("Unknown extra scalar source " +
                                d_src_name, __FILE__, __LINE__);
  }
  return 0;
}
