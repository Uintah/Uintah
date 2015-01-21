
#include <CCA/Components/Arches/ExtraScalarSrcFactory.h>
#include <CCA/Components/Arches/ExtraScalarSrc.h>
#include <CCA/Components/Arches/ZeroExtraScalarSrc.h>
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
      if (d_src_name == "zero_src") 
        return(scinew ZeroExtraScalarSrc(label, MAlb, d_src_label));
      else {
        throw ProblemSetupException("Unknown extra scalar source " +
                                    d_src_name, __FILE__, __LINE__);
      }
      return 0;
}
