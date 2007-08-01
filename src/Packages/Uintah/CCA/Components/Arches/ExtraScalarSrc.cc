//----- ExtraScalarSrc.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/ExtraScalarSrc.h>

using namespace std;
using namespace Uintah;

//****************************************************************************
// Interface constructor for ExtraScalarSrc
//****************************************************************************
ExtraScalarSrc::ExtraScalarSrc(const ArchesLabel* label, 
			       const MPMArchesLabel* MAlb,
                               const VarLabel* d_src_label):
                               d_lab(label), d_MAlab(MAlb),
                               d_scalar_nonlin_src_label(d_src_label)
{
  d_scalar_nonlin_src_label->addReference();
}

//****************************************************************************
// Destructor
//****************************************************************************
ExtraScalarSrc::~ExtraScalarSrc()
{
  VarLabel::destroy(d_scalar_nonlin_src_label);
}

