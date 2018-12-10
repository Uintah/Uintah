#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/ParameterNotFound.h>

using namespace std;
using namespace Uintah;

SourceTermBase::SourceTermBase( std::string src_name, MaterialManagerP& materialManager,
                                vector<std::string> required_labels, std::string type ) :
_src_name(src_name), _type(type), _materialManager( materialManager ), _required_labels(required_labels)
{
  _init_type = "constant";
  _stage = -1;
  _mult_srcs.clear();
  _simulationTimeLabel = VarLabel::find(simTime_name);

}

SourceTermBase::~SourceTermBase()
{
  VarLabel::destroy(_src_label);
}

void
SourceTermBase::set_stage( const int stage )
{
  if ( _stage == -1 ) {

    _stage = stage;

  } else if ( stage != _stage ){

    std::stringstream msg;

    msg << "Error: When trying to assign an algorithmic stage to source: " << _src_name << endl <<
      "This source has been associated with a transport equation being solved at a different stage.  \n Please check your input file and try again. " << std::endl;

    throw ProblemSetupException(msg.str(),__FILE__,__LINE__);

  }
}
