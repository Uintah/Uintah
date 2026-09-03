/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

#include <CCA/Components/MPM/Materials/Dissolution/DissolutionFactory.h>
#include <CCA/Components/MPM/Materials/Dissolution/NullDissolution.h>
#include <CCA/Components/MPM/Materials/Dissolution/ParticleBasedDissolution.h>
//#include <CCA/Components/MPM/Materials/Dissolution/ContactStressIndependent.h>
//#include <CCA/Components/MPM/Materials/Dissolution/ContactStressDependent.h>
//#include <CCA/Components/MPM/Materials/Dissolution/SaltPrecipitationModel.h>
//#include <CCA/Components/MPM/Materials/Dissolution/QuartzOvergrowth.h>
//#include <CCA/Components/MPM/Materials/Dissolution/NewQuartzOvergrowth.h>
#include <CCA/Components/MPM/Materials/Dissolution/CompositeDissolution.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <string>

using namespace std;
using namespace Uintah;

Dissolution* DissolutionFactory::create(const ProcessorGroup* myworld,
                                const ProblemSpecP& ps, MaterialManagerP &ss,
                                MPMLabel* lb, MPMFlags* flag)
{

   ProblemSpecP mpm_ps = 
     ps->findBlockWithOutAttribute("MaterialProperties")->findBlock("MPM");

   if(!mpm_ps){
    string warn = "ERROR: Missing either <MaterialProperties> or <MPM> block from input file";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
   }
   
   CompositeDissolution * dissolution_list = scinew CompositeDissolution(myworld,lb);

   for( ProblemSpecP child = mpm_ps->findBlock( "dissolution" ); 
                     child != nullptr; 
                     child = child->findNextBlock( "dissolution" ) ) {
     
     std::string dis_type;
     child->getWithDefault("type",dis_type, "null");
     
     if (dis_type == "null") {
      dissolution_list->add(scinew NullDissolution(myworld,ss,lb,flag));
      flag->d_doingDissolution=false;
      flag->d_computeNormals=false;
     }
     else if (dis_type == "particleBasedDissolution") {
      dissolution_list->add(scinew ParticleBasedDissolution(myworld,child,ss,lb,flag));
      flag->d_doingDissolution=true;
      flag->d_computeNormals=true;
     }
     else {
       cerr << "Unknown Dissolution Type R (" << dis_type << ")" << std::endl;;
       throw ProblemSetupException(" ERROR----->MPM:Unknown Dissolution type",
                                     __FILE__, __LINE__);
     }
   }

   // 
   if( dissolution_list->size() == 0 ) {
     proc0cout << "no dissolution - using null\n";
     dissolution_list->add(scinew NullDissolution(myworld,ss,lb,flag));
   }

   return dissolution_list;
}
