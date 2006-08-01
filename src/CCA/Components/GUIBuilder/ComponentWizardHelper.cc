#include <SCIRun/ComponentSkeletonWriter.h>
#include <CCA/Components/GUIBuilder/GUIBuilder.h>
#include <CCA/Components/GUIBuilder/CodePreviewDialog.h>
#include <Core/OS/Dir.h>
#include <CCA/Components/GUIBuilder/ComponentWizardHelper.h>
#include <vector>
namespace GUIBuilder {

using namespace SCIRun;


ComponentWizardHelper::ComponentWizardHelper(const sci::cca::GUIBuilder::pointer& bc, const std::string cName,const std::string cdName,const std::vector <PortDescriptor*> ppList,const std::vector <PortDescriptor*> upList,const bool isWithSidl)
    :builder(bc),compName(cName),compDirName(cdName),providesPortsList(ppList),usesPortsList(upList),isWithSidl(isWithSidl)
{
}
ComponentWizardHelper::~ComponentWizardHelper()
{
  std::string tmpDir = getTempDirName();
  Dir d1(tmpDir);

  if (d1.exists()) {
    d1.remove(std::string("tempheader.txt"));
    d1.remove(std::string("tempsource.txt"));
    d1.remove(std::string("tempsubmake.txt"));
    d1.remove();
  }
}
void ComponentWizardHelper::createComponent()
{
   Dir destDir = Dir(compDirName);
      if (!destDir.exists()) {
        destDir.create(compDirName);
      }
      //  std::string tmpDir = getTempDirName();
//   Dir temp(tmpDir);
//   if((temp.exists())&&(isPreviewed)) {
      
//       destDir = Dir(compDirName + ComponentSkeletonWriter::DIR_SEP + compName + std::string(".h") );
//       temp.copy("tempheader.txt", destDir);
//       destDir = Dir(compDirName + ComponentSkeletonWriter::DIR_SEP + compName + std::string(".cc") );
//       temp.copy("tempsource.txt", destDir);
//       destDir = Dir(compDirName + ComponentSkeletonWriter::DIR_SEP + std::string("sub.mk") );
//       temp.copy("tempsubmake.txt", destDir);
//       destDir = Dir(compDirName + ComponentSkeletonWriter::DIR_SEP + compName + std::string(".sidl") );
//       temp.copy("tempsidl.txt", destDir);
//   }
//    else {
    std::string dirName = compDirName;
    std::string sopf(compDirName + ComponentSkeletonWriter::DIR_SEP  + compName + ".h");
    std::string sosf(compDirName + ComponentSkeletonWriter::DIR_SEP  + compName + ".cc");
    std::string somf(compDirName + ComponentSkeletonWriter::DIR_SEP  + "sub.mk");
    std::string sosidlf(compDirName + ComponentSkeletonWriter::DIR_SEP  + compName + ".sidl");
    ComponentSkeletonWriter newComponent(compName,compDirName, providesPortsList,usesPortsList,isWithSidl);
    if(isWithSidl)
      newComponent.GenerateWithSidl(sopf,sosf,somf,sosidlf);
    else
      newComponent.generate(sopf,sosf,somf);
    //}
}
void ComponentWizardHelper::previewCode(std::string &sopf,std::string &sosf,std::string &somf, std::string &sosidlf)
{
  
  isPreviewed=true;
  std::string tmpDir = getTempDirName();
  Dir d1 = Dir(tmpDir);
  if (!d1.exists()) {
    Dir::create(tmpDir);
  }
  sopf = tmpDir + ComponentSkeletonWriter::DIR_SEP + "tempheader.txt";
  sosf = tmpDir + ComponentSkeletonWriter::DIR_SEP + "tempsource.txt";
  somf = tmpDir + ComponentSkeletonWriter::DIR_SEP + "tempsubmake.txt";
  sosidlf = tmpDir + ComponentSkeletonWriter::DIR_SEP + "tempsidl.txt";
  ComponentSkeletonWriter newComponent(compName,compDirName, providesPortsList,usesPortsList,isWithSidl);
  if(isWithSidl)
    newComponent.GenerateWithSidl(sopf,sosf,somf,sosidlf);
  else
    newComponent.generate(sopf,sosf,somf);
}
std::string ComponentWizardHelper::getTempDirName()
{
  std::string tmp(builder->getConfigDir());
  std::string home (getenv("HOME"));
  std::string tmpDirName = std::string(tmp  + ComponentSkeletonWriter::DIR_SEP + "ComponentGenerationWizard");
  return tmpDirName;

}
}
