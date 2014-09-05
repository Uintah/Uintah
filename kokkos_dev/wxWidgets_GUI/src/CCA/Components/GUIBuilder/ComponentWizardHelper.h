#ifndef CCA_Components_GUIBuilder_ComponentWizardHelper_h
#define CCA_Components_GUIBuilder_ComponentWizardHelper_h

#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#include <SCIRun/ComponentSkeletonWriter.h>
#include <CCA/Components/GUIBuilder/GUIBuilder.h>
#include <CCA/Components/GUIBuilder/CodePreviewDialog.h>
#include <vector>
namespace GUIBuilder {
class ComponentWizardHelper
{
 public: 
  ComponentWizardHelper::ComponentWizardHelper(const sci::cca::GUIBuilder::pointer& bc,const std::string,const std::string,const std::vector<PortDescriptor*>,const std::vector<PortDescriptor*>,const bool isWithSidl);
  ~ComponentWizardHelper();
  void ComponentWizardHelper::createComponent();
  void ComponentWizardHelper::previewCode(std::string &sopf,std::string &sosf,std::string &somf, std::string &sosidlf);
  std::string ComponentWizardHelper::getTempDirName();
  sci::cca::GUIBuilder::pointer builder;
  std::string compName;
  std::string compDirName;
  bool isPreviewed;
  std::vector <PortDescriptor*> providesPortsList;
  std::vector <PortDescriptor*> usesPortsList;
  bool isWithSidl;
};
}
#endif
