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
 *  BuilderWindow.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_BuilderWindow_h
#define SCIRun_Framework_BuilderWindow_h

#include <Core/CCA/ccaspec/cca_sidl.h>
#include <map>
#include <vector>
class QPopupMenu;
#include <qmainwindow.h>

namespace SCIRun {
  class BuilderWindow;
  class MenuTree : public QObject {
    Q_OBJECT
  public:
    MenuTree(BuilderWindow* builder);
    virtual ~MenuTree();
    gov::cca::ComponentDescription cd;
    void add(const std::vector<std::string>& name, int nameindex,
	     const gov::cca::ComponentDescription& desc,
	     const std::string& fullname);
    void coalesce();
    void populateMenu(QPopupMenu*);
  private:
    std::map<std::string, MenuTree*> child;
    BuilderWindow* builder;
  private slots:
    void instantiateComponent();
    
  };
  class BuilderWindow : public QMainWindow, public gov::cca::ComponentEventListener_interface {
    Q_OBJECT
  protected:
    void closeEvent( QCloseEvent* );

  private slots:
    void save();
    void saveAs();
    void load();
    void insert();
    void clear();
    void addInfo();
    void exit();

    void about();
  public:
    BuilderWindow(const gov::cca::Services& services);
    virtual ~BuilderWindow();

    void instantiateComponent(const gov::cca::ComponentDescription&);

    // From gov::cca::ComponentEventListener
    void componentActivity(const gov::cca::ComponentEvent& e);
  private:
    gov::cca::Services services;
    void buildPackageMenus();
    BuilderWindow(const BuilderWindow&);
    BuilderWindow& operator=(const BuilderWindow&);
  };
}

#endif
