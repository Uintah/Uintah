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
 *  CCAComponentDescription.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_CCAComponentDescription_h
#define SCIRun_Framework_CCAComponentDescription_h

#include <SCIRun/ComponentDescription.h>
#include <Core/CCA/spec/cca_sidl.h>

namespace SCIRun {
  class CCAComponentModel;
  class CCAComponentDescription : public ComponentDescription {
  public:
    CCAComponentDescription(CCAComponentModel* model);
    virtual ~CCAComponentDescription();

    virtual std::string getType() const;
    virtual const ComponentModel* getModel() const;
    virtual std::string getLoaderName() const;
    void setLoaderName(const std::string& loaderName);
  protected:
    friend class CCAComponentModel;
    friend class SCIRunLoader;
    CCAComponentModel* model;
    std::string type;
    std::string loaderName;
  private:
    CCAComponentDescription(const CCAComponentDescription&);
    CCAComponentDescription& operator=(const CCAComponentDescription&);
  };
}

#endif
