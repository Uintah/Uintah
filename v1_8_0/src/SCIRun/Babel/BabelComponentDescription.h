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
 *  BabelComponentDescription.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_BabelComponentDescription_h
#define SCIRun_Framework_BabelComponentDescription_h

#include <SCIRun/ComponentDescription.h>

namespace SCIRun {
  class BabelComponentModel;
  class BabelComponentDescription : public ComponentDescription {
  public:
    BabelComponentDescription(BabelComponentModel* model);
    virtual ~BabelComponentDescription();

    virtual std::string getType() const;
    virtual const ComponentModel* getModel() const;
  protected:
    friend class BabelComponentModel;
    BabelComponentModel* model;
    std::string type;

  private:
    BabelComponentDescription(const BabelComponentDescription&);
    BabelComponentDescription& operator=(const BabelComponentDescription&);
  };
} //namespace SCIRun

#endif

