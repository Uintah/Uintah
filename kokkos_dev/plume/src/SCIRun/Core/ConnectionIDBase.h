/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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


/*
 *  ConnectionIDBase.h: Baseementation of the SCI CCA Extension
 *                    ComponentID interface for SCIRun
 *
 *  Written by:
 *   Yarden Livnat
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   Sept 2005
 *
 *  Copyright (C) 2005 SCI Institute
 *
 */

#ifndef SCIRun_ConnectionIDBase_h
#define SCIRun_ConnectionIDBase_h

namespace SCIRun {

  using namespace sci::cca;

  template<class Interface>
  class ConnectionIDBase : public Interface
  {
  public:
    typedef ConnectionID::pointer pointer;
    typedef ComponentID::pointer cid_pointer;
    
    ConnectionIDBase( const cid_pointer &user,
		      const std::string &userPortName,
		      const cid_pointer &provider, 
		      const std::string &providerPortName )
      : provider(provider), providerPortName(providerPortName), user(user), userPortName(userPortName)
    {}

    virtual ~ConnectionIDBase() {}
    
    /** ? */
    virtual cid_pointer getProvider() { return provider; }

    /** ? */
    virtual cid_pointer getUser() { return user; }

    /** ? */
    virtual std::string getProviderPortName() { return providerPortName; }

    /** ? */
    virtual std::string getUserPortName() { return userPortName; }

  private:
    cid_pointer provider;
    std::string providerPortName;
    cid_pointer user;
    std::string userPortName;
  };
  
} // namespace SCIRun

#endif
