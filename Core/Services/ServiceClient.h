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


// ServiceClient.h

#ifndef CORE_SERVICES_SERVICECLIENT_H
#define CORE_SERVICES_SERVICECLIENT_H 1

#include <Core/ICom/IComAddress.h>
#include <Core/ICom/IComSocket.h>
#include <Core/Services/ServiceBase.h>
#include <Core/Services/ServiceDB.h>
#include <Core/Services/Service.h>
#include <Core/Thread/Thread.h>
#include <Core/Containers/LockingHandle.h>
#include <string>


namespace SCIRun {

class ServiceClient: public ServiceBase {
  public:

    // Constructor/destructor
    ServiceClient();
    virtual ~ServiceClient();

    // Run will be called by the thread environment as the entry point of
    // a new thread.

    bool  open(IComAddress address, std::string servicename, int session, std::string passwd);
    bool  close();
  
    IComSocket  getsocket();
    
    std::string geterror();
    std::string getremoteaddress();
    std::string getversion();
    std::string getsession();
    void        setsession(int session);
    
    ServiceClient*  clone();

    bool    send(IComPacketHandle &packet);
    bool    recv(IComPacketHandle &packet);
    bool    poll(IComPacketHandle &packet);
    void    seterror(std::string);
    void    clearerror();
  
  public:
    Mutex    lock;
    int      ref_cnt;
  
  private:

    int          session_;
    std::string  version_;
    std::string  error_;
    int          errno_;
    IComSocket   socket_;
};

typedef LockingHandle<ServiceClient> ServiceClientHandle;

inline bool ServiceClient::send(IComPacketHandle &packet)
{
  if(socket_.send(packet) == false) 
  {
    seterror(socket_.geterror());
    return(false);
  }
  clearerror();
  return(true);
}

inline bool ServiceClient::recv(IComPacketHandle &packet)
{
  if(socket_.recv(packet) == false) 
  {
    seterror(socket_.geterror());
    return(false);
  }
  clearerror();
  return(true);
}

inline bool ServiceClient::poll(IComPacketHandle &packet)
{
  if(socket_.poll(packet) == false) 
  {
    seterror(socket_.geterror());
    return(false);
  }
  clearerror();
  return(true);
}

inline std::string ServiceClient::geterror()
{
  return(error_);
}

inline std::string ServiceClient::getversion()
{
  return(version_);
}

inline void ServiceClient::seterror(std::string error)
{
  error_ = error;
}

inline void ServiceClient::clearerror()
{
  error_ = "";
}  

inline std::string ServiceClient::getremoteaddress()
{
  IComAddress address;
  socket_.getremoteaddress(address);
  return(address.geturl());
}

inline std::string ServiceClient::getsession()
{
  std::ostringstream oss;
  oss << session_;
  return(oss.str());
}

inline IComSocket ServiceClient::getsocket()
{
  return(socket_);
}

inline void ServiceClient::setsession(int session)
{
    session_ = session;
}

}

#endif
