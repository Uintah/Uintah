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


// Service.h

#ifndef CORE_SERVICES_SERVICE_H
#define CORE_SERVICES_SERVICE_H 1

#include <Core/ICom/IComSocket.h>
#include <Core/ICom/IComAddress.h>
#include <Core/ICom/IComPacket.h>

#include <Core/Thread/Mutex.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>

#include <Core/Services/ServiceLog.h>
#include <Core/Containers/LockingHandle.h>

#define DECLARE_SERVICE_MAKER(name) \
extern "C" Service* make_service_##name(ServiceContext &ctx) \
{ \
  return new name(ctx); \
}


namespace SCIRun {

class ServiceContext 
{
  public:
	std::string							servicename;
	int									session;
	std::string							packagename;
	std::map<std::string,std::string>   parameters;
	IComSocket							socket;
	ServiceLogHandle					log;
};


class Service;
typedef LockingHandle<Service> ServiceHandle;


class Service : public ServiceBase {
  public:

	// Constructor/destructor
	Service(ServiceContext &ctx);
	virtual ~Service();

	// Run will be called by the thread environment as the entry point of
	// a new thread.
	void			run();
	
	// Entrypoint of the actual function of the service
	virtual void	execute();

	// Get information about this services
	int				getsession();
    void            setsession(int session);
	std::string		getservicename();
	std::string		getpackagename();
	std::string		getparameter(std::string);
	IComSocket		getsocket();
	ServiceLogHandle getlog();
	void			putmsg(std::string line);
	// Communication functions
	
	inline bool			send(IComPacketHandle &packet);
	inline bool			recv(IComPacketHandle &packet);
	inline bool			poll(IComPacketHandle &packet);
	
	inline bool			getlocaladdress(IComAddress &address);
	inline bool			getremoteaddress(IComAddress &address);
	inline bool			isconnected();
	
	// Error retrieval mechanisms for communication errors
	
	inline std::string		geterror();
	inline int				geterrno();
	inline bool			haserror();
	
	// Error reporting, services log file
	
	void			errormsg(std::string error);
	void			warningmsg(std::string warning);
	
 public:
	Mutex			lock;
	int				ref_cnt;
	
	inline void			dolock();   // Locking Handle needs Mutex to be called lock so we cannot reuse that name
	inline void			unlock();
	
 private:
	
	ServiceContext	ctx_;
};


typedef	Service*  (*ServiceMaker)(ServiceContext &ctx);

inline void	Service::dolock()
{
	lock.lock();
}

inline void	Service::unlock()
{
	lock.unlock();
}

inline void Service::putmsg(std::string line)
{
	ctx_.log->putmsg(line);
}

inline std::string Service::geterror()
{
	return(ctx_.socket.geterror());
}

inline int Service::geterrno()
{
	return(ctx_.socket.geterrno());
}

inline bool Service::haserror()
{
	return(ctx_.socket.haserror());
}

inline int Service::getsession()
{
	return(ctx_.session);
}

inline void Service::setsession(int session)
{
    ctx_.session = session;
}

inline std::string Service::getservicename()
{
	return(ctx_.servicename);
}

inline std::string Service::getpackagename()
{
	return(ctx_.packagename);
}

inline std::string Service::getparameter(std::string name)
{
	return(ctx_.parameters[name]);
}

inline IComSocket Service::getsocket()
{
	return(ctx_.socket);
}

inline ServiceLogHandle Service::getlog()
{
	return(ctx_.log);
}


inline bool Service::send(IComPacketHandle &packet)
{
	return(ctx_.socket.send(packet));
}

inline bool Service::recv(IComPacketHandle &packet)
{
	return(ctx_.socket.recv(packet));
}

inline bool Service::poll(IComPacketHandle &packet)
{
	return(ctx_.socket.poll(packet));
}

inline bool Service::getlocaladdress(IComAddress &address)
{
	return(ctx_.socket.getlocaladdress(address));
}

inline	bool Service::getremoteaddress(IComAddress &address)
{
	return(ctx_.socket.getremoteaddress(address));
}

inline	bool Service::isconnected()
{
	return(ctx_.socket.isconnected());
}



}

#endif
