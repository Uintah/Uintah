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
 *  IComAddress.h 
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */
 
#ifndef JGS_SCI_CORE_ICOM_ADDRESS_H
#define JGS_SCI_CORE_ICOM_ADDRESS_H 1

#include <Core/Malloc/Allocator.h>
#include <Core/ICom/IComBase.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Containers/LockingHandle.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/utsname.h>

#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#ifndef INET6_ADDRSTRLEN
#define INET6_ADDRSTRLEN 46
#endif

#ifndef INET_ADDRSTRLEN
#define INET_ADDRSTRLEN 16
#endif


namespace SCIRun {

class IComAddress;

typedef std::vector<char> IPaddress;

class IComAddress : public IComBase {

public:

	// Constructors
	// Create an empty not valid address
	IComAddress();
	
	// Create an address from a URL name and a port number or service name
	// Optional the preference for a protocol can be specified, mainly for
	// Servers so they can have both a ipv4 and ipv6 server

	// protocol:
	// this is the communication protocol
	// Currently it accepts "scirun", "internal", and "sscirun":
	//   -scirun : packet communication over socket over the internet ipv4 or ipv6
	//   -sciruns : same as scirun but using SSL sockets
	//   -internal : Internal communication, not to be routed over the internet
	
	// Both scirun and sciruns need an IP/DNS address and a portnumber/servicename to function
	// The internal socket only needs a name.
	// A fourth argument called iptype indicates whether an IPv4 or IPv6 address is prefered

	IComAddress(std::string protocol, std::string name, std::string serv, std::string iptype = std::string(""));
	
	// Create an address from a internalname, this is a internal socket
	// This enables the internal implementation of the sockets, which is nothing
	// more then transferring packets from one thread to antoher thread.
	// Do not confuse these with unix_sockets, which are pipes to other programs
	IComAddress(std::string, std::string name); 
	
	// Create an address from a sockaddr structure
	// This function is mainly for internal usage
	// within the Socket classes
	IComAddress(std::string protocol, sockaddr *sa);
	
	// Create an address from URL type descriptor
	// e.g. scirun://sci.utah.edu:9950 or ip addresses scirun://255.255.130.120:9950
	IComAddress(std::string address);
	
	// Desctructor
	~IComAddress();
	
	// Check the status of an address:
	// isvalid     - only valid addresses can be used with socketes
	// isinternal  - This address only works with the internal communication model
	// isipv4      - Whether it is a ipv4 internet address
	// isipv6      - Whether it is a ipv6 internet address
	//
	// Since a call to the DNS can result into multiple addresses addressnum selects which address
	// is being used. Normally the first one should work fine.
	
	bool		isvalid();
	bool		isinternal();
	bool		isipv4(int addressnum = 0);
	bool		isipv6(int addressnum = 0);
	
	// Set the address after the object has been created
	// The return value tells whether the address is valid or not
	// Use the addresstype field only for servers that need to be started with a
	// certain protocol. Leave the name field empty for a internal address (ANY)
	
	// These functions are essentially the same as the constructor functions except that they reuse the
	// current object
	
	bool		setaddress(std::string protocol, std::string name,std::string serv,std::string iptype = "");
	bool		setaddress(std::string protocol, sockaddr* sa);
	bool		setaddress(std::string protocol, std::string name);
	bool		setaddress(std::string urlname);
	bool		setlocaladdress();
	
	// Clear an address and make it not valid
	void		clear();
	
	// Compare whether an address points to the same host
	bool		operator==(const IComAddress& a);
	
	// Get all the information out of an address
	
	// Get the port number
	unsigned short		getport(int addressnum = 0);
	
	// Get the IPaddress as a vector of bytes
	IPaddress   getipaddress(int addressnum = 0);
	
	// Get the Canonical name of a host as supplied by the DNS
 	std::string getinetname(int addressnum = 0);
	
	// Get the IP address as a string (no DNS, just xx.xx.xx.xx)
	std::string getipname(int addressnum = 0);
	
	// Get the name of the service if it is there otherwise it is just the portnumber as a string
	std::string getservname(int addressnum = 0);

	// Get the sockaddr structure associated with this structure
	sockaddr*   getsockaddr(int addressnum = 0);
	
	// Get the length of the sockaddr structure, so we do not need to compute it, based on the family entry
	socklen_t	getsockaddrlen(int addressnum = 0);

	// For Inet addresses only, sometimes one url has multiple
	// ipaddress, both IPv4 and IPv6. IComAddress creates all of them
	// and IComSocket tries them one by one until it establishes a
	// connection. This function just returns the number of addresses you can try
	int			getnumaddresses();
	
	// Once ICom has found an address that works, it uses selectaddress(number)
	// to reduce the number of addresses associated with IComAddress to one
	// After calling this function the selected address is addressnum=0
	bool		selectaddress(int addressnumber);
	
	// For the internal database it is much easier that any address is allowed
	// as long as it fits into a string. The followinbg function returns that
	// unique name. Makes a URL kind of name
	std::string getinternalname();

	// Get the protocol used in the address
	// To allow for future extensions, the protocol is entered as a string
	// Currently ony three are accepted, the others result in an error
	// However the code should be easy to adjust for entering an additional protocol
	std::string getprotocol();
	bool		setprotocol(std::string protocol);	
	
	// Combine the full address into a url
	// This becomes handy when sending the address
	// over the internet
	std::string	geturl(int addressnum = 0);
	
	// For debuging whether this class functions properly
	std::string	printaddress();
	
	//	The LockingHandle class uses clono to create a new copy
	IComAddress	*clone();
	
  public:
	// NOTE: Currenly the object is not thread-safe, meaning that calls
	// are NOT surrounded by locks!! Hence changing and using an address
	// at the same time is not a good idea. Hence often addresses are 
	// created and not changed anymore. In this case the object can be
	// used without endangering the underlying thread mode. If used between
	// different threads, use a handle to make sure it is being destroyed,
	// or copy the object as a whole. The latter strategy is used in most
	// of the ICom functions.
  
  private:

	std::string					protocol_;				// Which protocol is to be used

	bool						isinternal_;			// Is the address internal ?
	bool						isvalid_;				// IS the address valid ?
	
	std::vector<IPaddress>		ipaddress_;				// The IP address
	std::vector<unsigned short> portnum_;				// The port number
	
  	std::vector<std::string>	inetname_;				// internet name of the address
	std::vector<std::string>	ipname_;				// ipname of the address (ip number in text)
	std::vector<std::string>	servname_;				// Name of the port (service name)
	
	std::vector<sockaddr_in>	sin_;					// Ipv4 addresses
	std::vector<sockaddr_in6>	sin6_;					// Ipv6 addresses
	
	
};

}

#endif

