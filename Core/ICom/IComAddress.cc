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
 * FILE: IComAddress.cc
 * AUTH: Jeroen Stinstra
 */
 
 
#include <Core/ICom/IComAddress.h> 

#define JGS_SCIRUNS_DEFAULT_PORT "9554"
#define JGS_SCIRUN_DEFAULT_PORT "9553"

// Functions for addressing

namespace SCIRun {

// Create a global mutex for locking access to DNS, not thread-safe
// Each call into the kernel will be locked until the data has been
// processed.

SCIRun::Mutex DNSLock("dnslock");

IComAddress::IComAddress() :
isvalid_(false), isinternal_(false)
{ 
}

IComAddress::IComAddress(std::string address) :
isvalid_(false), isinternal_(false)
{ 
    setaddress(address);
}

IComAddress::IComAddress(std::string protocol, std::string name, std::string servname, std::string iptype) :
isvalid_(false), isinternal_(false)
{ 
    setaddress(protocol,name,servname,iptype);
}

IComAddress::IComAddress(std::string protocol, sockaddr* sa) :
isvalid_(false), isinternal_(false)
{
    setaddress(protocol,sa);
}

IComAddress::IComAddress(std::string protocol, std::string name) :
isvalid_(false), isinternal_(false)
{ 
    setaddress(protocol,name);
}

IComAddress::~IComAddress()
{
}

bool IComAddress::isvalid()
{
    return(isvalid_);
}

short IComAddress::getport(int addressnum)
{
    if ((isvalid_)&&(!isinternal_)) return(portnum_[addressnum]);
    return(0);
}

IPaddress IComAddress::getipaddress(int addressnum)
{
    if ((isvalid_)&&(!isinternal_)) 
    {
        return(ipaddress_[addressnum]);
    }
    else
    {
        return(IPaddress(0));
    }
}

std::string IComAddress::getinternalname()
{
    if (isinternal_) return(inetname_[0]);
}

void IComAddress::clear()
{
    isinternal_ = false;
    isvalid_ = false;
    portnum_.clear();    // vector for storing portnumbers
    inetname_.clear();    // vector for storing the inetnet "DNS" names
    ipname_.clear();    // vector for storing the ipaddress in text
    ipaddress_.clear(); // vector for storing the ipaddress bytes
    sin_.clear();        // vector for storing the ipv4 sockaddresses
    sin6_.clear();        // vector for storing the ipv6 sockaddresses
}


bool IComAddress::operator==(const IComAddress& a)
{
    if (a.isvalid_ == false) return(false);
    if (isvalid_ == false) return(false);
    if ((a.isinternal_ == true)&&(isinternal_ == true)) 
    { 
        if (a.inetname_[0] == inetname_[0]) return(true);
        return(false);
    }
    if ((a.isinternal_ != true)&&(isinternal_ != true))
    {
        for (size_t r=0; r < portnum_.size(); r++)
        {
            for (size_t q=0; q< a.portnum_.size(); q++)
            {
                {
                    if (ipaddress_[r].size() == a.ipaddress_[q].size())
                    if ((portnum_[r] == a.portnum_[q])||(servname_[r] == "any")||(a.servname_[q] == "any")) 
                    {
                        bool test = true;
                        for (size_t p=0; p < ipaddress_[r].size(); p++) if(a.ipaddress_[q][p] != ipaddress_[r][p]) test = false;
                        if (test) return(true);
                    }
                }
            }
        }
    }
    return(false);
}

int IComAddress::getnumaddresses()
{
    return(inetname_.size());
}

bool IComAddress::isipv4(int addressnum)
{
    if ((isvalid_)&&(!isinternal_)) if (addressnum < ipaddress_.size()) if(ipaddress_[addressnum].size() == 4) return(true);
    return(false);
}

bool IComAddress::isipv6(int addressnum)
{
    if ((isvalid_)&&(!isinternal_)) if (addressnum < ipaddress_.size()) if(ipaddress_[addressnum].size() == 16) return(true);
    return(false);
}

bool IComAddress::isinternal()
{
    if ((isinternal_)&&(isvalid_)) return(true);
    return(false);
}

bool IComAddress::setaddress(std::string protocol,std::string name,std::string servname,std::string iptype)
{
    clear();
    
    // The most complicated function in this class:
    
    // in case the user wants an internal address, use
    // the function that was made for internal addresses
    if (protocol == "internal") return(setaddress(protocol,name));
    
    // if the user did not give us any portnumbers to work with
    // get the default ones.
    
    // Here we should check the environment variables as well
    // Code for that should go here VVVVVVVVVVVVVV
    
    // Otherwisse default to build in values
    if ((servname == "")&&(protocol=="scirun")) servname = JGS_SCIRUN_DEFAULT_PORT;
    if ((servname == "")&&(protocol=="sciruns")) servname = JGS_SCIRUNS_DEFAULT_PORT;

    // If we still do not have a portnumber, just fail
    if (servname == "") return(false);
        
    // We need one of both to do a proper search    
    if ((servname == "any")&&(name=="")) return(false);
    
    // We know the protocol ...
    protocol_ = protocol;
    isinternal_ = false;
    isvalid_ = false;

    // Assume we do not specify what kind of address we want
    sa_family_t    socktype = AF_UNSPEC;
    // in case of a server you want to be able to specify whether you have a
    // ipv6 or ipv4 server. The following settings overrule the default setting
    if (iptype == "ipv4") socktype = AF_INET;
    if (iptype == "ipv6") socktype = AF_INET6;
    
    
    addrinfo *results, *res;
    addrinfo hints;

    // Lock all DNS functions
    DNSLock.lock();
    
    // Give some hints on what we are looking for
    hints.ai_flags = AI_CANONNAME;
    hints.ai_family = socktype;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = 0;
    hints.ai_addrlen = 0;
    hints.ai_canonname = 0;
    hints.ai_addr = 0;
    hints.ai_next = 0;
    
    try
    {
        results = 0;    // Make sure we do not free empty space

        if (name == "")
        {    // no name supplied, so we only want an any address template
            hints.ai_flags |= AI_PASSIVE;
            if ( ::getaddrinfo(0,servname.c_str(),&hints,&(results))) throw could_not_resolve_address();
        }
        else
        {
            if (servname == "any")
            {
                if ( ::getaddrinfo(name.c_str(),0,&hints,&(results))) throw could_not_resolve_address();            
            }
            else
            {
                if ( ::getaddrinfo(name.c_str(),servname.c_str(),&hints,&(results))) throw could_not_resolve_address();
            }
        }    
        
        
        if (results)
        {
            // Sort address in the following order: first ipv4 then ipv6
            res = results;
            
            // count the number of addresses that meet our specifications
            int numaddresses = 1;
            while (res->ai_next != 0) { numaddresses++; res = res->ai_next; }
            
            
            // we split the number of addresses into one ipv4 and one ipv6 cluster
            int numipv4 = 0;
            int numipv6 = 0;
            res = results;
            for (int p=0;p<numaddresses;p++)
            {
                if (res->ai_family == AF_INET) numipv4++;
                if (res->ai_family == AF_INET6) numipv6++;
                res = res->ai_next;
            }

            // total number of addresses spans both the ipv4 and ipv6 space
            numaddresses = numipv4+numipv6;

            // Set the canonical name of an address
            inetname_.resize(numaddresses);
            for (int p=0;p<numaddresses; p++) inetname_[p] = name;    // set the default name
            // If our search resulted in an unique DNS name, put this name in the inetname field for each address
            if (results->ai_canonname) for (int p=0;p<numaddresses; p++) inetname_[p] = std::string(results->ai_canonname);
            
            // Set the portname, this one is unique as well as we did supply it ourselves
            servname_.resize(numaddresses);
            for (int p=0;p<numaddresses; p++) servname_[p] = servname;

            // Set all other information retrieved using getaddrinfo()
            
            // Socketaddresses are split over sin_ and sin6_,
            // the first ones correspond to ipv4 addresses and the later ones to the ipv6 addresses
            sin_.resize(numipv4);
            sin6_.resize(numipv6);
            ipaddress_.resize(numaddresses);
            ipname_.resize(numaddresses);
            portnum_.resize(numaddresses);
            
            int sincnt = 0;    // count howmany ipv4 addresses we encountered sofar
            int sin6cnt = 0; // idem for ipv6
            res = results;
            for (size_t p=0;p<numaddresses;p++)
            {
                if (res->ai_family == AF_INET)
                {    // IPv4
                    sockaddr_in *saddr = reinterpret_cast<sockaddr_in *>(res->ai_addr);
                    // get sockaddr
                    sin_[sincnt] = *saddr;
                    char    *addr = reinterpret_cast<char *>(&(saddr->sin_addr));
                    ipaddress_[sincnt].resize(4);
                    // Copy the actual ipaddress
                    for (int q=0;q<4;q++) ipaddress_[sincnt][q] = addr[q];
                    portnum_[sincnt] = ntohs(saddr->sin_port);
                    // Copy the address as a text string
                    char    str[46];
                    if (!(::inet_ntop(AF_INET, &(ipaddress_[sincnt][0]), str, sizeof(str)))) throw could_not_resolve_address();
                    ipname_[sincnt] = std::string(str);
                    sincnt++;
                    
                }
                if (res->ai_family == AF_INET6)
                {    // IPv6
                    sockaddr_in6 *saddr = reinterpret_cast<sockaddr_in6 *>(res->ai_addr);
                    sin6_[sin6cnt] = *saddr;
                    char    *addr = reinterpret_cast<char *>(&(saddr->sin6_addr));
                    ipaddress_[sin6cnt+numipv4].resize(16);
                    for (int q=0;q<16;q++) ipaddress_[sin6cnt+numipv4][q] = addr[q];
                    portnum_[sin6cnt+numipv4] = ntohs(saddr->sin6_port);
                    char    str[46];
                    if (!(::inet_ntop(AF_INET6, &(ipaddress_[sin6cnt+numipv4][0]), str, sizeof(str)))) throw could_not_resolve_address();
                    ipname_[sin6cnt+numipv4] = std::string(str);
                    sin6cnt++;

                }
                
                res = res->ai_next;
            }

            
            // In case we did not find any suitable addresses
            if (numaddresses == 0) throw could_not_resolve_address();
            
            // free the results as it was memory that was allocated for us
            if (results) freeaddrinfo(results); results = 0;
        }
        DNSLock.unlock();    // unlock the DNS
        isvalid_ = true;
        return(true);

    }
    catch (...)
    {    // Make sure we do unlock the DNS and deallocate the memory used
        DNSLock.unlock();
        if (results) freeaddrinfo(results);
        clear();
        return(false);
    }
}


bool IComAddress::setaddress(std::string protocol, std::string name)
{
    clear();
    
    // in case the user does not want to have an internal address
    if (protocol != "internal")
    {
        return(setaddress(protocol,name,""));
    }

    // internal addresses are easy:
    // a string is used as an unique identifier
    
    protocol_ = protocol;
    inetname_.resize(1);
    inetname_[0] = name;
    isinternal_ = true;
    isvalid_ = true;
    return(true);
}


bool IComAddress::setprotocol(std::string protocol)
{
    if (!isvalid_) return(false);
    if (isinternal_) return(false);    // you cannot change the protocol of an internal address
    protocol_ = protocol;
    return(true);
}

bool IComAddress::setaddress(std::string str)
{
    std::string protocol;
    std::string name;
    std::string portnum;
    
    clear();
    
    protocol = "scirun";
    size_t protocolsep = str.find("://");
    if (protocolsep < str.size())
    {
        protocol = str.substr(0,protocolsep);
        str = str.substr(protocolsep+3);
    }
    
    size_t firstcolon = str.find(':');
    size_t firstdot = str.find('.');
    
    if (firstcolon >= str.size())
    {    // IPV4 with no port assignment or IPV6 using DNS
        name = str;
        portnum = "";
    }
    
    if (firstdot >= str.size())
    {    // IPV6 without port or internal name
        // internal name has no colons
        if (firstcolon < str.size())
        {    // a ':' but no '.' => must be an internalname with a port or
            // an ipv6 without port
            // an ipv6 needs to have second ':'
            std::string str2 = str.substr(firstcolon+1);
            size_t secondcolon = str2.find(':');
            if (secondcolon < str2.size())
            {
                size_t numcolon = 0;
                std::string str3 = str2;
                size_t poscolon = 0;
                while((poscolon = str3.find(':')) < str3.size()) {numcolon++; str3 = str3.substr(poscolon+1);}
                if (numcolon > 5)
                {
                    size_t lastcolon= str.rfind(':');
                    name = str.substr(0,lastcolon);
                    portnum = str.substr(lastcolon+1);
                }
                else
                {
                    name = str;
                    portnum = "";
                }
            } 
            else
            {
                name = str.substr(0,firstcolon);
                portnum = str.substr(firstcolon+1);
                if (protocol == "") protocol = "internal";
                if (protocol != "internal") return(false);
            }
        }
        else
        { // no colon, no dot => internal name
            name = str;
            portnum = "";
            if (protocol == "") protocol = "internal";
            if (protocol != "internal") return(false);
        }
    }
    if (firstcolon > firstdot)
    {    // must be an IPv4 address
        name = str.substr(0,firstcolon);
        portnum = str.substr(firstcolon+1);
    }

    if (firstdot < firstcolon)
    {    // must be an IPv4 address
        name = str.substr(0,firstdot);
        portnum = str.substr(firstdot+1);
    }
    
    // if there are any slashes before the namestring, remove them
    if (name[0] == '/') name = name.substr(1);
    if (name[0] == '/') name = name.substr(1);
    
    if (protocol == "internal")
    {    // for an internal address we do not need the portnumber
        return(setaddress(protocol,name));
    }
    else
    {    // this is a full internet address
        return(setaddress(protocol,name,portnum));
    }
}

bool IComAddress::setaddress(std::string protocol, sockaddr *sa)
{
    clear();
    
    // supply ineternet address as a socket address, this function is mainly intended for
    // the socket implementation to be able to translate a specific socket address back into
    // a string that is readable to the user
    
    if (protocol == "internal") return(false);
    
    if (sa->sa_family == AF_INET)
    {    // for ipv4 addresses
    
        // We only expect to have one unique address for this one
        inetname_.resize(1);
        servname_.resize(1);
        ipname_.resize(1);
        portnum_.resize(1);
        ipaddress_.resize(1);
        sin_.resize(1);
        
        // Unix design requires us to use a reinterpret_cast here
        sockaddr_in *saddr = reinterpret_cast<sockaddr_in *>(sa);
        sin_[0] = *saddr;    // store the full datastructure in the IComAddress object

        char    *addr = reinterpret_cast<char *>(&(saddr->sin_addr));    // this is in network order
        ipaddress_[0].resize(4);
        for (int p=0;p<4;p++) ipaddress_[0][p] = addr[p]; // copy the ipaddress byte by byte

        portnum_[0] = ntohs(saddr->sin_port);    // translate the portnumber from network order into local byte order
        
        char host[NI_MAXHOST];
        char serv[NI_MAXSERV];
    
        DNSLock.lock();    // getnameinfo is not thread safe hence use a mutex to gain exclusive access to the function
        ::getnameinfo(reinterpret_cast<sockaddr *>(&(sin_[0])),sizeof(sockaddr_in),host,NI_MAXHOST,serv,NI_MAXSERV,0);
        DNSLock.unlock();
        
        // Retrieve the ipname as text
        char    str[46];    // maximum for both ipv4 and ipv6
        // The next function needs better error management
        if (!(::inet_ntop(AF_INET, &(ipaddress_[0][0]), str, sizeof(str)))) str[0] = '\0';
        ipname_[0] = std::string(str);
        
        // Set the DNS names
        inetname_[0] = std::string(host);
        servname_[0] = std::string(serv);
    
        // we are done, so declare the address as valid
        isvalid_ = true;
        isinternal_ = false;
        protocol_ = protocol;
        return(true);
    }

    if (sa->sa_family == AF_INET6)
    {    // The same as before but now for ipv6
        inetname_.resize(1);
        servname_.resize(1);
        ipname_.resize(1);
        portnum_.resize(1);
        ipaddress_.resize(1);
        sin6_.resize(1);
        
        sockaddr_in6 *saddr = reinterpret_cast<sockaddr_in6 *>(sa);
        sin6_[0] = *saddr;

        char    *addr = reinterpret_cast<char *>(&(saddr->sin6_addr));
        ipaddress_[0].resize(16);
        for (int p=0;p<16;p++) ipaddress_[0][p] = addr[p];

        portnum_[0] = ntohs(saddr->sin6_port);
        
        char host[NI_MAXHOST];
        char serv[NI_MAXSERV];
    
        DNSLock.lock();
        ::getnameinfo(reinterpret_cast<sockaddr *>(&(sin6_[0])),sizeof(sockaddr_in6),host,NI_MAXHOST,serv,NI_MAXSERV,0);
        DNSLock.unlock();
        
        char    str[46];
        if (!(::inet_ntop(AF_INET6, &(ipaddress_[0][0]), str, sizeof(str)))) str[0] = '\0';
        ipname_[0] = std::string(str);
        
        inetname_[0] = std::string(host);
        servname_[0] = std::string(serv);
        
        isvalid_ = true;
        isinternal_ = false;
        protocol_ = protocol;
        return(true);
    }

    return(false);
}


sockaddr* IComAddress::getsockaddr(int numaddress)
{
    if (!isvalid_) return(0);
    // ordering of socket addresses: first the ipv4 and then the ipv6
    // addresses
    // use getsockaddrlen() to get the appropriate length of the socket
    if (numaddress+1 > sin_.size() )
    {    // ipv6
        numaddress -= sin_.size();
        if (numaddress < sin6_.size())
        {
            return(reinterpret_cast<sockaddr *>(&(sin6_[numaddress])));
        }
        else
        { // hence it must be ipv4
            return(0);
        }
    }
    else
    {
        return(reinterpret_cast<sockaddr *>(&(sin_[numaddress])));
    }
}

socklen_t    IComAddress::getsockaddrlen(int numaddress)
{
    if (!isvalid_) return(0);
    // ordering of socket addresses: first the ipv4 and then the ipv6
    // addresses
    if (numaddress+1 > sin_.size())
    {    // it has to be an ipv6 address
        if (numaddress < sin6_.size()) return(sizeof(sockaddr_in6));
        return(0);
    }
    else
    {    // otherwise it has to be ipv4
        return(sizeof(sockaddr_in));
    }
}

std::string IComAddress::getinetname(int numaddress)
{
    if ((!isvalid_)||(isinternal_)) return(std::string(""));
    if (inetname_.size() > numaddress)     return(inetname_[numaddress]);
    return(std::string(""));
}

std::string IComAddress::getipname(int numaddress)
{
    if ((!isvalid_)||(isinternal_)) return(std::string(""));
    if (ipname_.size() > numaddress)     return(ipname_[numaddress]);
    return(std::string(""));
}

std::string IComAddress::getservname(int numaddress)
{
    if ((!isvalid_)||(isinternal_)) return(std::string(""));
    if (servname_.size() > numaddress)     return(servname_[numaddress]);
    return(std::string(""));
}


std::string    IComAddress::geturl(int addressnum)
{
    std::string str("");    // initiate return string
    if (!isvalid_) return(str);    // invalid .....
    if (addressnum > inetname_.size()) return(str); // out of range
    
    if (isinternal_)    
    {    // an internal address is denoted as following
        // internal://my_internal_server_name:0
        // A portname is generated by set to zero as we do not use it
        str = "internal://" + inetname_[addressnum] + ":0";
    }
    if (!isinternal_)
    {
        // an ip address is structured according to the normal rules of IPaddresses
        // for the scirun protocol:
        // scirun://something.sci.utah.edu:9553
        //
        // As protocol does not automatically imply portnum, it is recommended to supply
        // a portnumber, hence it is generated in the output
        std::ostringstream oss;
        oss << portnum_[addressnum];
        str = protocol_ + "://" + inetname_[addressnum] + ":" + oss.str();
    }
    return(str);
}


std::string IComAddress::getprotocol()
{
    if (isvalid_) return(protocol_);    // invalid addresses do not have a protocol
    return(std::string(""));
}

bool IComAddress::selectaddress(int addressnum)
{
    if (!isvalid_) return(false);    // In case it is not valid do nothing
    if (addressnum >= inetname_.size()) return(false);    // See if number is too big
    if (isinternal_) return(true);    // internal address is always unique and hence will only have one entry
    
    // Get the data out of the structure, so we can resize the fields without losing
    // information.
    std::string inetname = inetname_[addressnum];
    std::string ipname = ipname_[addressnum];
    std::string servname = servname_[addressnum];
    IPaddress    ipaddress = ipaddress_[addressnum];
    short        portnum    = portnum_[addressnum];

    // resize the fields and put the selected field at the first place
    inetname_.resize(1);
    inetname_[0] = inetname;
    ipname_.resize(1);
    ipname_[0] = ipname;
    servname_[0].resize(1);
    servname_[0] = servname;
    
    ipaddress_.resize(1);
    ipaddress_[0] = ipaddress;
    portnum_.resize(1);
    portnum_[0] = portnum;
    
    // The socket address needs to be treated seperately as we sorted the data with first
    // the ipv4 addresses and the ipv6 addresses
    if (addressnum < sin_.size())
    {    // in case the selected one is an ipv4 address
        sockaddr_in sin = sin_[addressnum];
        sin_.resize(1);
        sin6_.resize(0);
        sin_[0] = sin;
    }
    else
    {    // in case of an ipv6 address
        sockaddr_in6 sin6 = sin6_[addressnum-sin_.size()];
        sin_.resize(0);
        sin6_.resize(1);
        sin6_[0] = sin6;
    }
    
    return(true);
}            

bool IComAddress::setlocaladdress()
{

    clear();
    struct utsname localname;
    
    if (::uname(&localname) < 0)
    {
        return(false);
    }
    
    std::string name(localname.nodename);
    return(setaddress("scirun",name,"any"));
}

// For debugging purposes
std::string IComAddress::printaddress()
{
    if (!isvalid()) return(std::string(""));
    
    std::ostringstream oss;
    oss << "Number of addresses:" << getnumaddresses() << "\n\n";
    
    if (!isinternal())
    {
        for (size_t p=0; p < portnum_.size(); p++)
        {
            oss << "IPaddress[" << p << "] =" ;
            for (int q=0; q < ipaddress_[p].size(); q++) oss << static_cast<unsigned short>(ipaddress_[p][q]) << ".";
            oss << "\n";
            oss << "Portnumber[" << p << "] = " << portnum_[p] << ".\n";
            oss << "DNSName[" << p << "] = " << inetname_[p] << ".\n";
            oss << "ServName[" << p << "] = " << servname_[p]  << ".\n";
            oss << "IPName[" << p << "] = " << ipname_[p] << ".\n\n";
        }
    }
    else
    {
        oss << "InternalAddress = " << inetname_[0] << ".\n";
    }
    
    return(oss.str());
}

IComAddress    *IComAddress::clone()
{
    IComAddress *address = scinew IComAddress;
    
    address->protocol_ = protocol_;
    address->isinternal_ = isinternal_;
    address->isvalid_ = isvalid_;    
    address->ipaddress_ = ipaddress_;
    address->portnum_ = portnum_;
    address->inetname_ = inetname_;
    address->ipname_ = ipname_;
    address->servname_ = servname_;
    address->sin_ = sin_;
    address->sin6_ = sin6_;
    
    return(address);
}

}

