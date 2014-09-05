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


#include <net/if.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>

#include <iostream>
#include <sstream>
#include <cstdlib>

#include <Core/Util/Assert.h>
#include <Core/Util/GenerateUUID.h>

/*
  GenerateUUID.cc: custom UUID utility for those who don't have uuid.h available

  Written by:
    Ayla Khan
    Scientific Computing and Imaging Institute 
    University of Utah
    September 2004
    Copyright (C) 2004 SCI Institute
*/

std::string genUUID()
{
    std::string mac = getHWAddr();
    ASSERT(! mac.empty());

    struct timeval tv = { 0, 0 };
    struct timezone tz;

    if (gettimeofday(&tv, &tz) < 0) {
	std::cerr << "Error: gettimeofday failed." << std::endl;
    }

    int randVal = 0;
    srand(getpid() + tv.tv_sec);

    if ((randVal = rand()) < 0) {
	std::cerr << "Error: rand failed." << std::endl;
	return 0;
    }

    std::ostringstream outs;
    outs << mac << tv.tv_sec << tv.tv_usec << randVal;

    while (outs.str().length() < 32) {
	srand(randVal);
	randVal = rand();
	outs << randVal;
    }

    return outs.str().substr(0, 8) + "-" + outs.str().substr(8, 4) + "-" + outs.str().substr(12, 4) + "-" + outs.str().substr(16, 4) + "-" + outs.str().substr(20, 12);
}

std::string getHWAddr()
{
    const int MAC_LEN = 13;

    int sock_desc;
    char mac[MAC_LEN];

    if ((sock_desc = socket(PF_INET, SOCK_DGRAM, 0)) < 0) {
	std::cerr << "Error: could not create datagram socket." << std::endl;
	return std::string();
    }


    struct if_nameindex *ifn = if_nameindex();
    if (ifn == NULL) {
	std::cerr << "Error: if_nameindex failed." << std::endl;
	return std::string();
    }
    struct if_nameindex *ifn_ptr = ifn;

    int ret;
    struct ifreq ifr_mac; // interface name, interface information

    while ((ifn_ptr->if_index > 0) && (ifn_ptr->if_name != NULL)) {
	memset(&ifr_mac, 0, sizeof(ifr_mac));
	memcpy(&ifr_mac.ifr_name, ifn_ptr->if_name, IFNAMSIZ);

	if ((ret = ioctl(sock_desc, SIOCGIFHWADDR, &ifr_mac)) < 0) {
	    std::cerr << "Error: ioctl failed." << std::endl;
	    return std::string();
	}

 	struct sockaddr hwaddr = ifr_mac.ifr_hwaddr;
	unsigned char *ptr = (unsigned char *) &hwaddr.sa_data[0];

	// use the first non-zero mac address
	unsigned int result = 0;
	for (int i = 0; i < 6; i++) {
	    result |= *(ptr + i);
	    if (result > 0) {
	      break;
	    }
	}
	if (result == 0) {
 	    ifn_ptr++;
 	    continue;
	} else {
	    snprintf(mac, MAC_LEN,
		"%02x%02x%02x%02x%02x%02x\n",
		*ptr, *(ptr + 1), *(ptr + 2),
		*(ptr + 3), *(ptr + 4), *(ptr + 5));
	    break;
	}
    }
    // if no mac address found?

    if_freenameindex(ifn);

    close(sock_desc);
    return std::string(mac, 0, MAC_LEN);
}
