
#include <Devices/TrackerServer.h>

#include <iostream.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termio.h>
#ifdef __sun
#include <string.h>
#define bzero(p,sz)  memset(p,0,sz)
#else
#ifdef linux
#include <string.h>
#else
#include <bstring.h>
#endif
#endif
#include <sys/types.h>
#include <sys/time.h>

static TrackerData current;
static int initialized;
static int mouse_fd;
static int head_fd;

char get_one(int which, int fd, char* buf, int* nread, int max)
{
    while(which >= *nread){
	int s=read(fd, buf+(*nread), max-(*nread));
	if(s < 0){
	    perror("read");
	    exit(-1);
	}
	(*nread)+=s;
    }
    return buf[which];
}

#define BIT21 (1<<21)
#define BIT20 (1<<20)
#define BIT14 (1<<14)
#define BIT13 (1<<13)
#define BIT7 0x80
#define BIT6 0x40
#define BIT5 0x20
#define BIT4 0x10
#define BIT3 0x08
#define BIT2 0x04
#define BIT1 0x02
#define BIT0 0x01

static int open_device(char* name)
{
    int fd=open(name, O_RDWR);
    if(fd == -1){
	perror("open");
	cerr << "Cannot open " << name << endl;
	exit(-1);
    }

    struct termio tio;
    if(ioctl(fd, TCGETA, &tio) < 0){
	perror("ioctl");
	exit(-1);
    }
    
    bzero(tio.c_cc, sizeof(tio.c_cc));
    tio.c_cc[VMIN]=0;
    tio.c_cc[VTIME]=1;

    tio.c_iflag=0;
    tio.c_oflag=0;
    tio.c_cflag=B19200|CS8|CREAD;
    tio.c_lflag=0;
    if(ioctl(fd, TCSETA, &tio) < 0){
	perror("ioctl");
	exit(-1);
    }

    char buf[2];
    int retry=2;
 again:
    int flush=2;  // Flush input and output...
    if(ioctl(fd, TCFLSH, flush) < 0){
	perror("ioctl");
	exit(-1);
    }

    // Send reset command
    cerr << "Sending reset command\n";
    buf[0]='*'; buf[1]='R';
    write(fd, buf, 2);
    sleep(1);
    buf[0]='*'; buf[1]=0x05;
    write(fd, buf, 2);
    int nread=0;
    int done=0;
    while(!done){
	switch(read(fd, buf, 2)){
	case -1:
	    perror("read");
	    exit(-1);
	case 0:
	    if(nread++ > 10){
		cerr << "Command timed out...\n";
		nread=0;
		if(retry){
		    retry--;
		    goto again;
		}
	    }
	    break; // Try again...
	case 1:
	    cerr << "Read got fouled up??\n";
	    if(retry){
		retry--;
		goto again;
	    }
	    break;
	case 2:
	    done=1;
	}
    }
	    
    if(buf[0] != 0xbf || buf[1] != 0x3f){
	if(retry){
	    retry--;
	    goto again;
	}
	cerr << "Diagnostics failed: \n";
	fprintf(stderr, "GOT: %x %x\n", buf[0], buf[1]);
	if(!(buf[0]&BIT0))
	    cerr << "Control Unit test\n";
	if(!(buf[0]&BIT1))
	    cerr << "Processor test\n";
	if(!(buf[0]&BIT2))
	    cerr << "EPROM checksum test\n";
	if(!(buf[0]&BIT3))
	    cerr << "RAM test\n";
	if(!(buf[0]&BIT4))
	    cerr << "Transmitter test\n";
	if(!(buf[0]&BIT5))
	    cerr << "Receiver test\n";
	if(!(buf[1]&BIT0))
	    cerr << "Serial port test\n";
	if(!(buf[1]&BIT1))
	    cerr << "EEPROM test\n";
	exit(-1);
    }
    buf[0]='*'; buf[1]='A';
    write(fd, buf, 2);
    buf[0]='*'; buf[1]='S';
    write(fd, buf, 2);
    return fd;
}

static int initialize_tracker()
{
    mouse_fd=open_device("/dev/ttyd1");
    if(mouse_fd != -1)
	cerr << "Flying mouse successfully initialized\n";
    head_fd=open_device("/dev/ttyd2");
    if(head_fd != -1)
	cerr << "Head tracker successfully initialized\n";
    initialized=1;
    if(mouse_fd == -1 && head_fd == -1)
	return 0;
    else
	return 1;
}

static int get_report(int fd, TrackerPosition& pos)
{
    char buf[16];
    int nread=0;
    char b1=get_one(0, fd, buf, &nread, 16);
    if(!(b1&BIT7)){
	cerr << "Framing error...\n";
	while(!(b1&BIT7)){
	    for(int i=0;i<15;i++)
		buf[i]=buf[i+1];
	    nread--;
	    b1=get_one(0, fd, buf, &nread, 16);
	}
    }
    TrackerPosition newpos;
    newpos.fringe=(b1&BIT6);
    newpos.out=(b1&BIT5);
    newpos.s=(b1&BIT3);
    newpos.l=(b1&BIT2);
    newpos.m=(b1&BIT1);
    newpos.r=(b1&BIT0);
    char b2=get_one(1, fd, buf, &nread, 16);
    char b3=get_one(2, fd, buf, &nread, 16);
    char b4=get_one(3, fd, buf, &nread, 16);
    newpos.x=(b2<<14)|(b3<<7)|b4;
    if(newpos.x&BIT20) // extend the sign...
	newpos.x=newpos.x-BIT21;
    char b5=get_one(4, fd, buf, &nread, 16);
    char b6=get_one(5, fd, buf, &nread, 16);
    char b7=get_one(6, fd, buf, &nread, 16);
    newpos.y=(b5<<14)|(b6<<7)|b7;
    if(newpos.y&BIT20) // extend the sign...
	newpos.y=newpos.y-BIT21;
    char b8=get_one(7, fd, buf, &nread, 16);
    char b9=get_one(8, fd, buf, &nread, 16);
    char b10=get_one(9, fd, buf, &nread, 16);
    newpos.z=(b8<<14)|(b9<<7)|b10;
    if(newpos.z&BIT20) // extend the sign...
	newpos.z=newpos.z-BIT21;

    char b11=get_one(10, fd, buf, &nread, 16);
    char b12=get_one(11, fd, buf, &nread, 16);
    newpos.pitch=(b11<<7)|b12;
    if(newpos.pitch&BIT13) // extend the sign...
	newpos.pitch=newpos.pitch-BIT14;
    char b13=get_one(12, fd, buf, &nread, 16);
    char b14=get_one(13, fd, buf, &nread, 16);
    newpos.yaw=(b13<<7)|b14;
    if(newpos.yaw&BIT13) // extend the sign...
	newpos.yaw=newpos.yaw-BIT14;
    char b15=get_one(14, fd, buf, &nread, 16);
    char b16=get_one(15, fd, buf, &nread, 16);
    newpos.roll=(b15<<7)|b16;
    if(newpos.roll&BIT13) // extend the sign...
	newpos.roll=newpos.roll-BIT14;
    if(newpos != pos){
	pos=newpos;
	return 1;
    } else {
	return 0;
    }
}

int TrackerPosition::operator!=(const TrackerPosition& t)
{
    return x!=t.x || y!=t.y || z!=t.z || pitch!=t.pitch
	|| yaw!=t.yaw || roll!=t.roll || out!=t.out
	    || fringe!=t.fringe || s!=t.s || l!=t.l
		|| m!=t.m || r!=t.r;
}

    
int GetTrackerData(TrackerData& data)
{
    if(!initialized){
	if(!initialize_tracker())
	    return 0;
	// Make sure that the first time, we register a change
	current.head_pos.x=-1;
	current.mouse_pos.x=-1;
    }
    int max=mouse_fd;
    if(head_fd > max)
	max=head_fd;
    fd_set readfds;
    int changed=0;
    // The first time, block.  After that, return if we don't have anthing more...
    struct timeval timeout;
    timeout.tv_sec=0;
    timeout.tv_usec=0;
    struct timeval* timeoutp=0;
    int ntry=0;
    current.head_moved=0;
    current.mouse_moved=0;
    while(ntry < 5){
	if(changed)
	    ntry++;
	FD_ZERO(&readfds);
	if(mouse_fd != -1)
	    FD_SET(mouse_fd, &readfds);
	if(head_fd != -1)
	    FD_SET(head_fd, &readfds);
	switch(select(max+1, &readfds, 0, 0, timeoutp)){
	case -1:
	    perror("select");
	    exit(-1);
	case 0:
	    if(changed){
		data=current;
		return 1;
	    }
	    break;
	default:
	    if(mouse_fd > -1 && FD_ISSET(mouse_fd, &readfds)){
		int c = get_report(mouse_fd, current.mouse_pos);
		changed |= c;
		current.mouse_moved|=c;
	    }
	    if(head_fd > -1 && FD_ISSET(head_fd, &readfds)){
		int c = get_report(head_fd, current.head_pos);
		changed |= c;
		current.head_moved|=c;
	    }
	    timeoutp=&timeout;
	}
    }
    data=current;
    return 1;
}
