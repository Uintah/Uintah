

#include <iostream.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termio.h>
#include <bstring.h>

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

#define BIT7 0x80
#define BIT6 0x40
#define BIT5 0x20
#define BIT4 0x10
#define BIT3 0x08
#define BIT2 0x04
#define BIT1 0x02
#define BIT0 0x01

void get_report(int fd)
{
    char buf[16];
    int nread=0;
    char b1=get_one(0, fd, buf, &nread, 16);
    if(!(b1&BIT7)){
	cerr << "Framing error...\n";
	return;
    }
    if(b1&BIT6)
	cout << "FRINGE ";
    if(b1&BIT5)
	cout << "OUT_OF_RANGE ";
    if(b1&BIT4)
	cout << "P ";
    if(b1&BIT3)
	cout << "S ";
    if(b1&BIT2)
	cout << "L ";
    if(b1&BIT1)
	cout << "M ";
    if(b1&BIT0)
	cout << "R ";
    char b2=get_one(1, fd, buf, &nread, 16);
    char b3=get_one(2, fd, buf, &nread, 16);
    char b4=get_one(3, fd, buf, &nread, 16);
    int xdist=(b2<<14)|(b3<<7)|b4;
    cout << "X:" << xdist << " ";
    char b5=get_one(4, fd, buf, &nread, 16);
    char b6=get_one(5, fd, buf, &nread, 16);
    char b7=get_one(6, fd, buf, &nread, 16);
    int ydist=(b5<<14)|(b6<<7)|b7;
    cout << "Y:" << ydist << " ";
    char b8=get_one(7, fd, buf, &nread, 16);
    char b9=get_one(8, fd, buf, &nread, 16);
    char b10=get_one(9, fd, buf, &nread, 16);
    int zdist=(b8<<14)|(b9<<7)|b10;
    cout << "Z:" << zdist << " ";

    char b11=get_one(10, fd, buf, &nread, 16);
    char b12=get_one(11, fd, buf, &nread, 16);
    int pitch=(b11<<7)|b12;
    cout << "PITCH:" << pitch << " ";
    char b13=get_one(12, fd, buf, &nread, 16);
    char b14=get_one(13, fd, buf, &nread, 16);
    int yaw=(b13<<7)|b14;
    cout << "YAW:" << yaw << " ";
    char b15=get_one(14, fd, buf, &nread, 16);
    char b16=get_one(15, fd, buf, &nread, 16);
    int roll=(b15<<7)|b16;
    cout << "ROLL:" << roll << endl;
}
    

main()
{
    int fd=open("/dev/ttyd1", O_RDWR);

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
    cerr << "Sending reset\n";
    buf[0]='*'; buf[1]='R';
    write(fd, buf, 2);
    sleep(1);
    cerr << "Sending diagnostics\n";
    buf[0]='*'; buf[1]=0x05;
    write(fd, buf, 2);
    int n=0;
    get_one(1, fd, buf, &n, 2);
    if(buf[0] != 0xbf || buf[1] != 0x3f){
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
//    buf[0]='*'; buf[1]='D';
//    write(fd, buf, 2);
    buf[0]='*'; buf[1]='D';
    write(fd, buf, 2);
    for(int i=0;i<1000;i++){
//	buf[0]='*'; buf[1]='d';
//	write(fd, buf, 2);
	get_report(fd);
    }
}
