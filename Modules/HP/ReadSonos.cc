/*
 *  ReadSonos.cc: ScalarField Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarField.h>
#include <Modules/HP/ScalarFieldHP.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>
#include <fstream.h>
#include <ctype.h>

class ReadSonos : public Module {
    ScalarFieldOPort* outport;
    TCLstring filename;
    ScalarFieldHandle handle;
    clString old_filename;
public:
    ReadSonos(const clString& id);
    ReadSonos(const ReadSonos&, int deep=0);
    virtual ~ReadSonos();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_ReadSonos(const clString& id)
{
    return scinew ReadSonos(id);
}
}

ReadSonos::ReadSonos(const clString& id)
: Module("ReadSonos", id, Source), filename("filename", id, this)
{
    // Create the output data handle and port
    outport=scinew ScalarFieldOPort(this, "Output Data", ScalarFieldIPort::Atomic);
    add_oport(outport);
}

ReadSonos::ReadSonos(const ReadSonos& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
    NOT_FINISHED("ReadSonos::ReadSonos");
}

ReadSonos::~ReadSonos()
{
}

Module* ReadSonos::clone(int deep)
{
    return scinew ReadSonos(*this, deep);
}

static unsigned long read_hex(int ndigits, istream& in)
{
    unsigned long res=0;
    for(int i=0;i<ndigits;i++){
	int dig=in.get();
	if(dig==' '){
	    dig=0;
	} else if(dig>='0' && dig <='9'){
	    dig-='0';
	} else if(dig>='a' && dig <='f'){
	    dig-='a';
	    dig+=10;
	} else if(dig>='A' && dig <='F'){
	    dig-='A';
	    dig+=10;
	} else {
	    cerr << "Invalid digit: " << (char)dig << '\n';
	    exit(1);
	}
	res=res*16+dig;
    }
    return res;
}

static void read_byte(int want, istream& in)
{
    int byte=in.get();
    if(byte != want){
	cerr << "Expected: " << want << ", got " << byte << '\n';
	exit(1);
    }
}

static void read_crlf(istream& in)
{
    read_byte('\r', in);
    read_byte('\n', in);
}

static void read_string(char* str, int len, istream& in)
{
    for(int i=0;i<len;i++){
	str[i]=in.get();
    }
    str[len]=0;
}

static char* app_tagname(unsigned long n)
{
    switch(n){
    case 0:
	return "(DSR)";
    case 1:
	return "(Stress)";
    case 2:
	return "(Acoustic Densitometry (AD))";
    case 3:
	return "(3D)";
    default:
	return "(UNKNOWN APP)";
    }
}

static char* filetype_tagname(unsigned long n)
{
    switch(n){
    case 0:
	return ": Still frame";
    case 1:
	return ": CLR (loop) memory";
    case 4:
	return ": Analysis report (still frame format)";
    case 5:
	return ": Study";
    case 6:
	return ": 3D rotate loop";
    case 7:
	return ": 3D summary loop";
    default:
	return ": UNKNOWN";
    }
}

static char* screenformat_tagname(unsigned long n)
{
    switch(n){
    case 0:
	return ": Quad Screen (half resolution horizontally and vertically) ";
    case 1:
	return ": Quarter Screen (half width - cropped, half height - cropped)";
    case 2:
	return ": Vertical Split Screen (half width - cropped, full height)";
    case 3:
	return ": Horizontal Split Screen (full width, half height- cropped)";
    case 4:
	return ": Full Screen";
    default:
	return ": UNKNOWN";
    }
}

static char* mapfamily_tagname(unsigned long n)
{
    switch(n){
    case 0:
	return ": System Black and White only";
    case 1:
	return ": System Color Flow";
    case 2:
	return ": System Color Flow with 6-Bit Velocity";
    case 3:
	return ": VCR Playback, Black and White only";
    case 4:
	return ": VCR Playback, Color";
    case 5:
	return ": AQ";
    case 6:
	return ": AQ-IBS";
    case 7:
	return ": Color Kinesis";
    case 8:
	return ": Color Velocity Imaging";
    case 9:
	return ": Angio";
    case 10:
	return ": System Color Flow with 8-Bit Velocity";
    default:
	return ": DEFAULT";
    }
}

void ReadSonos::execute()
{
    clString fn(filename.get());
    if(!handle.get_rep() || fn != old_filename){
	old_filename=fn;
	clString dbname(fn+"/hpsonos.db");
	ifstream in(dbname());
	unsigned long hdr_size=read_hex(8, in);
	cerr << "hdr_size=" << hdr_size << '\n';
	read_crlf(in);
	unsigned long nextfree=read_hex(8, in);
	cerr << "nextfree=" << nextfree << '\n';
	read_crlf(in);
	unsigned long nentries=read_hex(4, in);
	cerr << "nentries=" << nentries << '\n';
	read_crlf(in);
	unsigned long nholes=read_hex(4, in);
	cerr << "nholes=" << nholes << '\n';
	read_crlf(in);
	read_byte('\f', in);
	int now=in.tellg();
	if(hdr_size < now){
	    cerr << "Header too big!\n";
	    exit(1);
	} else if(hdr_size > now){
	    char subdir_name[12];
	    read_string(subdir_name, 11, in);
	    cerr << "subdir: " << subdir_name << '\n';
	    read_crlf(in);
	    char patientID[62];
	    read_string(patientID, 61, in);
	    cerr << "patient ID: " << patientID << '\n';
	    read_crlf(in);
	    char datetime[20];
	    read_string(datetime, 19, in);
	    read_crlf(in);
	    char ApplicInfo[121];
	    read_string(ApplicInfo, 120, in);
	    cerr << "Application info: " << ApplicInfo << '\n';
	    read_crlf(in);
	    unsigned long appID=read_hex(4, in);
	    cerr << "Application ID: " << appID << app_tagname(appID) << '\n';
	    read_crlf(in);
	    unsigned long filetype=read_hex(4, in);
	    cerr << "FileType: " << filetype << filetype_tagname(filetype) << '\n';
	    read_crlf(in);
	    read_byte('\f', in);
	    int now=in.tellg();
	    if(hdr_size < now){
		cerr << "Header went wrong!\n";
		cerr << "size: " << hdr_size << '\n';
		cerr << "have: " << now << '\n';
		exit(1);
	    } else if(hdr_size > now){
		unsigned long skip=hdr_size-now;
		cerr << "Skipping " << skip << " bytes of header\n";
		for(int i=0;i<skip;i++)
		    in.get();
	    }
	} else {
	    cerr << "Skipping extended header\n";
	}
	ScalarFieldHP* sfield=new ScalarFieldHP();
	int cur=in.tellg();
	for(int i=0;i<nentries;i++){
	    cerr << "\nEntry " << i << '\n';
	    unsigned long length=read_hex(4, in);
	    cerr << "length=" << length << '\n';
	    read_crlf(in);
	    char revision[9];
	    read_string(revision, 8, in);
	    cerr << "revision=" << revision << '\n';
	    read_crlf(in);
	    char filename[12];
	    read_string(filename, 11, in);
	    cerr << "filename=" << filename << '\n';
	    read_crlf(in);
	    char patientID[61];
	    read_string(patientID, 61, in);
	    cerr << "patientID=" << patientID << '\n';
	    read_crlf(in);
	    char datetime[20];
	    read_string(datetime, 19, in);
	    cerr << "Date/time=" << datetime << '\n';
	    read_crlf(in);
	    char AppInfo[121];
	    read_string(AppInfo, 120, in);
	    cerr << "AppInfo=" << AppInfo << '\n';
	    read_crlf(in);
	    unsigned long appID=read_hex(4, in);
	    cerr << "AppID=" << appID << app_tagname(appID) << '\n';
	    read_crlf(in);
	    unsigned long filetype=read_hex(4, in);
	    cerr << "filetype=" << filetype << filetype_tagname(filetype) << '\n';
	    read_crlf(in);
	    unsigned long screenfmt=read_hex(4, in);
	    cerr << "screenfmt=" << screenfmt << screenformat_tagname(screenfmt) << '\n';
	    read_crlf(in);
	    unsigned long mapfamily=read_hex(4, in);
	    cerr << "mapfamily=" << mapfamily << mapfamily_tagname(mapfamily) << '\n';
	    read_crlf(in);
	    unsigned long framecount=read_hex(4, in);
	    cerr << "framecount=" << framecount << '\n';
	    read_crlf(in);
	    read_byte('\f', in);
	    int now=in.tellg();
	    if(cur+length < now){
		cerr << "Header went wrong!\n";
		cerr << "size: " << hdr_size << '\n';
		cerr << "have: " << now << '\n';
		exit(1);
	    } else if(cur+length > now){
		unsigned long skip=cur+length-now;
		cerr << "Skipping " << skip << " bytes of record\n";
		for(int i=0;i<skip;i++)
		    in.get();
	    }
	    if(appID==3 && filetype==1){
		for(int i=0;i<11;i++)
		    filename[i]=tolower(filename[i]);
		clString f(filename);
		clString file(fn+"/"+f.substr(0, 8)+"."+f.substr(8, 3));
		cerr << "Reading file: " << file << '\n';
		sfield->read_image(file);
	    } else {
		cerr << "Skipping file: " << filename << '\n';
	    }
	    cur+=length;
	}
	handle=sfield;
    }
    outport->send(handle);
}
