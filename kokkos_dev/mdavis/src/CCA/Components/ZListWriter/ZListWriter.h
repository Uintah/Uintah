
#ifndef SCIRun_CCA_Component_ZListWriter_h
#define SCIRun_CCA_Component_ZListWriter_h

#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/PIDL/PIDL.h>

class ZListWriter;

class ZLUIPort : public virtual sci::cca::ports::UIPort {
public:
    ZLUIPort(const sci::cca::Services::pointer& svc) { services = svc; }
    virtual ~ZLUIPort(){}
    virtual int ui();
    void setParent(ZListWriter *com) { this->com = com; }
private:
    ZListWriter *com;       
    sci::cca::Services::pointer services;
};

class ZListPort : public virtual sci::cca::ports::ZListPort {
public:
    ZListPort(){};  
    virtual ~ZListPort(){};
    virtual SSIDL::array1<double> getList();
    void setParent(ZListWriter *com){this->com=com;}
private:
    ZListWriter *com;       
};

class ZLComponentIcon : public virtual sci::cca::ports::ComponentIcon {
public:
  virtual ~ZLComponentIcon() {}

  virtual std::string getDisplayName();
  virtual std::string getDescription();
  virtual std::string getIconShape();
  virtual int getProgressBar();
  void setParent(ZListWriter *com) { this->com = com; }
  ZListWriter *com;
  static const int STEPS = 10;
};


class ZListWriter : public sci::cca::Component {
public:
    std::vector<double> datalist;   
    ZListWriter();
    virtual ~ZListWriter();

    virtual void setServices(const sci::cca::Services::pointer& svc);

private:
    ZListWriter(const ZListWriter&);
    ZListWriter& operator=(const ZListWriter&);
    ZLUIPort *uiPort;
    ZLComponentIcon ciPort;
    sci::cca::Services::pointer services;
};

#endif
