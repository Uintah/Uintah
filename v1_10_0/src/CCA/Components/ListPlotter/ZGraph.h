#include <vector>
#include <qwidget.h>

class ZGraph : public QWidget
{
    Q_OBJECT
public:
    ZGraph( QWidget *parent = 0, const char *name = 0);
    void setData(const double *val, int size);	

public slots:
    void refresh();
    void setStyle(bool style);	
protected:
    virtual void paintEvent(QPaintEvent *);	

private:
	bool style; //0 symbol , 1 symbol+line.
	std::vector<double> val;
};
