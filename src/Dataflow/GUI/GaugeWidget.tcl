catch {rename GaugeWidget ""}

itcl_class GaugeWidget {
    inherit BaseWidget
    constructor {config} {
	BaseWidget::constructor
	set name GaugeWidget
    }
}
