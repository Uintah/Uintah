DEFAULT_TARGET ?= all

default:
	cd $(TOP) && $(MAKE) --print-directory $(DEFAULT_TARGET)

%:
	cd $(TOP) && $(MAKE) --print-directory $(DEFAULT_TARGET) $@
