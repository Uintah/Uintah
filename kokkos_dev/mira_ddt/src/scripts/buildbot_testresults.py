import os,sys
from shutil import copytree, ignore_patterns

copytree(sys.argv[1], 'buildbot-results/' + sys.argv[1], ignore=ignore_patterns('*.uda*','susdir','goldStandard','CheckPoints','inputs','replace_all_GS','tools','TestScripts'))


exit
