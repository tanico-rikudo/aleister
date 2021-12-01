
#!/bin/bash

set -euxo pipefail
. cat /shell_config.conf

function usage() {
cat <<_EOT_
Usage:
  $0 [-a] [-b] [-f filename] arg1 ...

Description:
  Launch aleister operation

Options:
  -s   symbol
  -p    prepro
  -l    learn/train
  -i   model id
  -n  model name
  -
_EOT_
exit 1
}


if [ "$OPTIND" = 1 ]; then
  while getopts a:b:f:h: OPT
  do
    case $OPT in
      s)
        symbol=$OPTARG
        echo "FLAG_A is $FLAG_A"            # for debug
        ;;
      e)
        FLAG_B="on"
        echo "FLAG_B is $FLAG_B"            # for debug
        ;;
      i)
        ARG_F=$OPTARG
        echo "ARG_F is $ARG_F"              # for debug
        ;;
      n)
        echo "h option. display help"       # for debug
        usage
        ;;
      \?)
        echo "Try to enter the h option." 1>&2
        ;;
    esac
  done
else
  echo "No installed getopts-command." 1>&2
  exit 1
fi
