#!/bin/bash 

#load common setting
eval 'source $BASE_DIR/geco_commons/shell/shell_config.conf'

function usage() {
cat <<_EOT_
Usage:
  $0 [-u user] [-s symbol] [-i filename] [-m model name] arg1 ...

Description:
  Launch aleister operation

Options:
  -e   mode( 'prepro','train','gtrain','rpredict', 'deploy_model')
  -u   user
  -s   symbol
  -i   model id
  -m   model name
_EOT_
exit 1
}

SOURCE=shell
CONFIGSOURCE=ini
CONFIGMODE=default


if [ "$OPTIND" = 1 ]; then
  while getopts e:u:s:i:m:h OPT
  do
    case $OPT in
      e)
        MODE=$OPTARG
        ;;
      u)
        USER=$OPTARG
        ;;
      s)
        SYMBOL=$OPTARG
        ;;
      i)
        ID=$OPTARG
        ;;
      m)
        MODEL=$OPTARG
        ;;
      h)
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

shift $((OPTIND - 1))

# prepro 
if [ $(( $# & 1 )) -eq 1 ]; then 
  echo odd period options
  exit 0
fi
if [ $# -ge 2 ]; then 
  train_start_date=$1
  train_end_date=$2
  train_period_opt="--train_start_date ${train_start_date} --train_end_date ${train_end_date}"
fi
if [ $# -ge 4 ]; then 
  valid_start_date=$3
  valid_end_date=$4
  valid_period_opt="--valid_start_date ${valid_start_date} --valid_end_date ${valid_end_date}"
fi
if [ $# -ge 6 ]; then 
  test_start_date=$5
  test_end_date=$6
  test_period_opt="--test_start_date ${test_start_date} --test_end_date ${test_end_date}"
fi
source deactivate
source activate py37
python_interpritor=python
execute_path=`dirname $(pwd)`
execute_path="${execute_path}/src"
cd ${execute_path}
command="${python_interpritor} master.py -u ${USER} -s ${SOURCE} -sym ${SYMBOL} -mode ${MODE} -cs ${CONFIGSOURCE} -cm ${CONFIGMODE} -id ${ID} -mn ${MODEL} "
command="${command} ${train_period_opt} ${valid_period_opt} ${test_period_opt} " 
echo $command
eval $command
