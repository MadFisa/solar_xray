#!/bin/bash
#Sample bash script to automate data download via PRADAN.
#The script expect a file with dates in YYYYmmdd format in each line as input.

## Modified version for XSM data download

#Prequisites:  
#       1. Decide the dates for which you need to download the data (give as a range) 
#       2. Login to Pradan in your browser and copy over the cookies in below script
#       3. Update the urlPrefix and payload variables for the desired payload
#   Caution: There are session download limits, request rate limit and session timeouts in place, etc.
#   Violations may lead to blocking. Use script to ease the manual efforts but do not load the server.


## Replace this with cookies from your login session in browser
cookies="JSESSIONID=e347f3b3e8e2381634a3b125b8eb; OAuth_Token_Request_State=ad7099ef-9f1c-47a4-a417-3532f19a316e"

urlPrefix="https://pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/xsm_collection"
payload="xsm"

#proxyOptions are required if your organization uses proxy to connect to Internet.
#proxyOptions="-e use_proxy=yes -e https_proxy=https://username:password@proxyserverIP:Port"
proxyOptions=""

while IFS= read -r cdate; do

#    cdate=`date +%Y/%m/%d -d "${start_date}+${iday} days"`
    dd=$(date -d "$cdate" '+%d')
    mm=$(date -d "$cdate" '+%m')
    yyyy=$(date -d "$cdate" '+%Y')

    file=ch2_xsm_${yyyy}${mm}${dd}_v1.zip

    echo $file;
    wget $proxyOptions --content-disposition --tries=1 --no-cookies -N --header "Cookie: $cookies" $urlPrefix/auto/$yyyy/$file?$payload;

    if [ $? -ne 0 ]; then

        wget $proxyOptions --content-disposition --tries=1 --no-cookies -N --header "Cookie: $cookies" $urlPrefix/$file?$payload;

        if [ $? -ne 0 ]; then

            echo "Error: Limits reached, terminating without downloading $file. You may login again later and reconfigure script to resume downloads." 
            exit -1;
        fi
    fi
done < "$1"

