#Sample bash script to automate data download via PRADAN. 
#Windows users may install wget.exe and write a batch script in the same lines.
#Prequisites:  	1. have the list of data files (with path) ready in a file eg: ohrcFileList.txt 
# 		2. Login to Pradan in your browser and copy over the cookies in below script
#		3. Update the urlPrefix and payload variables for the desired payload
#Caution: There are session download limits, request rate limit and session timeouts in place, etc.
#	Violations may lead to blocking. Use script to ease the manual efforts but do not load the server.

dataFilePaths=ohrcFileList.txt
cookies="JSESSIONID=5c8f32114d083536d214818d2af4; OAuth_Token_Request_State=feba92b8-3c10-4b3d-8265-8d18f65b2044"
urlPrefix="https://pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection"
payload="ohrc"
#proxyOptions are required if your organization uses proxy to connect to Internet.
#proxyOptions="-e use_proxy=yes -e https_proxy=127.0.0.1:8080"
proxyOptions=""

for file in $(cat $dataFilePaths); 
do 
	echo $file; 
	wget $proxyOptions --content-disposition --tries=1 --no-cookies --header "Cookie: $cookies" $urlPrefix/$file?$payload;
	if [ $? -ne 0 ]; then
		echo "Error: Limits reached, terminating without downloading $file. You may login again later and reconfigure script to resume downloads." 
		exit -1;
	fi
done
